// M4: y = W X with X of n columns (n = 1..4 tokens), the weights read ONLY ONCE for all columns.
// Used to verify MTP drafts: n tokens go through the model reading the weights as for one token.
// Same structure as matvec4.wgsl (one thread per piece of 32 values, reduction in shared memory), with:
//   p.n columns; column c of x starts at c * p.x_tok (+ z * p.x_step_i for the experts);
//   y[(c * nz + z) * rows + r]: for each column the same layout as today.
// Weight decoding identical to matvec4 (Q8_0 and Q6_K as realigned words, Q4_K as vec4): verified against the CPU in the bench.
// VARIANT C (matvec_nh.wgsl): like B, but x is in f16 (4 values in one vec2<u32>): for n = 2 columns with k = 4096
// x takes 16 KB instead of 32 (the cache threshold seen in a column bench). The weights are the same.
// VARIANT B (matvec_n4.wgsl): vec4<f32> accumulators with one component per column, always 4 columns computed
// (the columns beyond p.n re-read column p.n - 1 and do not write): no runtime-indexed arrays and no branches
// per column. Used to see whether the growing cost of variant A (column bench) comes from the arrays in scratch memory.
struct P { rows: u32, k: u32, nx: u32, with_id: u32, x_step_i: u32, tpr: u32, n: u32, x_tok: u32 }
@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> xh: array<vec2<u32>>;   // x in f16, 4 values per element
fn x4r(i: u32) -> vec4<f32> { let v = xh[i]; return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y)); }
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<uniform> p: P;

const NMAX: u32 = 4u;
var<workgroup> partial: array<vec4<f32>, 64>;

fn by(i: u32) -> u32 { return (raw[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu; }
fn h16(i: u32) -> f32 { return unpack2x16float(by(i) | (by(i + 1u) << 8u)).x; }
fn s8(b: u32) -> f32 { return f32((i32(b << 24u)) >> 24u); }
fn bw(w: u32, j: u32) -> u32 { return (w >> (8u * j)) & 0xFFu; }
fn word(w: u32, k: u32, s: u32) -> u32 { return (raw[w + k] >> s) | ((raw[w + k + 1u] << (31u - s)) << 1u); }

// Q8_0: block of 34 bytes (d f16 + 32 int8), aligned to 2.
fn dot4(qv: vec4<f32>, xo4: vec4<u32>, k: u32) -> vec4<f32> {
  return vec4<f32>(dot(qv, x4r(xo4.x + k)), dot(qv, x4r(xo4.y + k)), dot(qv, x4r(xo4.z + k)), dot(qv, x4r(xo4.w + k)));
}
fn sum4(xo4: vec4<u32>, k: u32) -> vec4<f32> {
  let u = vec4<f32>(1.0);
  return vec4<f32>(dot(u, x4r(xo4.x + k)), dot(u, x4r(xo4.y + k)), dot(u, x4r(xo4.z + k)), dot(u, x4r(xo4.w + k)));
}
fn piece_q8_0(base: u32, xo4: vec4<u32>) -> vec4<f32> {
  let wd = raw[base >> 2u]; let sd = 8u * (base & 3u);
  let d = unpack2x16float((wd >> sd) & 0xFFFFu).x;
  let q = base + 2u; let w = q >> 2u; let s = 8u * (q & 3u);
  var a = vec4<f32>(0.0);
  for (var k: u32 = 0u; k < 8u; k++) { a += dot4(vec4<f32>(unpack4xI8(word(w, k, s))), xo4, k); }
  return d * a;
}

// Q4_K: block of 144 bytes = 8 pieces; piece pz = 2j + half (low/high nibbles of bytes qs[32j..32j+32)).
fn scale_q4(s0: u32, s1: u32, s2: u32, pz: u32) -> vec2<u32> {
  if (pz < 4u) { return vec2<u32>(bw(s0, pz) & 63u, bw(s1, pz) & 63u); }
  let q = pz - 4u;
  return vec2<u32>((bw(s2, q) & 0xFu) | ((bw(s0, q) >> 6u) << 4u), (bw(s2, q) >> 4u) | ((bw(s1, q) >> 6u) << 4u));
}
fn piece_q4_k(base: u32, pz: u32, xo4: vec4<u32>) -> vec4<f32> {
  let w0 = base >> 2u;
  let dd = unpack2x16float(raw[w0]);
  let a = scale_q4(raw[w0 + 1u], raw[w0 + 2u], raw[w0 + 3u], pz);
  let d1 = dd.x * f32(a.x); let m1 = dd.y * f32(a.y);
  let wq = w0 + 4u + 8u * (pz / 2u); let sh = 4u * (pz % 2u);
  var sq = vec4<f32>(0.0); var sx = vec4<f32>(0.0);
  for (var k: u32 = 0u; k < 8u; k++) {
    sq += dot4(vec4<f32>(unpack4xU8((raw[wq + k] >> sh) & 0x0F0F0F0Fu)), xo4, k); sx += sum4(xo4, k);
  }
  return d1 * sq - m1 * sx;
}

// Q6_K: block of 210 bytes = 8 pieces (see matvec4.wgsl).
fn piece_q6_k(base: u32, pz: u32, xo4: vec4<u32>) -> vec4<f32> {
  let n = pz / 4u; let c4 = pz % 4u;
  let d = h16(base + 208u);
  let s0 = s8(by(base + 192u + 8u * n + 2u * c4)); let s1 = s8(by(base + 193u + 8u * n + 2u * c4));
  let ql = base + 64u * n + 32u * (c4 % 2u); let wl = ql >> 2u; let sl = 8u * (ql & 3u); let sh = 4u * (c4 / 2u);
  let qh = base + 128u + 32u * n; let wh = qh >> 2u; let shh = 8u * (qh & 3u); let sc2 = 2u * c4;
  var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0);
  for (var k: u32 = 0u; k < 8u; k++) {
    let lo = (word(wl, k, sl) >> sh) & 0x0F0F0F0Fu;
    let hi = ((word(wh, k, shh) >> sc2) & 0x03030303u) << 4u;
    let v = dot4(vec4<f32>(unpack4xU8(lo | hi)) - vec4<f32>(32.0), xo4, k);
    if (k < 4u) { a0 += v; } else { a1 += v; }
  }
  return d * (s0 * a0 + s1 * a1);
}

// F32: 32 values = 8 vec4 (weights read as words, bitcast).
fn piece_f32(w0: u32, xo4: vec4<u32>) -> vec4<f32> {
  var a = vec4<f32>(0.0);
  for (var k: u32 = 0u; k < 8u; k++) {
    let qv = vec4<f32>(bitcast<f32>(raw[w0 + 4u * k]), bitcast<f32>(raw[w0 + 4u * k + 1u]), bitcast<f32>(raw[w0 + 4u * k + 2u]), bitcast<f32>(raw[w0 + 4u * k + 3u]));
    a += dot4(qv, xo4, k);
  }
  return a;
}

// type: 0 = Q8_0, 1 = Q4_K, 2 = Q6_K, 3 = F32
fn row(kind: u32, wid: vec3<u32>, t: u32) {
  let tpr = p.tpr; let rpw = 64u / tpr;
  let r = (wid.x + wid.y * p.nx) * rpw + t / tpr; let lane = t % tpr;
  let valid = r < p.rows;
  let z = wid.z;
  var mat = z; if (p.with_id == 1u) { mat = ids[z]; }
  let u = p.k / 32u;
  var acc = vec4<f32>(0.0);
  // columns: c beyond p.n - 1 re-read the last valid column (valid read, result discarded)
  let col = min(vec4<u32>(0u, 1u, 2u, 3u), vec4<u32>(p.n - 1u));
  if (valid) {
    let rr = mat * p.rows + r;
    for (var i = lane; i < u; i += tpr) {
      let xo4 = (col * p.x_tok + vec4<u32>(z * p.x_step_i + 32u * i)) >> vec4<u32>(2u);
      if (kind == 0u) { acc += piece_q8_0((rr * u + i) * 34u, xo4); }
      else if (kind == 1u) { acc += piece_q4_k((rr * (u / 8u) + i / 8u) * 144u, i % 8u, xo4); }
      else if (kind == 2u) { acc += piece_q6_k((rr * (u / 8u) + i / 8u) * 210u, i % 8u, xo4); }
      else { acc += piece_f32(rr * p.k + 32u * i, xo4); }
    }
  }
  partial[t] = acc;
  workgroupBarrier();
  for (var m = tpr / 2u; m > 0u; m >>= 1u) {
    if (lane < m) { partial[t] += partial[t + m]; }
    workgroupBarrier();
  }
  if (valid && lane == 0u) {
    let v = partial[t]; let nz = num_z();
    y[z * p.rows + r] = v.x;
    if (p.n > 1u) { y[1u * p.rows * nz + z * p.rows + r] = v.y; }
    if (p.n > 2u) { y[2u * p.rows * nz + z * p.rows + r] = v.z; }
    if (p.n > 3u) { y[3u * p.rows * nz + z * p.rows + r] = v.w; }
  }
}
// number of matrices per column (1, or 8 experts): y for column c starts at c * rows * num_z
fn num_z() -> u32 { return select(1u, 8u, p.with_id == 1u); }

@compute @workgroup_size(64)
fn mv_q8_0(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(0u, wid, t); }
@compute @workgroup_size(64)
fn mv_q4_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(1u, wid, t); }
@compute @workgroup_size(64)
fn mv_q6_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(2u, wid, t); }
@compute @workgroup_size(64)
fn mv_f32(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(3u, wid, t); }
