// M4: y = W X with X of n columns (n = 1..4 tokens), the weights read ONLY ONCE for all columns.
// Used to verify MTP drafts: n tokens go through the model reading the weights as for one token.
// Same structure as matvec4.wgsl (one thread per piece of 32 values, reduction in shared memory), with:
//   p.n columns; column c of x starts at c * p.x_tok (+ z * p.x_step_i for the experts);
//   y[(c * nz + z) * rows + r]: for each column the same layout as today.
// Weight decoding identical to matvec4 (Q8_0 and Q6_K as realigned words, Q4_K as vec4): verified against the CPU in the bench.
struct P { rows: u32, k: u32, nx: u32, with_id: u32, x_step_i: u32, tpr: u32, n: u32, x_tok: u32 }
@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<uniform> p: P;

const NMAX: u32 = 4u;
var<workgroup> partial: array<array<f32, 64>, 4>;

fn by(i: u32) -> u32 { return (raw[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu; }
fn h16(i: u32) -> f32 { return unpack2x16float(by(i) | (by(i + 1u) << 8u)).x; }
fn s8(b: u32) -> f32 { return f32((i32(b << 24u)) >> 24u); }
fn bw(w: u32, j: u32) -> u32 { return (w >> (8u * j)) & 0xFFu; }
fn word(w: u32, k: u32, s: u32) -> u32 { return (raw[w + k] >> s) | ((raw[w + k + 1u] << (31u - s)) << 1u); }

// Q8_0: block of 34 bytes (d f16 + 32 int8), aligned to 2.
fn piece_q8_0(base: u32, xo4: ptr<function, array<u32, 4>>, acc: ptr<function, array<f32, 4>>) {
  let wd = raw[base >> 2u]; let sd = 8u * (base & 3u);
  let d = unpack2x16float((wd >> sd) & 0xFFFFu).x;
  let q = base + 2u; let w = q >> 2u; let s = 8u * (q & 3u);
  var a: array<f32, 4>;
  for (var k: u32 = 0u; k < 8u; k++) {
    let qv = vec4<f32>(unpack4xI8(word(w, k, s)));
    for (var c: u32 = 0u; c < NMAX; c++) { if (c < p.n) { a[c] += dot(qv, x4[(*xo4)[c] + k]); } }
  }
  for (var c: u32 = 0u; c < NMAX; c++) { (*acc)[c] += d * a[c]; }
}

// Q4_K: block of 144 bytes = 8 pieces; piece pz = 2j + half (low/high nibbles of bytes qs[32j..32j+32)).
fn scale_q4(s0: u32, s1: u32, s2: u32, pz: u32) -> vec2<u32> {
  if (pz < 4u) { return vec2<u32>(bw(s0, pz) & 63u, bw(s1, pz) & 63u); }
  let q = pz - 4u;
  return vec2<u32>((bw(s2, q) & 0xFu) | ((bw(s0, q) >> 6u) << 4u), (bw(s2, q) >> 4u) | ((bw(s1, q) >> 6u) << 4u));
}
fn piece_q4_k(base: u32, pz: u32, xo4: ptr<function, array<u32, 4>>, acc: ptr<function, array<f32, 4>>) {
  let w0 = base >> 2u;
  let dd = unpack2x16float(raw[w0]);
  let a = scale_q4(raw[w0 + 1u], raw[w0 + 2u], raw[w0 + 3u], pz);
  let d1 = dd.x * f32(a.x); let m1 = dd.y * f32(a.y);
  let wq = w0 + 4u + 8u * (pz / 2u); let sh = 4u * (pz % 2u);
  var sq: array<f32, 4>; var sx: array<f32, 4>;
  for (var k: u32 = 0u; k < 8u; k++) {
    let qv = vec4<f32>(unpack4xU8((raw[wq + k] >> sh) & 0x0F0F0F0Fu));
    for (var c: u32 = 0u; c < NMAX; c++) { if (c < p.n) { let xv = x4[(*xo4)[c] + k]; sq[c] += dot(qv, xv); sx[c] += dot(xv, vec4<f32>(1.0)); } }
  }
  for (var c: u32 = 0u; c < NMAX; c++) { (*acc)[c] += d1 * sq[c] - m1 * sx[c]; }
}

// Q6_K: block of 210 bytes = 8 pieces (see matvec4.wgsl).
fn piece_q6_k(base: u32, pz: u32, xo4: ptr<function, array<u32, 4>>, acc: ptr<function, array<f32, 4>>) {
  let n = pz / 4u; let c4 = pz % 4u;
  let d = h16(base + 208u);
  let s0 = s8(by(base + 192u + 8u * n + 2u * c4)); let s1 = s8(by(base + 193u + 8u * n + 2u * c4));
  let ql = base + 64u * n + 32u * (c4 % 2u); let wl = ql >> 2u; let sl = 8u * (ql & 3u); let sh = 4u * (c4 / 2u);
  let qh = base + 128u + 32u * n; let wh = qh >> 2u; let shh = 8u * (qh & 3u); let sc2 = 2u * c4;
  var a0: array<f32, 4>; var a1: array<f32, 4>;
  for (var k: u32 = 0u; k < 8u; k++) {
    let lo = (word(wl, k, sl) >> sh) & 0x0F0F0F0Fu;
    let hi = ((word(wh, k, shh) >> sc2) & 0x03030303u) << 4u;
    let qv = vec4<f32>(unpack4xU8(lo | hi)) - vec4<f32>(32.0);
    for (var c: u32 = 0u; c < NMAX; c++) { if (c < p.n) { let v = dot(qv, x4[(*xo4)[c] + k]); if (k < 4u) { a0[c] += v; } else { a1[c] += v; } } }
  }
  for (var c: u32 = 0u; c < NMAX; c++) { (*acc)[c] += d * (s0 * a0[c] + s1 * a1[c]); }
}

// F32: 32 values = 8 vec4 (weights read as words, bitcast).
fn piece_f32(w0: u32, xo4: ptr<function, array<u32, 4>>, acc: ptr<function, array<f32, 4>>) {
  for (var k: u32 = 0u; k < 8u; k++) {
    let qv = vec4<f32>(bitcast<f32>(raw[w0 + 4u * k]), bitcast<f32>(raw[w0 + 4u * k + 1u]), bitcast<f32>(raw[w0 + 4u * k + 2u]), bitcast<f32>(raw[w0 + 4u * k + 3u]));
    for (var c: u32 = 0u; c < NMAX; c++) { if (c < p.n) { (*acc)[c] += dot(qv, x4[(*xo4)[c] + k]); } }
  }
}

// type: 0 = Q8_0, 1 = Q4_K, 2 = Q6_K, 3 = F32
fn row(kind: u32, wid: vec3<u32>, t: u32) {
  let tpr = p.tpr; let rpw = 64u / tpr;
  let r = (wid.x + wid.y * p.nx) * rpw + t / tpr; let lane = t % tpr;
  let valid = r < p.rows;
  let z = wid.z;
  var mat = z; if (p.with_id == 1u) { mat = ids[z]; }
  let u = p.k / 32u;
  var acc: array<f32, 4>;
  if (valid) {
    let rr = mat * p.rows + r;
    for (var i = lane; i < u; i += tpr) {
      var xo4: array<u32, 4>;
      for (var c: u32 = 0u; c < NMAX; c++) { xo4[c] = (c * p.x_tok + z * p.x_step_i + 32u * i) >> 2u; }
      if (kind == 0u) { piece_q8_0((rr * u + i) * 34u, &xo4, &acc); }
      else if (kind == 1u) { piece_q4_k((rr * (u / 8u) + i / 8u) * 144u, i % 8u, &xo4, &acc); }
      else if (kind == 2u) { piece_q6_k((rr * (u / 8u) + i / 8u) * 210u, i % 8u, &xo4, &acc); }
      else { piece_f32(rr * p.k + 32u * i, &xo4, &acc); }
    }
  }
  for (var c: u32 = 0u; c < NMAX; c++) { partial[c][t] = acc[c]; }
  workgroupBarrier();
  for (var m = tpr / 2u; m > 0u; m >>= 1u) {
    if (lane < m) { for (var c: u32 = 0u; c < NMAX; c++) { partial[c][t] += partial[c][t + m]; } }
    workgroupBarrier();
  }
  if (valid && lane == 0u) {
    for (var c: u32 = 0u; c < NMAX; c++) { if (c < p.n) { y[c * p.rows * num_z() + z * p.rows + r] = partial[c][t]; } }
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
