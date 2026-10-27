// k-token step (quad.rs): matvec of the experts IN UNION for up to 4 tokens. The 4 rows
// choose 8 experts each: each expert of the union is read once and computed for the columns of the rows that use it.
// `uniforms` (from `merge4`): [n, then per expert z: id, slot0, slot1, slot2, slot3], slot = position (0..7) in the
// row's list, 0xFFFF if the row does not use it. x of row t: t * x_tok + slot * x_step_i (gate/up: x_step_i 0;
// down: 512). y of row t: t * y_tok + slot * rows + r. vec4 accumulators, one component per row (decoders of
// matvec_nh.wgsl, x in f16).
struct P { rows: u32, k: u32, nx: u32, y_tok: u32, x_step_i: u32, tpr: u32, n: u32, x_tok: u32 }
@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> xh: array<vec2<u32>>;   // x in f16, 4 values per element
fn x4r(i: u32) -> vec4<f32> { let v = xh[i]; return vec4<f32>(unpack2x16float(v.x), unpack2x16float(v.y)); }
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> uniforms: array<u32>;
@group(0) @binding(4) var<uniform> p: P;

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

// Q3_K: decoding in q3k.wgsl, appended by engine.rs (q3_source). kind 3.
fn piece_q3_k(base: u32, pz: u32, xo4: vec4<u32>) -> vec4<f32> {
  let s = q3_scales(base, pz);
  var a0 = vec4<f32>(0.0); var a1 = vec4<f32>(0.0);
  for (var k: u32 = 0u; k < 8u; k++) {
    let v = dot4(q3_vals(base, pz, k), xo4, k);
    if (k < 4u) { a0 += v; } else { a1 += v; }
  }
  return q3_d(base) * (s.x * a0 + s.y * a1);
}

// kind: 0 = Q8_0, 1 = Q4_K, 2 = Q6_K, 3 = Q3_K. One workgroup per (group of rows, expert z of the union).
fn row(kind: u32, wid: vec3<u32>, t: u32) {
  let z = wid.z;
  if (z >= uniforms[0]) { return; }                       // uniform per workgroup: no thread passes the barriers
  let e = 1u + 5u * z;
  let id = uniforms[e];
  let sl = vec4<u32>(uniforms[e + 1u], uniforms[e + 2u], uniforms[e + 3u], uniforms[e + 4u]);
  let has = sl != vec4<u32>(0xFFFFu);
  let tpr = p.tpr; let rpw = 64u / tpr;
  let r = (wid.x + wid.y * p.nx) * rpw + t / tpr; let lane = t % tpr;
  let valid = r < p.rows;
  let u = p.k / 32u;
  var acc = vec4<f32>(0.0);
  let xb = vec4<u32>(0u, 1u, 2u, 3u) * p.x_tok + select(vec4<u32>(0u), sl, has) * p.x_step_i;
  if (valid) {
    let rr = id * p.rows + r;
    for (var i = lane; i < u; i += tpr) {
      let xo4 = (xb + vec4<u32>(32u * i)) >> vec4<u32>(2u);
      if (kind == 1u) { acc += piece_q4_k((rr * (u / 8u) + i / 8u) * 144u, i % 8u, xo4); }
      else if (kind == 0u) { acc += piece_q8_0((rr * u + i) * 34u, xo4); }
      else if (kind == 3u) { acc += piece_q3_k((rr * (u / 8u) + i / 8u) * 110u, i % 8u, xo4); }
      else { acc += piece_q6_k((rr * (u / 8u) + i / 8u) * 210u, i % 8u, xo4); }
    }
  }
  partial[t] = acc;
  workgroupBarrier();
  for (var m = tpr / 2u; m > 0u; m >>= 1u) {
    if (lane < m) { partial[t] += partial[t + m]; }
    workgroupBarrier();
  }
  if (valid && lane == 0u) {
    let v = partial[t];
    if (has.x) { y[sl.x * p.rows + r] = v.x; }
    if (has.y) { y[p.y_tok + sl.y * p.rows + r] = v.y; }
    if (has.z) { y[2u * p.y_tok + sl.z * p.rows + r] = v.z; }
    if (has.w) { y[3u * p.y_tok + sl.w * p.rows + r] = v.w; }
  }
}

@compute @workgroup_size(64)
fn mv_q4_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(1u, wid, t); }
@compute @workgroup_size(64)
fn mv_q6_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(2u, wid, t); }
@compute @workgroup_size(64)
fn mv_q3_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(3u, wid, t); }
@compute @workgroup_size(64)
fn mv_q8_0(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(0u, wid, t); }
