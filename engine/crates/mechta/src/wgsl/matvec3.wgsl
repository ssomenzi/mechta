// y = W x for the file's formats. Reduction in shared memory. Decoding as in dequant.wgsl (verified against ggml).
// Experts: with p.with_id = 1 workgroup z uses matrix ids[z] (W = 256 matrices in a row) and the
// vector x starting at z * p.x_step_i. Rows beyond 65535: row = wid.x + wid.y * p.nx.
// M2, variant 3: for Q4_K. The bench says Q4_K stops at 58 GB/s where Q8_0 does 70-76,
// even with 151 MB per dispatch: the bottleneck is the work per byte, not the size. Here a Q4_K piece does:
// 1 word for d/dmin, 3 words for the scales, 8 words of nibbles, 8 vec4 reads of x, and the computation
// is d1 * (q . x) - m1 * (sum x) instead of 32 times (d1 q - m1) x. x is read as vec4: each piece
// starts at a multiple of 32 values and x_step_i is 0 or 512, so the indices are always aligned to 4.
struct P { rows: u32, k: u32, nx: u32, with_id: u32, x_step_i: u32, tpr: u32 }
@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> x4: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<uniform> p: P;

var<workgroup> partial: array<f32, 64>;

fn by(i: u32) -> u32 { return (raw[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu; }
fn h16(i: u32) -> f32 { return unpack2x16float(by(i) | (by(i + 1u) << 8u)).x; }
fn s8(b: u32) -> f32 { return f32((i32(b << 24u)) >> 24u); }
fn xf(i: u32) -> f32 { return x4[i >> 2u][i & 3u]; }
fn bw(w: u32, j: u32) -> u32 { return (w >> (8u * j)) & 0xFFu; }   // byte j of word w

// Q8_0: the piece is the block (34 bytes). As in matvec.wgsl.
fn piece_q8_0(base: u32, xo: u32) -> f32 {
  let d = h16(base); var s = 0.0;
  for (var i: u32 = 0u; i < 32u; i++) { s += d * s8(by(base + 2u + i)) * xf(xo + i); }
  return s;
}

// Q4_K: block of 256 = 8 pieces; piece pz = 2j + half: values 64j + 32 half + l.
// The block (144 bytes) is aligned to 4: d, dmin = word 0; scales (12 bytes) = words 1..3; qs = words 4..35.
// Scales: piece pz < 4: sc = byte pz & 63, min = byte pz+4 & 63;
//         pz >= 4: sc = (byte pz+4 & 0xF) | (high bits of byte pz-4), min = (byte pz+4 >> 4) | (high bits of byte pz).
fn piece_q4_k(base: u32, pz: u32, xo4: u32) -> f32 {
  let w0 = base >> 2u;
  let dd = unpack2x16float(raw[w0]);
  let s0 = raw[w0 + 1u]; let s1 = raw[w0 + 2u]; let s2 = raw[w0 + 3u];
  var sc: u32; var mn: u32;
  if (pz < 4u) { sc = bw(s0, pz) & 63u; mn = bw(s1, pz) & 63u; }
  else { let q = pz - 4u; sc = (bw(s2, q) & 0xFu) | ((bw(s0, q) >> 6u) << 4u); mn = (bw(s2, q) >> 4u) | ((bw(s1, q) >> 6u) << 4u); }
  let d1 = dd.x * f32(sc); let m1 = dd.y * f32(mn);
  let wq = w0 + 4u + 8u * (pz / 2u); let sh = 4u * (pz % 2u);
  var sq = 0.0; var sx = 0.0;
  for (var q4: u32 = 0u; q4 < 8u; q4++) {
    let w = raw[wq + q4] >> sh;
    let q = vec4<f32>(f32(w & 0xFu), f32((w >> 8u) & 0xFu), f32((w >> 16u) & 0xFu), f32((w >> 24u) & 0xFu));
    let xv = x4[xo4 + q4];
    sq += dot(q, xv); sx += dot(xv, vec4<f32>(1.0));
  }
  return d1 * sq - m1 * sx;
}

// Q6_K: as in matvec.wgsl.
fn piece_q6_k(base: u32, pz: u32, xo: u32) -> f32 {
  let ql = base; let qh = base + 128u; let sc = base + 192u; let d = h16(base + 208u);
  let n = pz / 4u; let c = pz % 4u;
  let lq = ql + 64u * n + 32u * (c % 2u); let sh = 4u * (c / 2u);
  var s = 0.0;
  for (var l: u32 = 0u; l < 32u; l++) {
    let q = f32(i32(((by(lq + l) >> sh) & 0xFu) | (((by(qh + 32u * n + l) >> (2u * c)) & 3u) << 4u)) - 32);
    s += d * s8(by(sc + 8u * n + l / 16u + 2u * c)) * q * xf(xo + l);
  }
  return s;
}

// type: 0 = Q8_0, 1 = Q4_K, 2 = Q6_K, 3 = F32
fn row(kind: u32, wid: vec3<u32>, t: u32) {
  let tpr = p.tpr; let rpw = 64u / tpr;
  let r = (wid.x + wid.y * p.nx) * rpw + t / tpr; let lane = t % tpr;
  let valid = r < p.rows;
  let z = wid.z;
  var mat = z; if (p.with_id == 1u) { mat = ids[z]; }
  let xo = z * p.x_step_i;
  let u = p.k / 32u;
  var s = 0.0;
  if (valid) {
    let rr = mat * p.rows + r;
    for (var i = lane; i < u; i += tpr) {
      let x0 = xo + 32u * i;
      if (kind == 0u) { s += piece_q8_0((rr * u + i) * 34u, x0); }
      else if (kind == 1u) { s += piece_q4_k((rr * (u / 8u) + i / 8u) * 144u, i % 8u, x0 >> 2u); }
      else if (kind == 2u) { s += piece_q6_k((rr * (u / 8u) + i / 8u) * 210u, i % 8u, x0); }
      else {
        let w0 = rr * p.k + 32u * i;
        for (var l: u32 = 0u; l < 32u; l++) { s += bitcast<f32>(raw[w0 + l]) * xf(x0 + l); }
      }
    }
  }
  partial[t] = s;
  workgroupBarrier();
  for (var m = tpr / 2u; m > 0u; m >>= 1u) {
    if (lane < m) { partial[t] += partial[t + m]; }
    workgroupBarrier();
  }
  if (valid && lane == 0u) { y[z * p.rows + r] = partial[t]; }
}

@compute @workgroup_size(64)
fn mv_q8_0(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(0u, wid, t); }
@compute @workgroup_size(64)
fn mv_q4_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(1u, wid, t); }
@compute @workgroup_size(64)
fn mv_q6_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(2u, wid, t); }
@compute @workgroup_size(64)
fn mv_f32(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(3u, wid, t); }
