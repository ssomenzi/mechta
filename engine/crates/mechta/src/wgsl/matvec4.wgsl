// y = W x for the formats of the file. Reduction in shared memory. Decoding as dequant.wgsl (verified against ggml).
// Experts: with p.with_id = 1 workgroup z uses matrix ids[z] (W = 256 matrices in a row) and the
// vector x starting at z * p.x_step_i. Rows beyond 65535: row = wid.x + wid.y * p.nx.
// M2, variant 4: matvec3 (Q4_K as vec4) + new Q8_0. The benches (experts, q4k3) say
// that Q8_0 does 70-76 GB/s on the bench and 60-66 in the engine, against the ~83 of the read-only kernel: here a Q8_0 piece
// reads 9 words (the 34-byte block is not aligned to 4: the 32 quants start at +2, i.e. at 0 or at 2 within the word),
// realigns them with a branchless funnel shift, and unpacks 4 int8 per word; x as vec4. Two unpackings:
// mv_q8_0 with unpack4xI8 (WGSL builtin), mv_q8_0b by hand (shifts and masks). mv_read only reads: it is the ceiling.
// Q6_K with the same method (realigned words, unpack4xU8); mv_read6 is the ceiling with the bytes of Q6_K.
// Bench q8: 75 → 83 GB/s on attn_qkv (= the ceiling), 54 → 79 on ssm_out (ceiling 81); the two versions are equal.
// Below, the original note of variant 3, for Q4_K. The bench says that Q4_K stops at 58 GB/s where Q8_0 does 70-76,
// even with 151 MB per dispatch: the brake is the work per byte, not the size. Here a Q4_K piece does:
// 1 word for d/dmin, 3 words for the scales, 8 words of nibbles, 8 vec4 reads of x, and the computation
// is d1 * (q . x) - m1 * (sum x) instead of 32 times (d1 q - m1) x. x is read as vec4: each piece
// starts at a multiple of 32 values and x_step_i is 0 or 512, so the indices are always aligned to 4.
struct P { rows: u32, k: u32, nx: u32, with_id: u32, x_step_i: u32, tpr: u32, x_off: u32, y_off: u32 }   // x_off/y_off: M4, second token
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

// Q8_0: the piece is the block (34 bytes): d (f16) in bytes 0-1, the 32 int8 quants in bytes 2..33.
// q = base + 2 is even: q % 4 is 0 or 2. Word k of the quants = (raw[w+k] >> s) | (raw[w+k+1] << (32 - s)) with
// s = 8 (q % 4): for s = 0 the second term must vanish, and "<< 32" is undefined in WGSL: it is written
// (hi << (31 - s)) << 1, which for s = 0 gives 0 and for s = 16 gives hi << 16.
fn d_q8(base: u32) -> f32 {
  let w = raw[base >> 2u]; let s = 8u * (base & 3u);   // base even: s = 0 or 16
  return unpack2x16float((w >> s) & 0xFFFFu).x;
}
fn word_q8(w: u32, k: u32, s: u32) -> u32 {
  return (raw[w + k] >> s) | ((raw[w + k + 1u] << (31u - s)) << 1u);
}
fn i8x4(w: u32) -> vec4<f32> {
  return vec4<f32>(f32(i32(w << 24u) >> 24u), f32(i32(w << 16u) >> 24u), f32(i32(w << 8u) >> 24u), f32(i32(w) >> 24u));
}
fn piece_q8_0(base: u32, xo4: u32) -> f32 {
  let d = d_q8(base);
  let q = base + 2u; let w = q >> 2u; let s = 8u * (q & 3u);
  var acc = 0.0;
  for (var k: u32 = 0u; k < 8u; k++) { acc += dot(vec4<f32>(unpack4xI8(word_q8(w, k, s))), x4[xo4 + k]); }
  return d * acc;
}
fn piece_q8_0b(base: u32, xo4: u32) -> f32 {
  let d = d_q8(base);
  let q = base + 2u; let w = q >> 2u; let s = 8u * (q & 3u);
  var acc = 0.0;
  for (var k: u32 = 0u; k < 8u; k++) { acc += dot(i8x4(word_q8(w, k, s)), x4[xo4 + k]); }
  return d * acc;
}
// Read only: the 8 aligned words of the piece, summed as f32. Measures the bandwidth ceiling in the same geometry.
fn piece_read(base: u32) -> f32 {
  let w = (base + 2u) >> 2u; var acc = 0.0;
  for (var k: u32 = 0u; k < 8u; k++) { acc += bitcast<f32>(raw[w + k]); }
  return acc;
}

// Q4_K: block of 256 = 8 pieces; piece pz = 2j + half: values 64j + 32 half + l.
// The block (144 bytes) is aligned to 4: d, dmin = word 0; scales (12 bytes) = words 1..3; qs = words 4..35.
// Scales: piece pz < 4: sc = byte pz & 63, min = byte pz+4 & 63;
//        pz >= 4: sc = (byte pz+4 & 0xF) | (high bits of byte pz-4), min = (byte pz+4 >> 4) | (high bits of byte pz).
// Q4_1 (Flint INT4): block = 5 words: d|m (f16 pair), then 16 bytes; low nibbles are values 0..15,
// high nibbles 16..31. Sum = d (q . x) + m (sum x).
fn piece_q4_1(w0: u32, xo4: u32) -> f32 {
  let dm = unpack2x16float(raw[w0]);
  var sq = 0.0; var sx = 0.0;
  for (var q4: u32 = 0u; q4 < 4u; q4++) {
    let w = raw[w0 + 1u + q4];
    let xl = x4[xo4 + q4]; let xh = x4[xo4 + 4u + q4];
    sq += dot(vec4<f32>(unpack4xU8(w & 0x0F0F0F0Fu)), xl) + dot(vec4<f32>(unpack4xU8((w >> 4u) & 0x0F0F0F0Fu)), xh);
    sx += dot(xl + xh, vec4<f32>(1.0));
  }
  return dm.x * sq + dm.y * sx;
}
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

// Q4_K in pieces of 64: the two pieces of 32 of the same half j (low and high nibbles) read the SAME 32 bytes
// of qs: with one thread per 64 values the 8 words of qs and the 4 words of d/dmin/scales are read only once instead
// of twice. Half the reads per byte compared to piece_q4_k; x read as before (vec4). u = k / 64 pieces per row.
fn scale_q4(s0: u32, s1: u32, s2: u32, pz: u32) -> vec2<u32> {
  if (pz < 4u) { return vec2<u32>(bw(s0, pz) & 63u, bw(s1, pz) & 63u); }
  let q = pz - 4u;
  return vec2<u32>((bw(s2, q) & 0xFu) | ((bw(s0, q) >> 6u) << 4u), (bw(s2, q) >> 4u) | ((bw(s1, q) >> 6u) << 4u));
}
fn piece_q4_k64(base: u32, j: u32, xo4: u32) -> f32 {
  let w0 = base >> 2u;
  let dd = unpack2x16float(raw[w0]);
  let s0 = raw[w0 + 1u]; let s1 = raw[w0 + 2u]; let s2 = raw[w0 + 3u];
  let a_lo = scale_q4(s0, s1, s2, 2u * j); let a_hi = scale_q4(s0, s1, s2, 2u * j + 1u);
  let d_lo = dd.x * f32(a_lo.x); let m_lo = dd.y * f32(a_lo.y);
  let d_hi = dd.x * f32(a_hi.x); let m_hi = dd.y * f32(a_hi.y);
  let wq = w0 + 4u + 8u * j;
  var sq_lo = 0.0; var sx_lo = 0.0; var sq_hi = 0.0; var sx_hi = 0.0;
  for (var k: u32 = 0u; k < 8u; k++) {
    let w = raw[wq + k];
    let xl = x4[xo4 + k]; let xh = x4[xo4 + 8u + k];
    sq_lo += dot(vec4<f32>(unpack4xU8(w & 0x0F0F0F0Fu)), xl); sx_lo += dot(xl, vec4<f32>(1.0));
    sq_hi += dot(vec4<f32>(unpack4xU8((w >> 4u) & 0x0F0F0F0Fu)), xh); sx_hi += dot(xh, vec4<f32>(1.0));
  }
  return (d_lo * sq_lo - m_lo * sx_lo) + (d_hi * sq_hi - m_hi * sx_hi);
}

// Q6_K: block of 256 = 8 pieces, 210 bytes: ql[128] (low nibbles), qh[64] (2 high bits), sc[16] (int8), d (f16 at +208).
// Piece pz: n = pz / 4 (half of 128), quarter c = pz % 4: values 128n + 32c + l. Low bits from ql[64n + 32 (c % 2) + l],
// high nibble if c >= 2; high bits from qh[32n + l] bit 2c; scale sc[8n + 2c + l/16] (two scales: l < 16 and l >= 16).
// 210 % 4 = 2: the block is aligned to 2 like Q8_0, so ql and qh of the piece (32 contiguous bytes each) are read
// as words with the same funnel shift. Per word: 4 values = (low nibbles) | (2 high bits << 4), then unpack4xU8 - 32.
fn piece_q6_k(base: u32, pz: u32, xo4: u32) -> f32 {
  let n = pz / 4u; let c = pz % 4u;
  let d = h16(base + 208u);
  let s0 = s8(by(base + 192u + 8u * n + 2u * c)); let s1 = s8(by(base + 193u + 8u * n + 2u * c));
  let ql = base + 64u * n + 32u * (c % 2u); let wl = ql >> 2u; let sl = 8u * (ql & 3u); let sh = 4u * (c / 2u);
  let qh = base + 128u + 32u * n; let wh = qh >> 2u; let shh = 8u * (qh & 3u); let sc2 = 2u * c;
  var acc0 = 0.0; var acc1 = 0.0;
  for (var k: u32 = 0u; k < 8u; k++) {
    let lo = (word_q8(wl, k, sl) >> sh) & 0x0F0F0F0Fu;
    let hi = ((word_q8(wh, k, shh) >> sc2) & 0x03030303u) << 4u;
    let q = vec4<f32>(unpack4xU8(lo | hi)) - vec4<f32>(32.0);
    let v = dot(q, x4[xo4 + k]);
    if (k < 4u) { acc0 += v; } else { acc1 += v; }
  }
  return d * (s0 * acc0 + s1 * acc1);
}
// Read only with the bytes of Q6_K: 8 words per piece, 26 bytes per piece (8 x 26 = 208 of the 210 of the block).
fn piece_read6(base: u32, pz: u32) -> f32 {
  let w = (base + 26u * pz) >> 2u; var acc = 0.0;
  for (var k: u32 = 0u; k < 8u; k++) { acc += bitcast<f32>(raw[w + k]); }
  return acc;
}

// Q3_K: decoding in q3k.wgsl, appended by engine.rs. kind 8.
fn piece_q3_k(base: u32, pz: u32, xo4: u32) -> f32 {
  let s = q3_scales(base, pz);
  var a0 = 0.0; var a1 = 0.0;
  for (var k: u32 = 0u; k < 8u; k++) {
    let v = dot(q3_vals(base, pz, k), x4[xo4 + k]);
    if (k < 4u) { a0 += v; } else { a1 += v; }
  }
  return q3_d(base) * (s.x * a0 + s.y * a1);
}

// kind: 8 = Q3_K, 0 = Q8_0 (unpack4xI8), 1 = Q4_K, 2 = Q6_K, 3 = F32, 4 = Q8_0 by hand, 5 = read only (bytes as Q8_0), 6 = read only (bytes as Q6_K),
//       7 = Q4_K in pieces of 64 (u = k / 64, tpr = min(64, k / 64))
fn row(kind: u32, wid: vec3<u32>, t: u32) {
  let tpr = p.tpr; let rpw = 64u / tpr;
  let r = (wid.x + wid.y * p.nx) * rpw + t / tpr; let lane = t % tpr;
  let valid = r < p.rows;
  let z = wid.z;
  var mat = z; if (p.with_id == 1u) { mat = ids[z]; }
  let xo = z * p.x_step_i + p.x_off;
  var u = p.k / 32u; if (kind == 7u) { u = p.k / 64u; }
  var s = 0.0;
  if (valid) {
    let rr = mat * p.rows + r;
    for (var i = lane; i < u; i += tpr) {
      let x0 = xo + 32u * i;
      if (kind == 7u) { s += piece_q4_k64((rr * (u / 4u) + i / 4u) * 144u, i % 4u, (xo + 64u * i) >> 2u); }
      else if (kind == 0u) { s += piece_q8_0((rr * u + i) * 34u, x0 >> 2u); }
      else if (kind == 4u) { s += piece_q8_0b((rr * u + i) * 34u, x0 >> 2u); }
      else if (kind == 5u) { s += piece_read((rr * u + i) * 34u); }
      else if (kind == 1u) { s += piece_q4_k((rr * (u / 8u) + i / 8u) * 144u, i % 8u, x0 >> 2u); }
      else if (kind == 2u) { s += piece_q6_k((rr * (u / 8u) + i / 8u) * 210u, i % 8u, x0 >> 2u); }
      else if (kind == 6u) { s += piece_read6((rr * (u / 8u) + i / 8u) * 210u, i % 8u); }
      else if (kind == 8u) { s += piece_q3_k((rr * (u / 8u) + i / 8u) * 110u, i % 8u, x0 >> 2u); }
      else if (kind == 9u) { s += piece_q4_1((rr * u + i) * 5u, x0 >> 2u); }
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
  if (valid && lane == 0u) { y[p.y_off + z * p.rows + r] = partial[t]; }
}

@compute @workgroup_size(64)
fn mv_q8_0(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(0u, wid, t); }
@compute @workgroup_size(64)
fn mv_q4_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(1u, wid, t); }
@compute @workgroup_size(64)
fn mv_q6_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(2u, wid, t); }
@compute @workgroup_size(64)
fn mv_f32(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(3u, wid, t); }
@compute @workgroup_size(64)
fn mv_q3_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(8u, wid, t); }
@compute @workgroup_size(64)
fn mv_q4_1(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(9u, wid, t); }
