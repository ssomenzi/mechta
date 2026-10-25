// y = W x for the file's formats. Reduction in shared memory. Decoding as in dequant.wgsl (verified against ggml).
// Experts: with p.with_id = 1 workgroup z uses matrix ids[z] (W = 256 matrices in a row) and the
// vector x starting at z * p.x_step_i. Rows beyond 65535: row = wid.x + wid.y * p.nx.
// M2: one thread per piece of 32 values. Variant 2: bytes read as 4-byte words (see span).
struct P { rows: u32, k: u32, nx: u32, with_id: u32, x_step_i: u32, tpr: u32 }
@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<storage, read> ids: array<u32>;
@group(0) @binding(4) var<uniform> p: P;

var<workgroup> partial: array<f32, 64>;

fn by(i: u32) -> u32 { return (raw[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu; }
fn h16(i: u32) -> f32 { return unpack2x16float(by(i) | (by(i + 1u) << 8u)).x; }
fn s8(b: u32) -> f32 { return f32((i32(b << 24u)) >> 24u); }

fn scale_min(s: u32, j: u32) -> vec2<u32> {
  if (j < 4u) { return vec2<u32>(by(s + j) & 63u, by(s + j + 4u) & 63u); }
  return vec2<u32>((by(s + j + 4u) & 0xFu) | ((by(s + j - 4u) >> 6u) << 4u), (by(s + j + 4u) >> 4u) | ((by(s + j) >> 6u) << 4u));
}

// The work is split into "pieces" of 32 consecutive values of the row (U = k / 32 pieces per row).
// A row has p.tpr threads (min(64, U)); a workgroup of 64 threads does 64 / tpr rows.
// Each piece: same decoding as dequant.wgsl (verified against ggml), multiplied by x.

// M2, variant 2: the bytes of a piece are read as 4-byte words (9 words cover 32 bytes at
// any alignment), then extracted from registers. Before: one memory read per byte.
struct Span { w: array<u32, 9>, o: u32 }
fn span(byte0: u32) -> Span {
  var s: Span; let w0 = byte0 >> 2u; s.o = byte0 & 3u;
  for (var i: u32 = 0u; i < 9u; i++) { s.w[i] = raw[w0 + i]; }
  return s;
}
fn sb(s: ptr<function, Span>, k: u32) -> u32 { let j = (*s).o + k; return ((*s).w[j >> 2u] >> ((j & 3u) * 8u)) & 0xFFu; }

// Q8_0: the piece is the block (34 bytes): d in bytes 0-1, the 32 values in bytes 2..33.
fn piece_q8_0(base: u32, xo: u32) -> f32 {
  var s = span(base);
  let d = unpack2x16float(sb(&s, 0u) | (sb(&s, 1u) << 8u)).x;
  var acc = 0.0;
  for (var i: u32 = 0u; i < 32u; i++) { acc += d * s8(sb(&s, 2u + i)) * x[xo + i]; }
  return acc;
}

// Q4_K: block of 256 = 8 pieces; piece pz = 2j + half: values 64j + 32 half + l.
// The block (144 bytes) is aligned to 4: d, dmin = word 0; scales = words 1..3; qs = words 4..35.
fn piece_q4_k(base: u32, pz: u32, xo: u32) -> f32 {
  let w0 = base >> 2u;
  let dd = unpack2x16float(raw[w0]); let d = dd.x; let dmin = dd.y;
  let a = scale_min(base + 4u, pz);
  let d1 = d * f32(a.x); let m1 = dmin * f32(a.y);
  let j = pz / 2u; let sh = 4u * (pz % 2u);
  var acc = 0.0;
  for (var q4: u32 = 0u; q4 < 8u; q4++) {
    let wq = raw[w0 + 4u + 8u * j + q4];
    for (var b: u32 = 0u; b < 4u; b++) {
      let q = (wq >> (8u * b + sh)) & 0xFu;
      acc += (d1 * f32(q) - m1) * x[xo + 4u * q4 + b];
    }
  }
  return acc;
}

// Q6_K: block of 256 = 8 pieces; piece pz: n = pz / 4 (half of 128), quarter c = pz % 4:
// values 128n + 32c + l. Low bits from ql[64n + l (+32 if c is odd)], high nibble if c >= 2;
// high bits from qh[32n + l] bit 2c; scale sc[8n + l/16 + 2c]. Block of 210 bytes: not aligned to 4.
fn piece_q6_k(base: u32, pz: u32, xo: u32) -> f32 {
  let n = pz / 4u; let c = pz % 4u;
  var lq = span(base + 64u * n + 32u * (c % 2u));
  var hq = span(base + 128u + 32u * n);
  let sh = 4u * (c / 2u);
  let d = h16(base + 208u);
  let s0 = d * s8(by(base + 192u + 8u * n + 2u * c)); let s1 = d * s8(by(base + 192u + 8u * n + 1u + 2u * c));
  var acc = 0.0;
  for (var l: u32 = 0u; l < 32u; l++) {
    let q = f32(i32(((sb(&lq, l) >> sh) & 0xFu) | (((sb(&hq, l) >> (2u * c)) & 3u) << 4u)) - 32);
    var sc = s0; if (l >= 16u) { sc = s1; }
    acc += sc * q * x[xo + l];
  }
  return acc;
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
      else if (kind == 1u) { s += piece_q4_k((rr * (u / 8u) + i / 8u) * 144u, i % 8u, x0); }
      else if (kind == 2u) { s += piece_q6_k((rr * (u / 8u) + i / 8u) * 210u, i % 8u, x0); }
      else {
        let w0 = rr * p.k + 32u * i;
        for (var l: u32 = 0u; l < 32u; l++) { s += bitcast<f32>(raw[w0 + l]) * x[x0 + l]; }
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
