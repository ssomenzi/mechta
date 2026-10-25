// y = W x for the file's formats. Reduction in shared memory. Decoding as in dequant.wgsl (verified against ggml).
// Experts: with p.with_id = 1 workgroup z uses matrix ids[z] (W = 256 matrices in a row) and the
// vector x starting at z * p.x_step_i. Rows beyond 65535: row = wid.x + wid.y * p.nx.
// M2: one thread per piece of 32 values (before: one thread per block, 8 of 64 working with Q4_K).
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

// Q8_0: the piece is the block (34 bytes).
fn piece_q8_0(base: u32, xo: u32) -> f32 {
  let d = h16(base); var s = 0.0;
  for (var i: u32 = 0u; i < 32u; i++) { s += d * s8(by(base + 2u + i)) * x[xo + i]; }
  return s;
}

// Q4_K: block of 256 = 8 pieces; piece pz = 2j + half: values 64j + 32 half + l.
fn piece_q4_k(base: u32, pz: u32, xo: u32) -> f32 {
  let d = h16(base); let dmin = h16(base + 2u); let qs = base + 16u;
  let j = pz / 2u; let high = pz % 2u;
  let a = scale_min(base + 4u, pz);
  let d1 = d * f32(a.x); let m1 = dmin * f32(a.y);
  var s = 0.0;
  for (var l: u32 = 0u; l < 32u; l++) {
    let q = (by(qs + 32u * j + l) >> (4u * high)) & 0xFu;
    s += (d1 * f32(q) - m1) * x[xo + l];
  }
  return s;
}

// Q6_K: block of 256 = 8 pieces; piece pz: n = pz / 4 (half of 128), quarter c = pz % 4:
// values 128n + 32c + l. Low bits from ql[64n + l (+32 if c is odd)], high nibble if c >= 2;
// high bits from qh[32n + l] bit 2c; scale sc[8n + l/16 + 2c].
fn piece_q6_k(base: u32, pz: u32, xo: u32) -> f32 {
  let ql = base; let qh = base + 128u; let sc = base + 192u; let d = h16(base + 208u);
  let n = pz / 4u; let c = pz % 4u;
  let lq = ql + 64u * n + 32u * (c % 2u); let sh = 4u * (c / 2u);
  var s = 0.0;
  for (var l: u32 = 0u; l < 32u; l++) {
    let q = f32(i32(((by(lq + l) >> sh) & 0xFu) | (((by(qh + 32u * n + l) >> (2u * c)) & 3u) << 4u)) - 32);
    s += d * s8(by(sc + 8u * n + l / 16u + 2u * c)) * q * x[xo + l];
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
fn mv_f32(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(3u, wid, t); }
