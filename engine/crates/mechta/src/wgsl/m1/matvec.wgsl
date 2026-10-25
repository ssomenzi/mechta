// M1 version (one thread per block), kept for comparisons with M2. Same as before.
struct P { rows: u32, k: u32, nx: u32, with_id: u32, x_step_i: u32 }
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

fn block_q8_0(base: u32, xo: u32) -> f32 {
  let d = h16(base); var s = 0.0;
  for (var i: u32 = 0u; i < 32u; i++) { s += d * s8(by(base + 2u + i)) * x[xo + i]; }
  return s;
}

fn block_q4_k(base: u32, xo: u32) -> f32 {
  let d = h16(base); let dmin = h16(base + 2u); let sc = base + 4u; let qs = base + 16u;
  var s = 0.0;
  for (var j: u32 = 0u; j < 4u; j++) {
    let a = scale_min(sc, 2u * j); let c = scale_min(sc, 2u * j + 1u);
    let d1 = d * f32(a.x); let m1 = dmin * f32(a.y);
    let d2 = d * f32(c.x); let m2 = dmin * f32(c.y);
    for (var l: u32 = 0u; l < 32u; l++) {
      let q = by(qs + 32u * j + l);
      s += (d1 * f32(q & 0xFu) - m1) * x[xo + 64u * j + l];
      s += (d2 * f32(q >> 4u) - m2) * x[xo + 64u * j + 32u + l];
    }
  }
  return s;
}

fn block_q6_k(base: u32, xo: u32) -> f32 {
  let ql = base; let qh = base + 128u; let sc = base + 192u; let d = h16(base + 208u);
  var s = 0.0;
  for (var n: u32 = 0u; n < 2u; n++) {
    for (var l: u32 = 0u; l < 32u; l++) {
      let is = l / 16u;
      let lq = ql + 64u * n; let hq = by(qh + 32u * n + l); let sg = sc + 8u * n;
      let q1 = f32(i32((by(lq + l) & 0xFu)       | (((hq >> 0u) & 3u) << 4u)) - 32);
      let q2 = f32(i32((by(lq + l + 32u) & 0xFu) | (((hq >> 2u) & 3u) << 4u)) - 32);
      let q3 = f32(i32((by(lq + l) >> 4u)        | (((hq >> 4u) & 3u) << 4u)) - 32);
      let q4 = f32(i32((by(lq + l + 32u) >> 4u)  | (((hq >> 6u) & 3u) << 4u)) - 32);
      let o = xo + 128u * n;
      s += d * s8(by(sg + is))      * q1 * x[o + l];
      s += d * s8(by(sg + is + 2u)) * q2 * x[o + l + 32u];
      s += d * s8(by(sg + is + 4u)) * q3 * x[o + l + 64u];
      s += d * s8(by(sg + is + 6u)) * q4 * x[o + l + 96u];
    }
  }
  return s;
}

fn reduce(t: u32, v: f32) -> f32 {
  partial[t] = v;
  workgroupBarrier();
  for (var m: u32 = 32u; m > 0u; m >>= 1u) {
    if (t < m) { partial[t] += partial[t + m]; }
    workgroupBarrier();
  }
  return partial[0];
}

fn row(kind: u32, wid: vec3<u32>, t: u32) {
  let r = wid.x + wid.y * p.nx;
  if (r >= p.rows) { return; }
  let z = wid.z;
  var mat = z; if (p.with_id == 1u) { mat = ids[z]; }
  let xo = z * p.x_step_i;
  var per = 32u; var bb = 34u;
  if (kind == 1u) { per = 256u; bb = 144u; }
  if (kind == 2u) { per = 256u; bb = 210u; }
  let nb = p.k / per;
  let base = (mat * p.rows + r) * nb * bb;
  var s = 0.0;
  if (kind == 3u) {
    let w0 = (mat * p.rows + r) * p.k;
    for (var i = t; i < p.k; i += 64u) { s += bitcast<f32>(raw[w0 + i]) * x[xo + i]; }
  } else {
    for (var b = t; b < nb; b += 64u) {
      if (kind == 0u) { s += block_q8_0(base + b * bb, xo + b * per); }
      else if (kind == 1u) { s += block_q4_k(base + b * bb, xo + b * per); }
      else { s += block_q6_k(base + b * bb, xo + b * per); }
    }
  }
  let tot = reduce(t, s);
  if (t == 0u) { y[z * p.rows + r] = tot; }
}

@compute @workgroup_size(64)
fn mv_q8_0(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(0u, wid, t); }
@compute @workgroup_size(64)
fn mv_q4_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(1u, wid, t); }
@compute @workgroup_size(64)
fn mv_q6_k(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(2u, wid, t); }
@compute @workgroup_size(64)
fn mv_f32(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) { row(3u, wid, t); }
