// Decoding of the ggml formats to f32, one thread per block. Bytes read from array<u32> at any
// offset (the Q8_0 and Q6_K blocks are not aligned to 4). Order of operations as in
// ggml-quants.c, to get the same rounding. All loops have constant bounds.
struct P { nblocks: u32 }
@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;
@group(0) @binding(2) var<uniform> p: P;

fn by(i: u32) -> u32 { return (raw[i >> 2u] >> ((i & 3u) * 8u)) & 0xFFu; }
fn h16(i: u32) -> f32 { return unpack2x16float(by(i) | (by(i + 1u) << 8u)).x; }
fn s8(b: u32) -> f32 { return f32((i32(b << 24u)) >> 24u); }

@compute @workgroup_size(64)
fn q8_0(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x; if (b >= p.nblocks) { return; }
  let base = b * 34u; let d = h16(base);
  for (var i: u32 = 0u; i < 32u; i++) { out[b * 32u + i] = d * s8(by(base + 2u + i)); }
}

// Q4_1 (Flint INT4): block of 32 = d (f16), m (f16), 16 bytes; y[j] = (qs[j] & 15) d + m,
// y[j + 16] = (qs[j] >> 4) d + m, as dequantize_row_q4_1 of ggml.
@compute @workgroup_size(64)
fn q4_1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x; if (b >= p.nblocks) { return; }
  let base = b * 20u; let d = h16(base); let m = h16(base + 2u);
  for (var j: u32 = 0u; j < 16u; j++) {
    let q = by(base + 4u + j);
    out[b * 32u + j] = f32(q & 0xFu) * d + m;
    out[b * 32u + j + 16u] = f32(q >> 4u) * d + m;
  }
}

fn scale_min(s: u32, j: u32) -> vec2<u32> {
  if (j < 4u) { return vec2<u32>(by(s + j) & 63u, by(s + j + 4u) & 63u); }
  return vec2<u32>((by(s + j + 4u) & 0xFu) | ((by(s + j - 4u) >> 6u) << 4u), (by(s + j + 4u) >> 4u) | ((by(s + j) >> 6u) << 4u));
}

@compute @workgroup_size(64)
fn q4_k(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x; if (b >= p.nblocks) { return; }
  let base = b * 144u; let d = h16(base); let dmin = h16(base + 2u);
  let sc = base + 4u; let qs = base + 16u;
  for (var j: u32 = 0u; j < 4u; j++) {
    let a = scale_min(sc, 2u * j); let c = scale_min(sc, 2u * j + 1u);
    let d1 = d * f32(a.x); let m1 = dmin * f32(a.y);
    let d2 = d * f32(c.x); let m2 = dmin * f32(c.y);
    for (var l: u32 = 0u; l < 32u; l++) {
      let q = by(qs + 32u * j + l);
      out[b * 256u + 64u * j + l]       = d1 * f32(q & 0xFu) - m1;
      out[b * 256u + 64u * j + 32u + l] = d2 * f32(q >> 4u) - m2;
    }
  }
}

@compute @workgroup_size(64)
fn q6_k(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x; if (b >= p.nblocks) { return; }
  let base = b * 210u; let ql = base; let qh = base + 128u; let sc = base + 192u; let d = h16(base + 208u);
  for (var n: u32 = 0u; n < 2u; n++) {
    for (var l: u32 = 0u; l < 32u; l++) {
      let is = l / 16u;
      let lq = ql + 64u * n; let hq = by(qh + 32u * n + l); let s = sc + 8u * n;
      let q1 = f32(i32((by(lq + l) & 0xFu)        | (((hq >> 0u) & 3u) << 4u)) - 32);
      let q2 = f32(i32((by(lq + l + 32u) & 0xFu)  | (((hq >> 2u) & 3u) << 4u)) - 32);
      let q3 = f32(i32((by(lq + l) >> 4u)         | (((hq >> 4u) & 3u) << 4u)) - 32);
      let q4 = f32(i32((by(lq + l + 32u) >> 4u)   | (((hq >> 6u) & 3u) << 4u)) - 32);
      let o = b * 256u + 128u * n;
      out[o + l]       = d * s8(by(s + is))      * q1;
      out[o + l + 32u] = d * s8(by(s + is + 2u)) * q2;
      out[o + l + 64u] = d * s8(by(s + is + 4u)) * q3;
      out[o + l + 96u] = d * s8(by(s + is + 6u)) * q4;
    }
  }
}

// Q3_K: dequantize_row_q3_K of ggml-quants.c, line by line (one thread per block of 256, 110 bytes:
// hmask[32], qs[64], scales[12], d). An implementation independent of q3k.wgsl, to check the Q3_K kernels against.
@compute @workgroup_size(64)
fn q3_k(@builtin(global_invocation_id) gid: vec3<u32>) {
  let b = gid.x; if (b >= p.nblocks) { return; }
  let base = b * 110u; let hm = base; let qs = base + 32u; let sc = base + 96u;
  let d_all = h16(base + 108u);
  let a0 = by(sc) | (by(sc + 1u) << 8u) | (by(sc + 2u) << 16u) | (by(sc + 3u) << 24u);
  let a1 = by(sc + 4u) | (by(sc + 5u) << 8u) | (by(sc + 6u) << 16u) | (by(sc + 7u) << 24u);
  let tmp = by(sc + 8u) | (by(sc + 9u) << 8u) | (by(sc + 10u) << 16u) | (by(sc + 11u) << 24u);
  var aux = array<u32, 4>(
    (a0 & 0x0f0f0f0fu) | (((tmp >> 0u) & 0x03030303u) << 4u),
    (a1 & 0x0f0f0f0fu) | (((tmp >> 2u) & 0x03030303u) << 4u),
    ((a0 >> 4u) & 0x0f0f0f0fu) | (((tmp >> 4u) & 0x03030303u) << 4u),
    ((a1 >> 4u) & 0x0f0f0f0fu) | (((tmp >> 6u) & 0x03030303u) << 4u));
  var is = 0u; var m = 1u; var o = b * 256u;
  for (var n: u32 = 0u; n < 2u; n++) {
    var shift = 0u;
    for (var j: u32 = 0u; j < 4u; j++) {
      var dl = d_all * (s8((aux[is / 4u] >> (8u * (is % 4u))) & 0xFFu) - 32.0); is++;
      for (var l: u32 = 0u; l < 16u; l++) {
        let q = (by(qs + 32u * n + l) >> shift) & 3u;
        out[o] = dl * (f32(q) - select(4.0, 0.0, (by(hm + l) & m) != 0u)); o++;
      }
      dl = d_all * (s8((aux[is / 4u] >> (8u * (is % 4u))) & 0xFFu) - 32.0); is++;
      for (var l: u32 = 0u; l < 16u; l++) {
        let q = (by(qs + 32u * n + l + 16u) >> shift) & 3u;
        out[o] = dl * (f32(q) - select(4.0, 0.0, (by(hm + l + 16u) & m) != 0u)); o++;
      }
      shift += 2u; m <<= 1u;
    }
  }
}
