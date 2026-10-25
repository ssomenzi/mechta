// Q3_K, shared by matvec4 / matvec_u / matvec_uh / gemm (appended to them by engine.rs, `q3_source`).
// Block of 256 values = 8 pieces of 32, 110 bytes: hmask[32] at 0, qs[64] at 32, scales[12] at 96, d (f16) at 108.
// Piece pz: n = pz / 4 (half of 128), j = pz % 4: values 128n + 32j + l. Low 2 bits = qs[32n + l] >> 2j, high bit =
// hmask[l] bit (4n + j); value = low + 4 * high - 4. Scale is = 8n + 2j for l < 16, is + 1 for l >= 16: 6-bit scales
// unpacked from the 12 bytes as in ggml (dequantize_row_q3_K), minus 32. 110 % 4 = 2: blocks are aligned to 2 bytes
// like Q6_K, so the words are read with the same funnel shift.
fn q3_par(w: u32, k: u32, s: u32) -> u32 { return (raw[w + k] >> s) | ((raw[w + k + 1u] << (31u - s)) << 1u); }
fn q3_d(base: u32) -> f32 {
  let b = base + 108u; let v = (raw[b >> 2u] >> (8u * (b & 3u))) & 0xFFFFu;   // b is even: the f16 never crosses a word
  return unpack2x16float(v).x;
}
// the two scales of piece pz (l < 16, l >= 16), already minus 32
fn q3_scales(base: u32, pz: u32) -> vec2<f32> {
  let sb = base + 96u; let ws = sb >> 2u; let ss = 8u * (sb & 3u);
  let a0 = q3_par(ws, 0u, ss); let a1 = q3_par(ws, 1u, ss); let tmp = q3_par(ws, 2u, ss);
  let n = pz / 4u; let j = pz % 4u;
  let wi = 2u * n + j / 2u;
  var w: u32;
  if (wi == 0u) { w = (a0 & 0x0F0F0F0Fu) | ((tmp & 0x03030303u) << 4u); }
  else if (wi == 1u) { w = (a1 & 0x0F0F0F0Fu) | (((tmp >> 2u) & 0x03030303u) << 4u); }
  else if (wi == 2u) { w = ((a0 >> 4u) & 0x0F0F0F0Fu) | (((tmp >> 4u) & 0x03030303u) << 4u); }
  else { w = ((a1 >> 4u) & 0x0F0F0F0Fu) | (((tmp >> 6u) & 0x03030303u) << 4u); }
  let b = 8u * (2u * (j % 2u));
  return vec2<f32>(f32((w >> b) & 0xFFu) - 32.0, f32((w >> (b + 8u)) & 0xFFu) - 32.0);
}
// values 4k .. 4k+3 of piece pz (k = 0..7), not scaled: low + 4 * high - 4, in -4..3
fn q3_vals(base: u32, pz: u32, k: u32) -> vec4<f32> {
  let n = pz / 4u; let j = pz % 4u;
  let qb = base + 32u + 32u * n; let hb = base;
  let lo = (q3_par(qb >> 2u, k, 8u * (qb & 3u)) >> (2u * j)) & 0x03030303u;
  let hi = (q3_par(hb >> 2u, k, 8u * (hb & 3u)) >> (4u * n + j)) & 0x01010101u;
  return vec4<f32>(unpack4xU8(lo | (hi << 2u))) - vec4<f32>(4.0);
}
