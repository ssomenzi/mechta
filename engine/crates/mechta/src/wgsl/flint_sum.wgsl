// Flint, end of a MoE layer: o = ((sum_j weights[j] * ed[j]) + shared) + residual, in llama.cpp's order
// (experts weighted, added one after the other, then the shared expert, then the residual). n = 2048.
@group(0) @binding(0) var<storage, read> ed: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read> sd: array<f32>;
@group(0) @binding(3) var<storage, read> res: array<f32>;
@group(0) @binding(4) var<storage, read_write> o: array<f32>;

@compute @workgroup_size(64)
fn moe_sum6(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= 2048u) { return; }
  var m = ed[i] * weights[0];
  for (var j: u32 = 1u; j < 6u; j++) { m = m + ed[j * 2048u + i] * weights[j]; }
  o[i] = (m + sd[i]) + res[i];
}

// prefill: token tk (grid y) of the block, experts at (tk * 8 + j) * 2048, weights at tk * 8 + j (8 slots).
struct PN { n: u32, nb: u32, c: u32, d: u32 }
@group(0) @binding(5) var<uniform> pn: PN;
@compute @workgroup_size(64)
fn moe_sum6_n(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let tk = gid.y; if (i >= 2048u || tk >= pn.n) { return; }
  let e0 = tk * 8u * 2048u; let w0 = tk * 8u;
  var m = ed[e0 + i] * weights[w0];
  for (var j: u32 = 1u; j < 6u; j++) { m = m + ed[e0 + j * 2048u + i] * weights[w0 + j]; }
  o[tk * 2048u + i] = (m + sd[tk * 2048u + i]) + res[tk * 2048u + i];
}
