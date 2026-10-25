// Flint MoE, one token. Reference: build_moe_ffn of llama.cpp with gating = sigmoid, 6 of 256, weights normalized.
@group(0) @binding(0) var<storage, read> lg: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;

// prefill: n = tokens of the block (binding 4, as routing.wgsl so gather can share the uniform)
struct PN { n: u32, nb: u32, c: u32, d: u32 }
@group(0) @binding(4) var<uniform> pn: PN;

var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
var<workgroup> chosen: array<f32, 6>;

// probs = sigmoid(logits); top-6 by probs (on a tie the lower index wins); weights = probs[top] / max(sum, 6.103515625e-5).
@compute @workgroup_size(256)
fn routing_sig(@builtin(local_invocation_index) t: u32) { routing_t(t, 0u, 0u); }
// prefill: one workgroup per token; logits and probs at tok * 256, ids and weights at tok * 8 (8 slots, as routing.wgsl,
// so the expert GEMMs and gather of Qwen work unchanged); slots 6 and 7 get id 0xFFFFFFFF, which no expert matches.
@compute @workgroup_size(256)
fn routing_sig_n(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let tok = wid.x; if (tok >= pn.n) { return; }
  if (t == 0u) { ids[tok * 8u + 6u] = 0xFFFFFFFFu; ids[tok * 8u + 7u] = 0xFFFFFFFFu; weights[tok * 8u + 6u] = 0.0; weights[tok * 8u + 7u] = 0.0; }
  routing_t(t, tok * 256u, tok * 8u);
}
fn routing_t(t: u32, ol: u32, oi: u32) {
  let pr = 1.0 / (1.0 + exp(-lg[ol + t]));
  probs[ol + t] = pr;
  var mine = pr;
  for (var k: u32 = 0u; k < 6u; k++) {
    rv[t] = mine; ri[t] = t;
    workgroupBarrier();
    for (var m: u32 = 128u; m > 0u; m >>= 1u) {
      if (t < m) {
        let a = rv[t]; let b = rv[t + m];
        if (b > a || (b == a && ri[t + m] < ri[t])) { rv[t] = b; ri[t] = ri[t + m]; }
      }
      workgroupBarrier();
    }
    let winner = ri[0];
    if (t == 0u) { ids[oi + k] = winner; chosen[k] = rv[0]; }
    if (t == winner) { mine = -1.0; }
    workgroupBarrier();
  }
  if (t == 0u) {
    var s = 0.0;
    for (var k: u32 = 0u; k < 6u; k++) { s += chosen[k]; }
    s = max(s, 6.103515625e-5);
    for (var k: u32 = 0u; k < 6u; k++) { weights[oi + k] = chosen[k] / s; }
  }
}
