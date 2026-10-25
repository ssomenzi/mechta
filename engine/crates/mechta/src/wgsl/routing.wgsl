// MoE routing: as build_moe_ffn of llama.cpp. One workgroup, one thread per expert (256).
// probs = softmax(logits); top-8 (on a tie the lower index wins);
// weights = probs[top] / max(sum, 6.103515625e-5).
// M2: before it was a single thread (0.45 ms per layer).
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;
// M3 (prefill): n = tokens of the block (rewritten at every block), nb = NB; grp = groups per expert (see gather)
struct P { n: u32, nb: u32, c: u32, d: u32 }
@group(0) @binding(4) var<uniform> p: P;
@group(0) @binding(5) var<storage, read_write> grp: array<u32>;

var<workgroup> rv: array<f32, 256>;
var<workgroup> ri: array<u32, 256>;
var<workgroup> chosen: array<f32, 8>;

@compute @workgroup_size(256)
fn routing(@builtin(local_invocation_index) t: u32) { routing_t(t, 0u, 0u, 0u); }
// M4: second token of the double step: logits at 256.., probs at 256.., weights at 8.. (ids in its own buffer, bound separately)
@compute @workgroup_size(256)
fn routing_b(@builtin(local_invocation_index) t: u32) { routing_t(t, 256u, 8u, 0u); }
// M3: one workgroup per token (grid = NB, tokens beyond p.n exit): logits at tok*256, probs at tok*256, ids and weights at tok*8.
@compute @workgroup_size(256)
fn routing_n(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let tok = wid.x; if (tok >= p.n) { return; }
  routing_t(t, tok * 256u, tok * 8u, tok * 8u);
}
// M3: groups the tokens by expert. One workgroup, one thread per expert e: counts the entries ids[i] == e (i < n*8),
// exclusive scan, grp[2e] = start, grp[2e+1] = how many; list at grp[512 + start + j] = (tok << 3) | slot,
// in token order. This way the expert GEMM reads each used expert only once (union over n tokens).
var<workgroup> cnt: array<u32, 256>;
@compute @workgroup_size(256)
fn gather(@builtin(local_invocation_index) t: u32) {
  let tot = p.n * 8u;
  var q = 0u;
  for (var i: u32 = 0u; i < tot; i++) { if (ids[i] == t) { q++; } }
  cnt[t] = q;
  workgroupBarrier();
  var start = 0u;
  for (var e: u32 = 0u; e < t; e++) { start += cnt[e]; }
  grp[2u * t] = start; grp[2u * t + 1u] = q;
  var j = 0u;
  for (var i: u32 = 0u; i < tot; i++) { if (ids[i] == t) { grp[512u + start + j] = ((i / 8u) << 3u) | (i % 8u); j++; } }
}
fn routing_t(t: u32, ol: u32, op: u32, oi: u32) {
  let lg = logits[ol + t];
  // maximum
  rv[t] = lg;
  workgroupBarrier();
  for (var m: u32 = 128u; m > 0u; m >>= 1u) {
    if (t < m) { rv[t] = max(rv[t], rv[t + m]); }
    workgroupBarrier();
  }
  let mx = rv[0];
  workgroupBarrier();
  // sum of the exponentials
  let e = exp(lg - mx);
  rv[t] = e;
  workgroupBarrier();
  for (var m: u32 = 128u; m > 0u; m >>= 1u) {
    if (t < m) { rv[t] += rv[t + m]; }
    workgroupBarrier();
  }
  let pr = e * (1.0 / rv[0]);
  probs[ol + t] = pr;
  workgroupBarrier();
  // top-8: 8 rounds of "maximum with index"; already chosen entries count as -1
  var mine = pr;
  for (var k: u32 = 0u; k < 8u; k++) {
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
    var sum = 0.0;
    for (var k: u32 = 0u; k < 8u; k++) { sum += chosen[k]; }
    let den = max(sum, 6.103515625e-5);
    for (var k: u32 = 0u; k < 8u; k++) { weights[op + k] = chosen[k] / den; }
  }
}
