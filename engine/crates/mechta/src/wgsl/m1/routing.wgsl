// M1 version (a single thread), kept for comparisons with M2. Same as before.
@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> probs: array<f32>;
@group(0) @binding(2) var<storage, read_write> ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> weights: array<f32>;

@compute @workgroup_size(1)
fn routing() {
  var mx = logits[0];
  for (var i: u32 = 1u; i < 256u; i++) { mx = max(mx, logits[i]); }
  var tot = 0.0;
  for (var i: u32 = 0u; i < 256u; i++) { let v = exp(logits[i] - mx); probs[i] = v; tot += v; }
  let inv = 1.0 / tot;
  for (var i: u32 = 0u; i < 256u; i++) { probs[i] = probs[i] * inv; }
  var taken: array<bool, 256>;
  for (var i: u32 = 0u; i < 256u; i++) { taken[i] = false; }
  var sum = 0.0;
  for (var k: u32 = 0u; k < 8u; k++) {
    var best = 0u; var bv = -1.0;
    for (var i: u32 = 0u; i < 256u; i++) { if (!taken[i] && probs[i] > bv) { bv = probs[i]; best = i; } }
    taken[best] = true; ids[k] = best; weights[k] = bv; sum += bv;
  }
  let den = max(sum, 6.103515625e-5);
  for (var k: u32 = 0u; k < 8u; k++) { weights[k] = weights[k] / den; }
}
