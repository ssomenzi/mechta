// Bench of the recurrent step of the gated delta net (M2). Three variants, same formula as ops.wgsl::delta:
//   S = S * exp(g); dl = (v - S^T k) * beta; S = S + k dl^T; out = S^T q / sqrt(128).
// One workgroup per V head (32), one thread per column j. 128 x 128 f32 per head, 2 MiB in total.
// a = normalized q|k (q in [0, 2048), k in [2048, 4096)), b = qkv after conv (v from 4096), c = beta|gate (32 + 32).
//   delta1: state o[h*16384 + j*128 + i] = S[i][j] (as in ggml and as the engine does today). Thread j reads addresses
//           512 bytes apart from its neighbours: each wave read touches 64 different cache lines. Two passes.
//   delta2: state o[h*16384 + i*128 + j] = S[i][j]: at each step i the 128 threads read 512 contiguous bytes. Two passes.
//   delta3: like delta2, but the column stays in registers between the two passes: one read and one write per value.
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read_write> o: array<f32>;
@group(0) @binding(4) var<storage, read_write> o2: array<f32>;

const INV_SQRT_128: f32 = 0.08838834764831845;

@compute @workgroup_size(128)
fn delta1(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let h = wid.x; let kh = h % 16u;
  let q0 = kh * 128u; let k0 = 2048u + kh * 128u; let v = b[4096u + h * 128u + j];
  let beta = c[h]; let eg = exp(c[32u + h]);
  let s0 = h * 16384u + j * 128u;
  var sk = 0.0;
  for (var i: u32 = 0u; i < 128u; i++) { let m = o[s0 + i] * eg; o[s0 + i] = m; sk += m * a[k0 + i]; }
  let dl = (v - sk) * beta;
  var sq = 0.0;
  for (var i: u32 = 0u; i < 128u; i++) { let m = o[s0 + i] + a[k0 + i] * dl; o[s0 + i] = m; sq += m * a[q0 + i]; }
  o2[h * 128u + j] = sq * INV_SQRT_128;
}

@compute @workgroup_size(128)
fn delta2(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let h = wid.x; let kh = h % 16u;
  let q0 = kh * 128u; let k0 = 2048u + kh * 128u; let v = b[4096u + h * 128u + j];
  let beta = c[h]; let eg = exp(c[32u + h]);
  let s0 = h * 16384u + j;
  var sk = 0.0;
  for (var i: u32 = 0u; i < 128u; i++) { let m = o[s0 + i * 128u] * eg; o[s0 + i * 128u] = m; sk += m * a[k0 + i]; }
  let dl = (v - sk) * beta;
  var sq = 0.0;
  for (var i: u32 = 0u; i < 128u; i++) { let m = o[s0 + i * 128u] + a[k0 + i] * dl; o[s0 + i * 128u] = m; sq += m * a[q0 + i]; }
  o2[h * 128u + j] = sq * INV_SQRT_128;
}

@compute @workgroup_size(128)
fn delta3(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let h = wid.x; let kh = h % 16u;
  let q0 = kh * 128u; let k0 = 2048u + kh * 128u; let v = b[4096u + h * 128u + j];
  let beta = c[h]; let eg = exp(c[32u + h]);
  let s0 = h * 16384u + j;
  var col: array<f32, 128>;
  var sk = 0.0;
  for (var i: u32 = 0u; i < 128u; i++) { let m = o[s0 + i * 128u] * eg; col[i] = m; sk += m * a[k0 + i]; }
  let dl = (v - sk) * beta;
  var sq = 0.0;
  for (var i: u32 = 0u; i < 128u; i++) { let m = col[i] + a[k0 + i] * dl; o[s0 + i * 128u] = m; sq += m * a[q0 + i]; }
  o2[h * 128u + j] = sq * INV_SQRT_128;
}
