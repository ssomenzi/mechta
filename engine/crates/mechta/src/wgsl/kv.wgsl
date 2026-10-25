// KV cache in a compact format. The Rust side cuts this file at the `// FORMAT <name>` lines and puts
// the chosen block in place of the KV-ACCESS block of ops.wgsl. Every block declares the same names:
//   b4 / c4  — K and V cache (bindings 9 and 10, read by the attention kernels through kk() and vv())
//   kvo      — the cache being written (binding 11), used by the three writer kernels below
//   kk(i), vv(i) — 4 elements of K / V as vec4<f32>, by vec4 index i (element index / 4)
// Writer kernels (same dispatch and uniforms as rope / rope2 / copy of ops.wgsl, which write the f32 cache):
//   kv_rope  — K of one token: RoPE on the first 64 dims of each 256-dim head, row r of `a` at r * step_a,
//              written at element p.extra + r * p.step_o; b = [cos 0..31 | sin 0..31]
//   kv_rope2 — the same for several tokens: table of row r at (r / p.flag) * 64
//   kv_copy  — V: p.n contiguous elements of `a` written at element p.extra
// Element offsets are multiples of 256, so a writer thread always owns whole words (f16: 2 elements).
// The value of dim d after RoPE (d < 64) or as is (d >= 64), as in rope / rope2.
// FORMAT f16 — 2 bytes per element, two halves per u32 (pack2x16float)
@group(0) @binding(9) var<storage, read> b4: array<vec2<u32>>;
@group(0) @binding(10) var<storage, read> c4: array<vec2<u32>>;
@group(0) @binding(11) var<storage, read_write> kvo: array<u32>;
fn kk(i: u32) -> vec4<f32> { let w = b4[i]; return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn vv(i: u32) -> vec4<f32> { let w = c4[i]; return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn kv_src(x0: u32, d: u32, tb: u32) -> f32 {
  if (d < 32u) { return a[x0 + d] * b[tb + d] - a[x0 + d + 32u] * b[tb + 32u + d]; }
  if (d < 64u) { let i = d - 32u; return a[x0 + i] * b[tb + 32u + i] + a[x0 + d] * b[tb + i]; }
  return a[x0 + d];
}
fn kv_row(x0: u32, o0: u32, tb: u32, t: u32) {
  if (t >= 128u) { return; }
  let d = 2u * t;
  kvo[(o0 + d) / 2u] = pack2x16float(vec2<f32>(kv_src(x0, d, tb), kv_src(x0, d + 1u, tb)));
}
@compute @workgroup_size(256)
fn kv_copy(@builtin(global_invocation_id) gid: vec3<u32>) {
  let e = 2u * gid.x; if (e >= p.n) { return; }
  kvo[(p.extra + e) / 2u] = pack2x16float(vec2<f32>(a[e], a[e + 1u]));
}
// FORMAT common — appended after the chosen block
@compute @workgroup_size(256)
fn kv_rope(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let r = wid.x; kv_row(r * p.step_a, p.extra + r * p.step_o, 0u, t);
}
@compute @workgroup_size(256)
fn kv_rope2(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let r = wid.x; kv_row(r * p.step_a, p.extra + r * p.step_o, (r / p.flag) * 64u, t);
}
