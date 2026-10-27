// Operations of a layer (decode, one token), in the order and with the formulas of llama.cpp 42d9581.
// Same bindings for all entry points: a..e read-only, o and o2 written, p uniform.
// Each entry point uses only the ones it needs. f32 values in the uniform passed as bits (bitcast).
struct P { n: u32, rows: u32, f0: u32, f1: u32, step_a: u32, step_o: u32, flag: u32, extra: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read> d: array<f32>;
@group(0) @binding(4) var<storage, read> e: array<f32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
@group(0) @binding(6) var<storage, read_write> o2: array<f32>;
@group(0) @binding(7) var<uniform> p: P;
@group(0) @binding(8) var<storage, read_write> o3: array<f32>;   // third output (delta2: ao)
// KV-ACCESS-BEGIN — the Rust side replaces this block when the KV cache is not f32 (see kv.wgsl)
@group(0) @binding(9) var<storage, read> b4: array<vec4<f32>>;    // the same buffer as b, read as vec4 (attn_v3: rows of K)
@group(0) @binding(10) var<storage, read> c4: array<vec4<f32>>;   // the same buffer as c, read as vec4 (attn_fd: rows of V)
// K and V of the cache as vec4 of f32, by vec4 index (element index / 4)
fn kk(i: u32) -> vec4<f32> { return b4[i]; }
fn vv(i: u32) -> vec4<f32> { return c4[i]; }
// KV-ACCESS-END

var<workgroup> partial: array<f32, 256>;

fn silu(v: f32) -> f32 { return v / (1.0 + exp(-v)); }
fn sigm(v: f32) -> f32 { return 1.0 / (1.0 + exp(-v)); }

// RMSNorm over `rows` rows of n values (row r at a[r * step_a ..]):
//   y = x / sqrt(mean(x^2) + eps) * scale, then (flag & 1) * weight b[i], then (flag & 2) * silu(c[r*n + i]).
// f0 = eps, f1 = scale (f32 bits). One workgroup of 256 threads per row.
@compute @workgroup_size(256)
fn rmsnorm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let r = wid.x; let x0 = r * p.step_a; let o0 = r * p.step_o + p.extra;   // extra: output offset (MTP: concat)
  var s = 0.0;
  for (var i = t; i < p.n; i += 256u) { let v = a[x0 + i]; s += v * v; }
  partial[t] = s;
  workgroupBarrier();
  for (var m: u32 = 128u; m > 0u; m >>= 1u) {
    if (t < m) { partial[t] += partial[t + m]; }
    workgroupBarrier();
  }
  let scale = 1.0 / sqrt(partial[0] / f32(p.n) + bitcast<f32>(p.f0));
  if ((p.flag & 4u) != 0u) {
    // M4: output in f16, two values per f32 slot (for the 2-column matvecs with x in f16); o0 and n even
    for (var i = 2u * t; i < p.n; i += 512u) {
      let v = vec2<f32>(rms_val(x0, r, i, scale), rms_val(x0, r, i + 1u, scale));
      o[(o0 + i) / 2u] = bitcast<f32>(pack2x16float(v));
    }
    return;
  }
  for (var i = t; i < p.n; i += 256u) { o[o0 + i] = rms_val(x0, r, i, scale); }
}
fn rms_val(x0: u32, r: u32, i: u32, scale: f32) -> f32 {
  var v = a[x0 + i] * scale * bitcast<f32>(p.f1);
  if ((p.flag & 1u) != 0u) { v = v * b[i]; }
  if ((p.flag & 2u) != 0u) { v = v * silu(c[r * p.n + i]); }
  return v;
}

// Causal DeltaNet convolution + SiLU, one thread per channel (n = 8192).
// a = new qkv, b = conv1d weights [channel*4 + j], c = state [channel*3 + j] (oldest first).
// o = output after SiLU, o2 = new state.
@compute @workgroup_size(64)
fn conv(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x; if (ch >= p.n) { return; }
  let s0 = c[ch * 3u]; let s1 = c[ch * 3u + 1u]; let s2 = c[ch * 3u + 2u]; let fresh = a[ch];
  let w = ch * 4u;
  var s = 0.0;
  s += s0 * b[w]; s += s1 * b[w + 1u]; s += s2 * b[w + 2u]; s += fresh * b[w + 3u];
  o[ch] = silu(s);
  o2[ch * 3u] = s1; o2[ch * 3u + 1u] = s2; o2[ch * 3u + 2u] = fresh;
}

// DeltaNet beta and gate (n = 32 V heads): a = raw beta, b = raw alpha, c = ssm_dt, d = ssm_a.
// o[0..n] = sigmoid(beta), o[n..2n] = softplus(alpha + dt) * a.
@compute @workgroup_size(32)
fn beta_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let h = gid.x; if (h >= p.n) { return; }
  o[h] = sigm(a[h]);
  let z = b[h] + c[h];
  var sp = z; if (z <= 20.0) { sp = log(1.0 + exp(z)); }   // ggml_softplus: x > 20 -> x
  o[p.n + h] = sp * d[h];
}

// One recurrent step of the gated delta net. One workgroup per V head (32), one thread per column j.
// a = normalized q|k (q in [0, 2048), k in [2048, 4096)), b = qkv after conv (v from 4096),
// c = beta|gate (32 + 32). Output o2[h*128 + j] = (S^T q)[j] / sqrt(128). V head h uses K head h % 16.
// M2 (delta bench): state o[h*16384 + i*128 + j] = S[i][j], so at each step i the 128 threads
// read 512 contiguous bytes; the column stays in registers between the two passes (one read and one write per
// value). On the bench: 0.137-0.155 ms → 0.041 ms (27 → 102 GB/s). Only this kernel reads the state.
// The old layout (o[h*16384 + j*128 + i], as in ggml, two passes in memory) remains in delta1 (MECHTA_DELTA=v1).
@compute @workgroup_size(128)
fn delta(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
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
  o2[h * 128u + j] = sq * (1.0 / sqrt(128.0));
}


// o = a + b (residual), n values.
@compute @workgroup_size(64)
fn sum(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= p.n) { return; }
  o[i] = a[i] + b[i];
}

// SwiGLU: o = silu(a) * b, n values.
@compute @workgroup_size(64)
fn swiglu(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= p.n) { return; }
  o[i] = silu(a[i]) * b[i];
}

// MoE sum and residual, n = 2048, rows = 8 experts:
// a = down outputs [8 * n], b = normalized weights [8], c = shared expert [n], d = raw shared gate [1],
// e = residual. ggml order: ((w0 d0 + w1 d1) + ...) + sigmoid(g) sh, then + residual.
// o = ffn_out (before the residual), o2 = l_out.
@compute @workgroup_size(64)
// M4: with extra = number of tokens (0 or 1 = one token) token tok has a[(tok*rows + k)*n + i], b[tok*rows + k],
// c[tok*n + i], d[tok], e[tok*n + i]; o and o2 contiguous per token.
// flag = offset of e, o and o2 (2-token MTP draft: the MoE runs only for the second token, whose inputs are in
// the place of the first but whose residual and output are at 2048).
fn moe_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= p.n * max(p.extra, 1u)) { return; }
  let tok = i / p.n; let i0 = i % p.n;
  var s = a[(tok * p.rows) * p.n + i0] * b[tok * p.rows];
  for (var k: u32 = 1u; k < p.rows; k++) { s += a[(tok * p.rows + k) * p.n + i0] * b[tok * p.rows + k]; }
  let f = s + c[tok * p.n + i0] * sigm(d[tok]);
  o[p.flag + i] = f;
  o2[p.flag + i] = f + e[p.flag + i];
}

// Copy: o[extra + i] = a[i], n values (V into the cache).
@compute @workgroup_size(64)
fn copy(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= p.n) { return; }
  o[p.extra + i] = a[i];
}

// NEOX RoPE on the first 64 dimensions of `rows` heads of n = 256 (row r at a[r * step_a ..],
// output at o[extra + r * step_o ..], to write K directly into the cache).
// b = table [cos 0..31 | sin 0..31] for the position, computed on the CPU as in ggml.
// One thread per pair (i, i + 32); dimensions 64..255 are copied.
@compute @workgroup_size(256)
fn rope(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let r = wid.x; let x0 = r * p.step_a; let o0 = p.extra + r * p.step_o;
  if (t < 32u) {
    let x1 = a[x0 + t]; let x2 = a[x0 + t + 32u];
    let co = b[t]; let si = b[32u + t];
    o[o0 + t] = x1 * co - x2 * si;
    o[o0 + t + 32u] = x1 * si + x2 * co;
  } else if (t >= 64u) {
    o[o0 + t] = a[x0 + t];
  }
}

// Attention of one Q head per workgroup (16), 256 threads (one per dimension).
// a = q after RoPE [16 * 256], b = K cache [pos * 512 + kvh * 256 + d], c = V cache (same order),
// n = end of the positions (1..1024), extra = first position (0; 1 for the MTP, whose cache has no position 0),
// f0 = scale. GQA: heads 0..7 -> KV 0, 8..15 -> KV 1.
// o[h * 256 + d] = sum_t softmax(q k_t * scale)_t v_t[d].
var<workgroup> punt: array<f32, 1024>;

// Attention output gate: o[h*256 + d] = a[h*256 + d] * sigmoid(b[h*512 + 256 + d]), n = 4096.
@compute @workgroup_size(64)
fn attn_gate(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= p.n) { return; }
  let h = i / 256u; let dd = i % 256u;
  o[i] = a[i] * sigm(b[h * 512u + 256u + dd]);
}

// Argmax of the logits on the GPU (M2): 8 bytes are read instead of 1 MB. On ties the lowest index wins,
// like the CPU argmax. Step 1: one workgroup per 256 logits → o[g] = max, o2[g] = index (bits in an f32).
// Step 2: one workgroup over the p.n partials (a = maxima, b = indices) → o[0] = max, o2[0] = index.
var<workgroup> am_v: array<f32, 256>;
var<workgroup> am_i: array<u32, 256>;
fn am_reduce(t: u32) {
  for (var m: u32 = 128u; m > 0u; m >>= 1u) {
    if (t < m) {
      let a = am_v[t]; let b = am_v[t + m];
      if (b > a || (b == a && am_i[t + m] < am_i[t])) { am_v[t] = b; am_i[t] = am_i[t + m]; }
    }
    workgroupBarrier();
  }
}
@compute @workgroup_size(256)
fn argmax1(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let i = wid.x * 256u + t;
  am_v[t] = select(-3.0e38, a[p.extra + i], i < p.n); am_i[t] = i;   // extra = input offset (second token)
  workgroupBarrier();
  am_reduce(t);
  if (t == 0u) { o[p.rows + wid.x] = am_v[0]; o2[p.rows + wid.x] = bitcast<f32>(am_i[0]); }
}
@compute @workgroup_size(256)
fn argmax2(@builtin(local_invocation_index) t: u32) {
  var v = -3.0e38; var ix = 0xFFFFFFFFu;
  for (var j = t; j < p.n; j += 256u) {
    let bv = a[p.extra + j]; let bi = bitcast<u32>(b[p.extra + j]);
    if (bv > v || (bv == v && bi < ix)) { v = bv; ix = bi; }
  }
  am_v[t] = v; am_i[t] = ix;
  workgroupBarrier();
  am_reduce(t);
  if (t == 0u) { o[p.rows] = am_v[0]; o2[p.rows] = bitcast<f32>(am_i[0]); }
}

// ===================== M4: kernels for the 2-token step (accepted token + draft) =====================
// Layout: the work buffers hold the two tokens in a row (token t at t * size), except where noted.

// beta and gate over rows tokens: a[t*32 + h], b[t*32 + h] → o[t*64 + h], o[t*64 + 32 + h].
@compute @workgroup_size(64)
fn beta_gate2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= p.n * p.rows) { return; }
  let t = i / p.n; let h = i % p.n;
  o[t * 2u * p.n + h] = sigm(a[i]);
  let z = b[i] + c[h];
  var sp = z; if (z <= 20.0) { sp = log(1.0 + exp(z)); }
  o[t * 2u * p.n + p.n + h] = sp * d[h];
}

// Convolution over rows tokens in sequence (n = 8192 channels). a = qkv [t*8192 + ch], c = input state, o2 = state after
// the last token. Output o in "qk first" layout: for ch < 4096 o[t*4096 + ch], for ch >= 4096
// o[rows*4096 + t*4096 + ch - 4096]: so the q|k rows of all tokens are contiguous for the norm, and v comes after.
@compute @workgroup_size(64)
fn conv2(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x; if (ch >= p.n) { return; }
  var s0 = c[ch * 3u]; var s1 = c[ch * 3u + 1u]; var s2 = c[ch * 3u + 2u];
  let w = ch * 4u;
  for (var t: u32 = 0u; t < p.rows; t++) {
    let fresh = a[t * p.n + ch];
    let s = s0 * b[w] + s1 * b[w + 1u] + s2 * b[w + 2u] + fresh * b[w + 3u];
    var oi = t * 4096u + ch;
    if (ch >= 4096u) { oi = p.rows * 4096u + t * 4096u + ch - 4096u; }
    o[oi] = silu(s);
    s0 = s1; s1 = s2; s2 = fresh;
    if (t == 0u) { o3[ch * 3u] = s0; o3[ch * 3u + 1u] = s1; o3[ch * 3u + 2u] = s2; }   // state after the first token (draft rejection)
  }
  o2[ch * 3u] = s0; o2[ch * 3u + 1u] = s1; o2[ch * 3u + 2u] = s2;
}

// Recurrent step over 2 tokens in sequence (same computation as `delta`). a = normalized q|k [t*4096 + ...],
// b = output of conv2 (v at 8192 + t*4096 + h*128 + j), c = beta|gate [t*64 + ...].
// o = state: read (S_in) and rewritten with the state after the second token (S2); o2 = state after the first token (S1),
// to restart from there if the draft is rejected; o3 = outputs [t*4096 + h*128 + j].
@compute @workgroup_size(128)
fn delta2(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let h = wid.x; let kh = h % 16u;
  let s0 = h * 16384u + j;
  var col: array<f32, 128>;
  for (var i: u32 = 0u; i < 128u; i++) { col[i] = o[s0 + i * 128u]; }
  for (var t: u32 = 0u; t < 2u; t++) {
    let q0 = t * 4096u + kh * 128u; let k0 = t * 4096u + 2048u + kh * 128u;
    let v = b[8192u + t * 4096u + h * 128u + j];
    let beta = c[t * 64u + h]; let eg = exp(c[t * 64u + 32u + h]);
    var sk = 0.0;
    for (var i: u32 = 0u; i < 128u; i++) { let m = col[i] * eg; col[i] = m; sk += m * a[k0 + i]; }
    let dl = (v - sk) * beta;
    var sq = 0.0;
    for (var i: u32 = 0u; i < 128u; i++) { let m = col[i] + a[k0 + i] * dl; col[i] = m; sq += m * a[q0 + i]; }
    o3[t * 4096u + h * 128u + j] = sq * (1.0 / sqrt(128.0));
    if (t == 0u) { for (var i: u32 = 0u; i < 128u; i++) { o2[s0 + i * 128u] = col[i]; } }
  }
  for (var i: u32 = 0u; i < 128u; i++) { o[s0 + i * 128u] = col[i]; }
}

// RoPE for several tokens: like `rope`, but row r uses the table of token r / flag (flag = rows per token),
// b = tables in a row (64 values per token).
@compute @workgroup_size(256)
fn rope2(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let r = wid.x; let x0 = r * p.step_a; let o0 = p.extra + r * p.step_o; let tb = (r / p.flag) * 64u;
  if (t < 32u) {
    let x1 = a[x0 + t]; let x2 = a[x0 + t + 32u];
    let co = b[tb + t]; let si = b[tb + 32u + t];
    o[o0 + t] = x1 * co - x2 * si;
    o[o0 + t + 32u] = x1 * si + x2 * co;
  } else if (t >= 64u) {
    o[o0 + t] = a[x0 + t];
  }
}


// M4: swiglu and attention gate with output in f16 (two values per slot): one thread per pair, n = number of values.
@compute @workgroup_size(64)
fn swiglu16(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = 2u * gid.x; if (i >= p.n) { return; }
  o[gid.x] = bitcast<f32>(pack2x16float(vec2<f32>(silu(a[i]) * b[i], silu(a[i + 1u]) * b[i + 1u])));
}
@compute @workgroup_size(64)
fn attn_gate16(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = 2u * gid.x; if (i >= p.n) { return; }
  let h = i / 256u; let dd = i % 256u;   // i and i+1 are in the same head (256 is even)
  o[gid.x] = bitcast<f32>(pack2x16float(vec2<f32>(a[i] * sigm(b[h * 512u + 256u + dd]), a[i + 1u] * sigm(b[h * 512u + 256u + dd + 1u]))));
}

// ===================== M4: attention parallel over positions =====================
// `attn` runs a SEQUENTIAL loop over the positions with 8 barriers per position: at 900 of context it costs 7.8 ms per token
// (0.4 at 30). Here: phase 1, one thread per position (in steps of 256) computes the score q . k_s reading its row
// of K (q is in shared memory); two reductions in total (max, sum); phase 2, one thread per dimension d
// accumulates sum_s w_s v_s[d] with V reads contiguous across threads. Same formulas and same inputs as `attn`/`attn2`.

// attn_v3: like attn_v2, but the K row is read as vec4 (64 reads per row instead of 256) and q is in vec4.

// ===================== M4: flash-decoding with GQA =====================
// One workgroup per (KV head, block of 128 positions, token): reads K and V of the block ONCE and computes the 8 Q heads
// that share that KV head. Phase 1 (threads 0..127, one per position): 8 scores per position (q in shared
// memory, K as vec4); max and sum per head over the block. Phase 2 (256 threads, one per dimension): acc[h][d] =
// sum_s w[h][s] v[s][d], V read once and contiguous. Partial output per (token, head, block): [max, sum, acc 256].
// `attn_fd_combine` merges the blocks: o[d] = sum_b e^(m_b - M) acc_b[d] / sum_b e^(m_b - M) sum_b.
// Blocks past the valid positions write max = -3e38 and sum 0. NB = CTX / 128 blocks, always launched.
const FD_B: u32 = 128u;
var<workgroup> fq: array<vec4<f32>, 512>;      // 8 heads x 64 vec4
var<workgroup> fs: array<f32, 1024>;           // 8 heads x 128 scores
var<workgroup> fm: array<f32, 8>;
var<workgroup> fsum: array<f32, 8>;
// Prefill: with p.step_o = T > 0 one workgroup does the T tokens of the batch ONE AFTER ANOTHER on the
// same block of positions (grid z = 1): K and V of the block are read from memory once and then from the L2 cache,
// instead of T workgroups far apart in time. With step_o = 0 (decode) the token is wid.z as before.
@compute @workgroup_size(256)
fn attn_fd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let nt = max(p.step_o, 1u);
  for (var j: u32 = 0u; j < nt; j++) {
    attn_fd_one(wid.x, wid.y, select(wid.z, j, p.step_o > 0u), t);
    workgroupBarrier();
  }
}
fn attn_fd_one(kvh: u32, block: u32, tl: u32, t: u32) {
  let tok = tl + p.step_a;   // step_a: first token of the batch (prefill in pieces of 8)
  let scale = bitcast<f32>(p.f0);
  let s0 = p.extra; let n = p.n + tok;
  let nb = p.rows;                                   // blocks launched (CTX / 128)
  let start = s0 + block * FD_B;
  if (start >= n) { return; }                        // block past the valid positions: the merge does not read it
  // q of the 8 heads (8 x 256 f32) in shared memory, as vec4
  for (var i = t; i < 512u; i += 256u) {
    let h = 8u * kvh + i / 64u; let d0 = (i % 64u) * 4u; let q0 = tok * 4096u + h * 256u + d0;
    fq[i] = vec4<f32>(a[q0], a[q0 + 1u], a[q0 + 2u], a[q0 + 3u]);
  }
  workgroupBarrier();
  // phase 1 (2 threads per position, half a K row each = 512 contiguous B; before, 128 threads with one row each)
  let pos = t / 2u; let side = t % 2u;
  let s = s0 + block * FD_B + pos;
  let valid = s < n;
  let phase = p.f1;                                    // measurement: 1 = skip phase 2 (V), 2 = skip phase 1 (K)
  var d8: array<f32, 8>;
  if (valid && phase != 2u) {
    let kb = (s * 512u + kvh * 256u) / 4u + side * 32u;
    for (var i: u32 = 0u; i < 32u; i++) {
      let k4 = kk(kb + i);
      for (var h: u32 = 0u; h < 8u; h++) { d8[h] += dot(fq[h * 64u + side * 32u + i], k4); }
    }
  }
  if (side == 1u) { for (var h: u32 = 0u; h < 8u; h++) { fs[h * FD_B + pos] = d8[h]; } }
  workgroupBarrier();
  if (side == 0u) { for (var h: u32 = 0u; h < 8u; h++) { fs[h * FD_B + pos] = select(-3.0e38, (d8[h] + fs[h * FD_B + pos]) * scale, valid); } }
  workgroupBarrier();
  // max and sum per head (8 threads, 128 values each: little work)
  if (t < 8u) {
    var mx = -3.0e38;
    for (var i: u32 = 0u; i < FD_B; i++) { mx = max(mx, fs[t * FD_B + i]); }
    fm[t] = mx;
    var sm = 0.0;
    for (var i: u32 = 0u; i < FD_B; i++) { let w = select(0.0, exp(fs[t * FD_B + i] - mx), fs[t * FD_B + i] > -1.0e38); fs[t * FD_B + i] = w; sm += w; }
    fsum[t] = sm;
  }
  workgroupBarrier();
  // phase 2 (measured: it was the heavy phase, 2.9 ms of 5.2 with one f32 per thread per position): 64 groups of
  // 4 dimensions (vec4) x 4 lanes of positions; V read as vec4, contiguous across threads; the 4 lanes are summed in fq.
  let g = t % 64u; let lane = t / 64u;
  var acc: array<vec4<f32>, 8>;
  var end = min(n, start + FD_B);
  if (phase == 1u) { end = 0u; }
  for (var sp = start + lane; sp < end; sp += 4u) {
    let v = vv((sp * 512u + kvh * 256u) / 4u + g); let i = sp - start;
    for (var h: u32 = 0u; h < 8u; h++) { acc[h] += fs[h * FD_B + i] * v; }
  }
  for (var h: u32 = 0u; h < 8u; h++) {
    workgroupBarrier();
    if (lane > 0u) { fq[(lane - 1u) * 64u + g] = acc[h]; }
    workgroupBarrier();
    if (lane == 0u) {
      let r = acc[h] + fq[g] + fq[64u + g] + fq[128u + g];
      let base = ((tl * 16u + 8u * kvh + h) * nb + block) * 258u + 2u + 4u * g;
      o[base] = r.x; o[base + 1u] = r.y; o[base + 2u] = r.z; o[base + 3u] = r.w;
      if (g == 0u) { o[base - 2u] = fm[h]; o[base - 1u] = fsum[h]; }
    }
  }
}
// attn_fd2: the double step (MTP) with K and V read ONCE for both tokens. attn_fd launched one workgroup
// per token (grid z = 2), so every block of the cache was read twice: the MTP cycle paid the context twice. Here a
// workgroup (KV head, block of 128 positions) holds the q of 2 tokens x 8 heads (16 rows) and uses every K row for
// 16 scores and every V row for 16 accumulators. Token tl sees positions [extra, n + tl). Same partial output as
// attn_fd (per token, head, block), merged by attn_fd_combine.
var<workgroup> gq: array<vec4<f32>, 1024>;     // 16 rows (token * 8 + head) x 64 vec4
var<workgroup> gmx: array<f32, 16>;
var<workgroup> gsm: array<f32, 16>;
// attn_fd2b (speed path 1): attn_fd2 with blocks of 256 positions (half the q loads, partial results and
// merge work per position) and max / sum computed by all 256 threads (16 per row) instead of 16 threads alone.
// Partial results at the same place as attn_fd2 (stride p.rows), merged by attn_fd_combine with p.flag = 256.
const F2_B: u32 = 256u;
var<workgroup> hs: array<f32, 4096>;            // 16 rows x 256 scores
var<workgroup> hred: array<f32, 256>;
// attn_fd2c (speed path 1): attn_fd2b with less on-chip memory traffic. Phase 1: a thread does 2 positions
// (pos, pos + 128) at once, so each q vec4 read from shared memory serves two K rows. Scores stored as [position][row]
// (hs4), so phase 2 reads the 16 weights of a position with 4 vec4 reads instead of 16 scalar reads.
var<workgroup> hs4: array<vec4<f32>, 1024>;     // [position 256][row 16 as 4 vec4]
// attn_fd2h: attn_fd2c with 4 rows of V asked for at once in phase 2. Measured per piece:
// the V read is the most expensive piece of attn_fd2c although its lanes read consecutive addresses; test of
// the idea that too few reads are in flight (each group of 64 lanes read one row of 1 KB and used it at once).
@compute @workgroup_size(256)
fn attn_fd2h(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let kvh = wid.x; let block = wid.y;
  let scale = bitcast<f32>(p.f0); let nb = p.rows;
  let start = p.extra + block * F2_B;
  if (start >= p.n + 1u) { return; }
  for (var i = t; i < 1024u; i += 256u) {
    let r = i / 64u; let q0 = (r / 8u) * 4096u + (8u * kvh + r % 8u) * 256u + (i % 64u) * 4u;
    gq[i] = vec4<f32>(a[q0], a[q0 + 1u], a[q0 + 2u], a[q0 + 3u]);
  }
  workgroupBarrier();
  // phase 1: thread = (pair of positions pos / pos + 128, half of the row)
  let pos = t / 2u; let side = t % 2u;
  let sa = start + pos; let sb = sa + 128u;
  let va = sa < p.n + 1u; let vb = sb < p.n + 1u;
  var da: array<f32, 16>; var db: array<f32, 16>;
  if (va && p.f1 != 2u) {
    let ka = (sa * 512u + kvh * 256u) / 4u + side * 32u; let kb2 = (sb * 512u + kvh * 256u) / 4u + side * 32u;
    for (var i: u32 = 0u; i < 32u; i++) {
      let k4a = kk(ka + i);
      let k4b = select(vec4<f32>(0.0), kk(kb2 + i), vb);
      for (var r: u32 = 0u; r < 16u; r++) { let q = gq[r * 64u + side * 32u + i]; da[r] += dot(q, k4a); db[r] += dot(q, k4b); }
    }
  }
  // the two halves of each row meet in hs (scalar view of hs4 through the row index)
  if (side == 1u) {
    for (var r4: u32 = 0u; r4 < 4u; r4++) {
      hs4[pos * 4u + r4] = vec4<f32>(da[4u * r4], da[4u * r4 + 1u], da[4u * r4 + 2u], da[4u * r4 + 3u]);
      hs4[(pos + 128u) * 4u + r4] = vec4<f32>(db[4u * r4], db[4u * r4 + 1u], db[4u * r4 + 2u], db[4u * r4 + 3u]);
    }
  }
  workgroupBarrier();
  if (side == 0u) {
    for (var r4: u32 = 0u; r4 < 4u; r4++) {
      let pa = hs4[pos * 4u + r4]; let pb = hs4[(pos + 128u) * 4u + r4];
      var oa: vec4<f32>; var ob: vec4<f32>;
      for (var c: u32 = 0u; c < 4u; c++) {
        let r = 4u * r4 + c;
        oa[c] = select(-3.0e38, (da[r] + pa[c]) * scale, sa < p.n + r / 8u);
        ob[c] = select(-3.0e38, (db[r] + pb[c]) * scale, sb < p.n + r / 8u);
      }
      hs4[pos * 4u + r4] = oa; hs4[(pos + 128u) * 4u + r4] = ob;
    }
  }
  workgroupBarrier();
  // max and sum: row r = t / 16, 16 threads per row, 16 positions each
  let rr = t / 16u; let ln = t % 16u; let r4i = rr / 4u; let rc = rr % 4u;
  var mx = -3.0e38;
  for (var i: u32 = 0u; i < 16u; i++) { mx = max(mx, hs4[(ln * 16u + i) * 4u + r4i][rc]); }
  hred[t] = mx;
  workgroupBarrier();
  if (ln == 0u) { var m2 = -3.0e38; for (var i: u32 = 0u; i < 16u; i++) { m2 = max(m2, hred[t + i]); } gmx[rr] = m2; }
  workgroupBarrier();
  let rmx = gmx[rr];
  var sm = 0.0;
  for (var i: u32 = 0u; i < 16u; i++) {
    let j = (ln * 16u + i) * 4u + r4i; let v = hs4[j][rc];
    let w = select(0.0, exp(v - rmx), v > -1.0e38); hs4[j][rc] = w; sm += w;
  }
  hred[t] = sm;
  workgroupBarrier();
  if (ln == 0u) { var s2 = 0.0; for (var i: u32 = 0u; i < 16u; i++) { s2 += hred[t + i]; } gsm[rr] = s2; }
  workgroupBarrier();
  // phase 2: 64 groups of 4 dims x 4 lanes of positions; the 16 weights of a position in 4 vec4 reads
  let g = t % 64u; let lane = t / 64u;
  var acc: array<vec4<f32>, 16>;
  var end = min(p.n + 1u, start + F2_B);
  if (p.f1 == 1u) { end = 0u; }
  // 4 rows of V asked for before any of them is used (positions sp, sp+4, sp+8, sp+12 of this lane), then the rest
  var sp = start + lane;
  for (; sp + 12u < end; sp += 16u) {
    let b0 = (sp * 512u + kvh * 256u) / 4u + g;
    let v0 = vv(b0); let v1 = vv(b0 + 512u); let v2 = vv(b0 + 1024u); let v3 = vv(b0 + 1536u);
    let i = sp - start;
    for (var r4: u32 = 0u; r4 < 4u; r4++) {
      let w0 = hs4[i * 4u + r4]; let w1 = hs4[(i + 4u) * 4u + r4]; let w2 = hs4[(i + 8u) * 4u + r4]; let w3 = hs4[(i + 12u) * 4u + r4];
      acc[4u * r4] += w0.x * v0 + w1.x * v1 + w2.x * v2 + w3.x * v3;
      acc[4u * r4 + 1u] += w0.y * v0 + w1.y * v1 + w2.y * v2 + w3.y * v3;
      acc[4u * r4 + 2u] += w0.z * v0 + w1.z * v1 + w2.z * v2 + w3.z * v3;
      acc[4u * r4 + 3u] += w0.w * v0 + w1.w * v1 + w2.w * v2 + w3.w * v3;
    }
  }
  for (; sp < end; sp += 4u) {
    let v = vv((sp * 512u + kvh * 256u) / 4u + g); let i = sp - start;
    for (var r4: u32 = 0u; r4 < 4u; r4++) {
      let w = hs4[i * 4u + r4];
      acc[4u * r4] += w.x * v; acc[4u * r4 + 1u] += w.y * v; acc[4u * r4 + 2u] += w.z * v; acc[4u * r4 + 3u] += w.w * v;
    }
  }
  for (var r: u32 = 0u; r < 16u; r++) {
    workgroupBarrier();
    if (lane > 0u) { gq[(lane - 1u) * 64u + g] = acc[r]; }
    workgroupBarrier();
    if (lane == 0u) {
      let x = acc[r] + gq[g] + gq[64u + g] + gq[128u + g];
      let base = (((r / 8u) * 16u + 8u * kvh + r % 8u) * nb + block) * 258u + 2u + 4u * g;
      o[base] = x.x; o[base + 1u] = x.y; o[base + 2u] = x.z; o[base + 3u] = x.w;
      if (g == 0u) { o[base - 2u] = gmx[r]; o[base - 1u] = gsm[r]; }
    }
  }
}
// attn_fd2f: attn_fd2c with phase 1 reading K with consecutive lanes on consecutive addresses. Measured:
// the K read of phase 1 held attn_fd2c at 55 GB/s, while the same grid
// reads K and V at 78-79 GB/s that way. K goes through shared memory in chunks of 16 positions (row stride 65 vec4,
// against bank conflicts); in phase 1 a thread is (position of the chunk, query row) and does the whole dot product.
// Softmax and phase 2 as attn_fd2c.
var<workgroup> kt: array<vec4<f32>, 1040>;   // [position 16][65 vec4: 64 of the row + 1 of padding]
// merge of the blocks: workgroup (head, token), thread = dimension d. a = partials, p.rows = nb.
@compute @workgroup_size(256)
fn attn_fd_combine(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let h = wid.x; let tl = wid.y; let tok = tl + p.step_a; let nb = p.rows;
  let bb = select(FD_B, p.flag, p.flag > 0u);           // positions per block (attn_fd: 128; attn_fd4 of the prefill: 64)
  let nbv = (p.n + tok - p.extra + bb - 1u) / bb;       // valid blocks (the others are not written)
  var M = -3.0e38;
  for (var b: u32 = 0u; b < nbv; b++) { M = max(M, a[((tl * 16u + h) * nb + b) * 258u]); }
  var tot = 0.0; var acc = 0.0;
  for (var b: u32 = 0u; b < nbv; b++) {
    let base = ((tl * 16u + h) * nb + b) * 258u;
    let e = exp(a[base] - M);
    tot += e * a[base + 1u]; acc += e * a[base + 2u + t];
  }
  o[tok * 4096u + h * 256u + t] = acc / tot;
}

// ===================== attn_fa: flash attention for the prefill =====================
// At 100K attn_fd4 + the merge were 25 of 28 ms per token, far from what the GPU can do: every
// K/V value read from shared memory fed only 1-2 multiply-adds. Here a workgroup = (KV head, chunk of FA_CH positions,
// group of 8 tokens): 64 rows (8 tokens x 8 heads that share the KV head). For each tile of 64 keys:
//   S = Q K^T as a 64 x 64 tile, each thread 4 rows x 4 keys, dims in pieces of 32 through shared memory (as gemm.wgsl);
//   online softmax per row (running max and sum); O += P V, each thread 8 rows (one token, 8 heads) x 8 dims, V in
//   pieces of 16 keys. Token T (index in the prefill block) sees positions [p.extra, p.n + T).
// Output: partials [max, sum, acc 256] per (token of the dispatch, head, chunk), merged by attn_fd_combine with
// p.flag = FA_CH. p.step_a = first token of the dispatch (64 tokens, grid z = 8 groups), p.rows = chunks in the
// context (CTX / FA_CH), p.step_o = first chunk of this dispatch: a dispatch covers at most FA_SPLIT chunks (Rust side),
// so one submit stays short at long context (the desktop needs the GPU too).
const FA_CH: u32 = 2048u;
const FA_QK: u32 = 9u;                          // vec4 per row of Qs/Ks (8 + 1 of padding against bank conflicts)
// Phase profile: QK was 69% of the kernel, limited by shared-memory reads. Q, K and V sit in
// shared memory as packed f16 (vec2<u32> = 4 values): half the bytes per multiply-add; products and sums stay f32.
var<workgroup> fab: array<vec2<u32>, 1152>;     // QK: Qs [row 64][9] at 0, Ks [key 64][9] at 576 · PV: Vs [key 16][dim/4 64]
var<workgroup> fas: array<vec4<f32>, 1088>;     // S then P: [key 64][row/4 16 + 1 of padding]: PV reads 8 rows as 2 vec4
var<workgroup> fam: array<f32, 64>;             // running max per row
var<workgroup> fal: array<f32, 64>;             // running sum per row
var<workgroup> fac: array<f32, 64>;             // rescale of the accumulators of this tile
fn pk4(v: vec4<f32>) -> vec2<u32> { return vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw)); }
fn up4(w: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
@compute @workgroup_size(256)
fn attn_fa(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let kvh = wid.x; let chunk = wid.y + p.step_o; let grp = wid.z;   // step_o: first chunk of this dispatch
  let scale = bitcast<f32>(p.f0); let nb = p.rows;
  let tb = p.step_a + 8u * grp;                    // block index of the first token of this group
  let cs = p.extra + chunk * FA_CH;
  let lim = p.n + tb + 7u;                          // positions seen by the last token of the group: [extra, lim)
  if (cs >= lim) { return; }
  let ntile = min(FA_CH / 64u, (lim - cs + 63u) / 64u);
  let phase = p.f1;                                  // measuring only (MECHTA_FA_PHASE): 1 = no QK, 2 = no softmax, 3 = no PV, 4 = no Q/K loads
  if (t < 64u) { fam[t] = -3.0e38; fal[t] = 0.0; }
  let rg = t / 16u; let kg = t % 16u;               // QK: rows rg*4.., keys kg*4..
  let tq = t / 32u; let dg = t % 32u;               // PV: token tq (rows tq*8..), dims dg*8..
  var acc: array<vec4<f32>, 16>;                    // PV: [head i][2 vec4]
  for (var tile: u32 = 0u; tile < ntile; tile++) {
    let k0 = cs + tile * 64u;
    var sc: array<vec4<f32>, 4>;                    // scores: [row i] = 4 keys
    for (var dp: u32 = 0u; dp < 8u; dp++) {
      if (phase == 1u) { break; }
      workgroupBarrier();
      for (var i = t; i < 512u; i += 256u) {
        if (phase == 4u) { break; }
        let r = i / 8u; let j = i % 8u; let d0 = dp * 32u + j * 4u;
        let q0 = (tb + r / 8u) * 4096u + (8u * kvh + r % 8u) * 256u + d0;
        fab[r * FA_QK + j] = pk4(vec4<f32>(a[q0], a[q0 + 1u], a[q0 + 2u], a[q0 + 3u]));
        fab[576u + r * FA_QK + j] = pk4(kk(((k0 + r) * 512u + kvh * 256u + d0) / 4u));
      }
      workgroupBarrier();
      for (var j: u32 = 0u; j < 8u; j++) {
        let q_0 = up4(fab[(rg * 4u) * FA_QK + j]); let q_1 = up4(fab[(rg * 4u + 1u) * FA_QK + j]);
        let q_2 = up4(fab[(rg * 4u + 2u) * FA_QK + j]); let q_3 = up4(fab[(rg * 4u + 3u) * FA_QK + j]);
        let k_0 = up4(fab[576u + (kg * 4u) * FA_QK + j]); let k_1 = up4(fab[576u + (kg * 4u + 1u) * FA_QK + j]);
        let k_2 = up4(fab[576u + (kg * 4u + 2u) * FA_QK + j]); let k_3 = up4(fab[576u + (kg * 4u + 3u) * FA_QK + j]);
        sc[0] += vec4<f32>(dot(q_0, k_0), dot(q_0, k_1), dot(q_0, k_2), dot(q_0, k_3));
        sc[1] += vec4<f32>(dot(q_1, k_0), dot(q_1, k_1), dot(q_1, k_2), dot(q_1, k_3));
        sc[2] += vec4<f32>(dot(q_2, k_0), dot(q_2, k_1), dot(q_2, k_2), dot(q_2, k_3));
        sc[3] += vec4<f32>(dot(q_3, k_0), dot(q_3, k_1), dot(q_3, k_2), dot(q_3, k_3));
      }
    }
    // scaled and masked scores into fas
    let vis = p.n + tb + rg / 2u;                       // rows rg*4.. are one token: it sees positions < vis
    for (var c: u32 = 0u; c < 4u; c++) {
      let s = k0 + kg * 4u + c;
      fas[(kg * 4u + c) * 17u + rg] = select(vec4<f32>(-3.0e38), vec4<f32>(sc[0][c], sc[1][c], sc[2][c], sc[3][c]) * scale, s < vis);
    }
    workgroupBarrier();
    // online softmax: one thread per row
    if (t < 64u && phase != 2u) {
      let mo = fam[t];
      var mx = mo;
      let r4 = t / 4u; let rc = t % 4u;
      for (var c: u32 = 0u; c < 64u; c++) { mx = max(mx, fas[c * 17u + r4][rc]); }
      var sm = 0.0;
      for (var c: u32 = 0u; c < 64u; c++) { let v = fas[c * 17u + r4][rc]; let w = select(0.0, exp(v - mx), v > -1.0e38); fas[c * 17u + r4][rc] = w; sm += w; }
      let cf = select(0.0, exp(mo - mx), mo > -1.0e38);
      fac[t] = cf; fam[t] = mx; fal[t] = fal[t] * cf + sm;
    }
    workgroupBarrier();
    for (var i: u32 = 0u; i < 8u; i++) { let cf = fac[tq * 8u + i]; acc[2u * i] *= cf; acc[2u * i + 1u] *= cf; }
    // O += P V, V in pieces of 16 keys
    for (var vp: u32 = 0u; vp < 4u; vp++) {
      if (phase == 3u) { break; }
      workgroupBarrier();
      for (var i = t; i < 1024u; i += 256u) {
        let kr = i / 64u; let d4 = i % 64u;
        fab[i] = pk4(vv(((k0 + vp * 16u + kr) * 512u + kvh * 256u) / 4u + d4));
      }
      workgroupBarrier();
      for (var kr: u32 = 0u; kr < 16u; kr++) {
        let v0 = up4(fab[kr * 64u + dg * 2u]); let v1 = up4(fab[kr * 64u + dg * 2u + 1u]);
        let col = vp * 16u + kr;
        let pa = fas[col * 17u + tq * 2u]; let pb = fas[col * 17u + tq * 2u + 1u];   // heads 0-3 and 4-7 of token tq
        acc[0] += pa.x * v0; acc[1] += pa.x * v1; acc[2] += pa.y * v0; acc[3] += pa.y * v1;
        acc[4] += pa.z * v0; acc[5] += pa.z * v1; acc[6] += pa.w * v0; acc[7] += pa.w * v1;
        acc[8] += pb.x * v0; acc[9] += pb.x * v1; acc[10] += pb.y * v0; acc[11] += pb.y * v1;
        acc[12] += pb.z * v0; acc[13] += pb.z * v1; acc[14] += pb.w * v0; acc[15] += pb.w * v1;
      }
    }
  }
  workgroupBarrier();
  // partials: token of the dispatch = 8 * grp + tq, head 8 * kvh + i
  for (var i: u32 = 0u; i < 8u; i++) {
    let r = tq * 8u + i;
    let base = (((8u * grp + tq) * 16u + 8u * kvh + i) * nb + chunk) * 258u;
    let o0 = acc[2u * i]; let o1 = acc[2u * i + 1u];
    o[base + 2u + dg * 8u] = o0.x; o[base + 3u + dg * 8u] = o0.y; o[base + 4u + dg * 8u] = o0.z; o[base + 5u + dg * 8u] = o0.w;
    o[base + 6u + dg * 8u] = o1.x; o[base + 7u + dg * 8u] = o1.y; o[base + 8u + dg * 8u] = o1.z; o[base + 9u + dg * 8u] = o1.w;
    if (dg == 0u) { o[base] = fam[r]; o[base + 1u] = fal[r]; }
  }
}

// ===================== M3: kernels for the prefill in blocks of NB tokens =====================
// conv_n: like conv2 over rows = n tokens (uniform rewritten at each block), without the intermediate state; v at extra*4096
// (extra = fixed NB) + t*4096, so the q|k rows of all tokens stay contiguous and their norm does not depend on n.
@compute @workgroup_size(64)
fn conv_n(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x; if (ch >= p.n) { return; }
  var s0 = c[ch * 3u]; var s1 = c[ch * 3u + 1u]; var s2 = c[ch * 3u + 2u];
  let w = ch * 4u;
  for (var t: u32 = 0u; t < p.rows; t++) {
    let fresh = a[t * p.n + ch];
    let s = s0 * b[w] + s1 * b[w + 1u] + s2 * b[w + 2u] + fresh * b[w + 3u];
    var oi = t * 4096u + ch;
    if (ch >= 4096u) { oi = p.extra * 4096u + t * 4096u + ch - 4096u; }
    o[oi] = silu(s);
    s0 = s1; s1 = s2; s2 = fresh;
  }
  o2[ch * 3u] = s0; o2[ch * 3u + 1u] = s1; o2[ch * 3u + 2u] = s2;
}
// delta_n: the recurrence over rows = n tokens in one dispatch, state in registers for the whole block (read and written
// once). a = normalized q|k [t*4096 + ...], b = output of conv_n (v at extra*4096 + t*4096 + h*128 + j),
// c = beta|gate [t*64 + ...]. o = state, o3 = outputs [t*4096 + h*128 + j].
@compute @workgroup_size(128)
fn delta_n(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let h = wid.x; let kh = h % 16u;
  let s0 = h * 16384u + j;
  var col: array<f32, 128>;
  for (var i: u32 = 0u; i < 128u; i++) { col[i] = o[s0 + i * 128u]; }
  for (var t: u32 = 0u; t < p.rows; t++) {
    let q0 = t * 4096u + kh * 128u; let k0 = t * 4096u + 2048u + kh * 128u;
    let v = b[p.extra * 4096u + t * 4096u + h * 128u + j];
    let beta = c[t * 64u + h]; let eg = exp(c[t * 64u + 32u + h]);
    var sk = 0.0;
    for (var i: u32 = 0u; i < 128u; i++) { let m = col[i] * eg; col[i] = m; sk += m * a[k0 + i]; }
    let dl = (v - sk) * beta;
    var sq = 0.0;
    for (var i: u32 = 0u; i < 128u; i++) { let m = col[i] + a[k0 + i] * dl; col[i] = m; sq += m * a[q0 + i]; }
    o3[t * 4096u + h * 128u + j] = sq * (1.0 / sqrt(128.0));
  }
  for (var i: u32 = 0u; i < 128u; i++) { o[s0 + i * 128u] = col[i]; }
}

// ===================== Prefill: attention with K/V read once per 4 tokens =====================
// attn_fd does each token on its own: in the prefill at long context (30K) K and V are reread for every token and the prompt
// costs 15 ms/token. Here a workgroup = (KV head, block of 64 positions); two passes of 4 tokens of the batch of 8:
// 32 rows of q (4 tokens x 8 heads) in shared memory (f32, 32 KB); phase 1: thread = (position, group of 8 rows),
// reads its K row ONCE and computes 8 scores; phase 2: thread = (group of 4 dimensions, group of 8 rows), reads
// V once per position and accumulates 8 rows. Token tok sees positions [extra, n + tok) (n = pos + 1 of the block).
// Output as attn_fd (partials [max, sum, acc 256] per (token of the batch, head, block)), merged with attn_fd_combine
// (p.flag = positions per block). p.step_a = first token of the batch, p.rows = blocks launched (CTX / p.flag).
// A workgroup covers p.flag positions (64 = default; 128-512 with MECHTA_PREFILL_BLOCK), in tiles of 64
// with a running max and sum per row (online softmax): 4x fewer partials for attn_fd_combine to merge at long context
// (the merge was 4.3 of 25.5 ms/token at 100K).
const F4_B: u32 = 64u;
var<workgroup> q4s: array<vec4<f32>, 2048>;   // [row 32][vec4 64]
var<workgroup> s4: array<f32, 2048>;          // [row 32][position 64]
var<workgroup> m4: array<f32, 32>;
var<workgroup> l4: array<f32, 32>;
var<workgroup> c4s: array<f32, 32>;           // per row: rescale of the accumulators when the max grows
@compute @workgroup_size(256)
fn attn_fd4(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let kvh = wid.x; let block = wid.y;
  let scale = bitcast<f32>(p.f0);
  let s0 = p.extra; let nb = p.rows; let t0 = p.step_a;
  let bl = select(F4_B, p.flag, p.flag > 0u);          // positions per workgroup (multiple of 64)
  let start = s0 + block * bl;
  if (start >= p.n + t0 + 7u) { return; }            // no token of the batch sees this block
  for (var round_i: u32 = 0u; round_i < 2u; round_i++) {
    let tb = t0 + 4u * round_i;                          // first token of this pass
    // q of the 32 rows (row = tl * 8 + h)
    for (var i = t; i < 2048u; i += 256u) {
      let row = i / 64u; let tl = row / 8u; let h = row % 8u; let d0 = (i % 64u) * 4u;
      let q0 = (tb + tl) * 4096u + (8u * kvh + h) * 256u + d0;
      q4s[i] = vec4<f32>(a[q0], a[q0 + 1u], a[q0 + 2u], a[q0 + 3u]);
    }
    if (t < 32u) { m4[t] = -3.0e38; l4[t] = 0.0; }
    workgroupBarrier();
    let ps = t % 64u; let rg = t / 64u;               // phase 1: thread (position ps, row group rg)
    let g = t % 64u; let rq = t / 64u;                // phase 2: thread (group of 4 dimensions g, row group rq)
    var acc: array<vec4<f32>, 8>;
    let end_tok = p.n + tb + 3u;                     // positions seen by the last token of this pass: [s0, end_tok)
    let ntile = select(0u, min(bl / F4_B, (end_tok - start + F4_B - 1u) / F4_B), end_tok > start);   // uniform: tiles with a valid position
    for (var tile: u32 = 0u; tile < ntile; tile++) {
      let t_in = start + tile * F4_B;
      // phase 1: scores of this tile
      let s = t_in + ps;
      var d8: array<f32, 8>;
      if (s < end_tok) {
        let kb = (s * 512u + kvh * 256u) / 4u;
        for (var i: u32 = 0u; i < 64u; i++) {
          let k4 = kk(kb + i);
          for (var r: u32 = 0u; r < 8u; r++) { d8[r] += dot(q4s[(rg * 8u + r) * 64u + i], k4); }
        }
      }
      for (var r: u32 = 0u; r < 8u; r++) {
        let row = rg * 8u + r; let tl = row / 8u;
        s4[row * 64u + ps] = select(-3.0e38, d8[r] * scale, s < p.n + tb + tl);
      }
      workgroupBarrier();
      // running max and sum per row; c4s = how much the old accumulators shrink
      if (t < 32u) {
        var mx = m4[t];
        for (var i: u32 = 0u; i < 64u; i++) { mx = max(mx, s4[t * 64u + i]); }
        let cf = select(0.0, exp(m4[t] - mx), m4[t] > -1.0e38);
        var sm = 0.0;
        for (var i: u32 = 0u; i < 64u; i++) { let v = s4[t * 64u + i]; let w = select(0.0, exp(v - mx), v > -1.0e38); s4[t * 64u + i] = w; sm += w; }
        c4s[t] = cf; m4[t] = mx; l4[t] = l4[t] * cf + sm;
      }
      workgroupBarrier();
      // phase 2: rescale, then add this tile's weighted V
      for (var r: u32 = 0u; r < 8u; r++) { acc[r] *= c4s[rq * 8u + r]; }
      let end = min(end_tok, t_in + F4_B);
      for (var sp = t_in; sp < end; sp++) {
        let v = vv((sp * 512u + kvh * 256u) / 4u + g); let i = sp - t_in;
        for (var r: u32 = 0u; r < 8u; r++) { acc[r] += s4[(rq * 8u + r) * 64u + i] * v; }
      }
      workgroupBarrier();
    }
    for (var r: u32 = 0u; r < 8u; r++) {
      let row = rq * 8u + r; let tl = row / 8u; let h = row % 8u;
      let base = (((4u * round_i + tl) * 16u + 8u * kvh + h) * nb + block) * 258u;
      o[base + 2u + 4u * g] = acc[r].x; o[base + 3u + 4u * g] = acc[r].y; o[base + 4u + 4u * g] = acc[r].z; o[base + 5u + 4u * g] = acc[r].w;
      if (g == 0u) { o[base] = m4[row]; o[base + 1u] = l4[row]; }
    }
    workgroupBarrier();
  }
}

// ===================== k-token verify step (quad.rs) =====================
// The state is read once and written once per step: first the rows ACCEPTED by the previous step are applied (their
// inputs were saved by that step), the committed state is written, then the new rows run on a copy in registers for
// their outputs only, and their inputs are saved for the next step. p.rows = new rows (0 = commit only),
// p.extra = pending rows, p.flag = row layout of the buffers (K = 4).
// conv_q: a = new qkv [t*8192 + ch], b = conv1d weights, c = committed state in, d = pending raw qkv
// [t*8192 + ch], o = output in "qk first" layout (v at flag*4096 + t*4096), o2 = committed state out,
// o3 = this step's raw qkv saved for the next one.
@compute @workgroup_size(64)
fn conv_q(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ch = gid.x; if (ch >= p.n) { return; }
  var s0 = c[ch * 3u]; var s1 = c[ch * 3u + 1u]; var s2 = c[ch * 3u + 2u];
  let w = ch * 4u;
  for (var t: u32 = 0u; t < p.extra; t++) { s0 = s1; s1 = s2; s2 = d[t * p.n + ch]; }
  o2[ch * 3u] = s0; o2[ch * 3u + 1u] = s1; o2[ch * 3u + 2u] = s2;
  for (var t: u32 = 0u; t < p.rows; t++) {
    let fresh = a[t * p.n + ch];
    let s = s0 * b[w] + s1 * b[w + 1u] + s2 * b[w + 2u] + fresh * b[w + 3u];
    var oi = t * 4096u + ch;
    if (ch >= 4096u) { oi = p.flag * 4096u + t * 4096u + ch - 4096u; }
    o[oi] = silu(s);
    s0 = s1; s1 = s2; s2 = fresh;
    o3[t * p.n + ch] = fresh;
  }
}
// delta_q: one workgroup per V head (32), one thread per column j. a = normalized q|k [t*4096 + ...], b = conv_q output
// (v at flag*4096 + t*4096 + h*128 + j), c = beta|gate [t*64 + ...], d = pending inputs, o = committed state (read and
// written once), o2 = this step's inputs saved, o3 = outputs [t*4096 + h*128 + j].
// Saved inputs per row (DQ_ROW values): k [16 heads x 128] at 0, v [32 x 128] at 2048, beta [32] at 6144, gate [32] at 6176.
const DQ_ROW: u32 = 6208u;
@compute @workgroup_size(128)
fn delta_q(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) j: u32) {
  let h = wid.x; let kh = h % 16u;
  let s0 = h * 16384u + j;
  var col: array<f32, 128>;
  for (var i: u32 = 0u; i < 128u; i++) { col[i] = o[s0 + i * 128u]; }
  for (var t: u32 = 0u; t < p.extra; t++) {
    let r = t * DQ_ROW; let k0 = r + kh * 128u;
    let v = d[r + 2048u + h * 128u + j];
    let beta = d[r + 6144u + h]; let eg = exp(d[r + 6176u + h]);
    var sk = 0.0;
    for (var i: u32 = 0u; i < 128u; i++) { let m = col[i] * eg; col[i] = m; sk += m * d[k0 + i]; }
    let dl = (v - sk) * beta;
    for (var i: u32 = 0u; i < 128u; i++) { col[i] += d[k0 + i] * dl; }
  }
  for (var i: u32 = 0u; i < 128u; i++) { o[s0 + i * 128u] = col[i]; }
  for (var t: u32 = 0u; t < p.rows; t++) {
    let q0 = t * 4096u + kh * 128u; let k0 = t * 4096u + 2048u + kh * 128u;
    let v = b[p.flag * 4096u + t * 4096u + h * 128u + j];
    let beta = c[t * 64u + h]; let g = c[t * 64u + 32u + h]; let eg = exp(g);
    var sk = 0.0;
    for (var i: u32 = 0u; i < 128u; i++) { let m = col[i] * eg; col[i] = m; sk += m * a[k0 + i]; }
    let dl = (v - sk) * beta;
    var sq = 0.0;
    for (var i: u32 = 0u; i < 128u; i++) { let m = col[i] + a[k0 + i] * dl; col[i] = m; sq += m * a[q0 + i]; }
    o3[t * 4096u + h * 128u + j] = sq * (1.0 / sqrt(128.0));
    let r = t * DQ_ROW;
    if (h < 16u) { o2[r + kh * 128u + j] = a[k0 + j]; }
    o2[r + 2048u + h * 128u + j] = v;
    if (j == 0u) { o2[r + 6144u + h] = beta; o2[r + 6176u + h] = g; }
  }
}
