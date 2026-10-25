// Flint, one token: RoPE (NEOX, 128 dims), K/V write into the cache, attention with per-head sinks.
// Reference: llama.cpp src/models/flint.cpp + ggml soft_max with sinks (the sink logit joins max and sum, not the output).
// Cache layout: row = slot = position % ring, 4 KV heads x 128 = 512 floats. Full layers: ring = ctx (no wrap).
// Sliding layers: ring = 1024, so a prefill block of up to 511 + 1 tokens never overwrites a position that a token of
// the same block still sees (it sees the current position and the 512 before it).
// One token: keys at positions lo .. lo + n - 1, the new K/V at row `slot`, scratch stride ctx.
// Prefill: n = tokens of the block, first at position pos0; win = 513 on sliding layers, 0 = no window; g0 = first group
// of 8 tokens of this dispatch.
struct P { n: u32, slot: u32, lo: u32, ring: u32, ctx: u32, pos0: u32, win: u32, g0: u32 }
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read> c: array<f32>;
@group(0) @binding(3) var<storage, read> d: array<f32>;
@group(0) @binding(4) var<storage, read_write> sc: array<f32>;
@group(0) @binding(5) var<storage, read_write> o: array<f32>;
@group(0) @binding(6) var<storage, read_write> o2: array<f32>;
@group(0) @binding(7) var<uniform> p: P;
// KV-FORMAT: the Rust side puts here the block of the chosen cache format (f32 or f16, see flint.rs). Every block
// declares the K/V caches at bindings 9, 10 (read) and 11, 12 (written) and: k4(i) / v4(i) = elements 4 i .. 4 i + 3 of
// K / V as vec4<f32>; kv_put(i, k, v) = elements 2 i, 2 i + 1 of K and of V.

// RoPE NEOX: a = q or k (p.heads x 128), b = table (cos of the 64 pairs, then sin), o = rotated.
// Pair i is (i, i + 64), angle = pos * 10000^(-2i/128) (table from the CPU, as ggml_rope_cache_init).
@compute @workgroup_size(64)
fn rope_neox(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let h0 = wid.x * 128u;
  let x0 = a[h0 + t]; let x1 = a[h0 + t + 64u];
  let cs = b[t]; let sn = b[64u + t];
  o[h0 + t] = x0 * cs - x1 * sn;
  o[h0 + t + 64u] = x0 * sn + x1 * cs;
}

// K (a) and V (b), 512 values each, into the caches at row p.slot: thread i writes elements 2 i, 2 i + 1.
@compute @workgroup_size(64)
fn kv_write(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= 256u) { return; }
  kv_put(p.slot * 256u + i, vec2<f32>(a[2u * i], a[2u * i + 1u]), vec2<f32>(b[2u * i], b[2u * i + 1u]));
}

// One workgroup per query head (32), one thread per dimension (128). a = q (32 x 128), b = K cache, c = V cache,
// d = sinks (32), sc = scratch (32 x p.ctx), o = output (32 x 128). Positions p.lo .. p.lo + p.n - 1 are read.
// Query head h reads KV head h / 8. Scores scaled by 1/sqrt(128) before the softmax, as ggml.

// ===================== prefill of a block (F4) =====================
// RoPE of a block: workgroup (head, token); table of token t at b[t * 128]. Rows in a: (token * heads + head) * 128.
@compute @workgroup_size(64)
fn rope_neox_n(@builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nw: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let h0 = (wid.y * nw.x + wid.x) * 128u; let tb = wid.y * 128u;
  let x0 = a[h0 + t]; let x1 = a[h0 + t + 64u];
  let cs = b[tb + t]; let sn = b[tb + 64u + t];
  o[h0 + t] = x0 * cs - x1 * sn;
  o[h0 + t + 64u] = x0 * sn + x1 * cs;
}

// K (a) and V (b) of the p.n tokens of the block into the caches: token t at row (p.pos0 + t) % p.ring.
@compute @workgroup_size(64)
fn kv_write_n(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; let tk = gid.y; if (i >= 256u || tk >= p.n) { return; }
  let r = (p.pos0 + tk) % p.ring; let x = tk * 512u + 2u * i;
  kv_put(r * 256u + i, vec2<f32>(a[x], a[x + 1u]), vec2<f32>(b[x], b[x + 1u]));
}

// Flash attention of a block, as attn_fa of ops.wgsl (Qwen) for Flint: workgroup = (KV head, group of 8 tokens),
// 64 rows = 8 tokens x the 8 query heads of that KV head. Tiles of 64 keys: S = Q K^T (each thread 4 rows x 4 keys,
// dims in pieces of 32 through shared memory as packed f16), online softmax per row, O += P V (each thread 8 rows of
// one token x 4 dims, V in pieces of 16 keys). The sink of the row's head starts the softmax: running max = sink,
// running sum = 1 (its exp), so it joins max and sum and never the output, as ggml. Token T (position pos0 + T)
// sees [lo_T, pos0 + T], lo_T = pos - (win - 1) on sliding layers. a = Q of the block [t * 4096 + head * 128],
// b = K cache, c = V cache, d = sinks, o = output [t * 4096 + head * 128]. Grid (4, 1, groups of this dispatch).
var<workgroup> fab: array<vec2<u32>, 1152>;
var<workgroup> fas: array<vec4<f32>, 1088>;
var<workgroup> fam: array<f32, 64>;
var<workgroup> fal: array<f32, 64>;
var<workgroup> fac: array<f32, 64>;
fn pk4(v: vec4<f32>) -> vec2<u32> { return vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw)); }
fn up4(w: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn kv4(buf: u32, pos: u32, kvh: u32, d0: u32) -> vec4<f32> {
  let i = ((pos % p.ring) * 512u + kvh * 128u + d0) / 4u;
  if (buf == 0u) { return k4(i); }
  return v4(i);
}
fn lo_di(pos: u32) -> u32 { if (p.win > 0u && pos + 1u > p.win) { return pos + 1u - p.win; } return 0u; }
const FQ: u32 = 9u;
@compute @workgroup_size(256)
fn attn_pf(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let kvh = wid.x; let tb = 8u * (wid.z + p.g0);
  if (tb >= p.n) { return; }
  let last = min(tb + 7u, p.n - 1u);
  let last_pos = p.pos0 + last;
  let lo = lo_di(p.pos0 + tb);
  let ntile = (last_pos + 64u - lo) / 64u;
  let scale = 1.0 / sqrt(128.0);
  if (t < 64u) { fam[t] = d[8u * kvh + t % 8u]; fal[t] = 1.0; }
  let rg = t / 16u; let kg = t % 16u;
  let tq = t / 32u; let dg = t % 32u;
  var acc: array<vec4<f32>, 8>;
  for (var tile: u32 = 0u; tile < ntile; tile++) {
    let k0 = lo + tile * 64u;
    var sc: array<vec4<f32>, 4>;
    for (var dp: u32 = 0u; dp < 4u; dp++) {
      workgroupBarrier();
      for (var i = t; i < 512u; i += 256u) {
        let r = i / 8u; let j = i % 8u; let d0 = dp * 32u + j * 4u;
        let tk = tb + r / 8u;
        var q = vec4<f32>(0.0);
        if (tk < p.n) { let q0 = tk * 4096u + (8u * kvh + r % 8u) * 128u + d0; q = vec4<f32>(a[q0], a[q0 + 1u], a[q0 + 2u], a[q0 + 3u]); }
        fab[r * FQ + j] = pk4(q);
        var kv = vec4<f32>(0.0);
        if (k0 + r <= last_pos) { kv = kv4(0u, k0 + r, kvh, d0); }
        fab[576u + r * FQ + j] = pk4(kv);
      }
      workgroupBarrier();
      for (var j: u32 = 0u; j < 8u; j++) {
        let q_0 = up4(fab[(rg * 4u) * FQ + j]); let q_1 = up4(fab[(rg * 4u + 1u) * FQ + j]);
        let q_2 = up4(fab[(rg * 4u + 2u) * FQ + j]); let q_3 = up4(fab[(rg * 4u + 3u) * FQ + j]);
        let k_0 = up4(fab[576u + (kg * 4u) * FQ + j]); let k_1 = up4(fab[576u + (kg * 4u + 1u) * FQ + j]);
        let k_2 = up4(fab[576u + (kg * 4u + 2u) * FQ + j]); let k_3 = up4(fab[576u + (kg * 4u + 3u) * FQ + j]);
        sc[0] += vec4<f32>(dot(q_0, k_0), dot(q_0, k_1), dot(q_0, k_2), dot(q_0, k_3));
        sc[1] += vec4<f32>(dot(q_1, k_0), dot(q_1, k_1), dot(q_1, k_2), dot(q_1, k_3));
        sc[2] += vec4<f32>(dot(q_2, k_0), dot(q_2, k_1), dot(q_2, k_2), dot(q_2, k_3));
        sc[3] += vec4<f32>(dot(q_3, k_0), dot(q_3, k_1), dot(q_3, k_2), dot(q_3, k_3));
      }
    }
    // rows rg*4 .. rg*4+3 are 4 heads of token rg / 2
    let pt = p.pos0 + tb + rg / 2u; let lt = lo_di(pt);
    for (var cc: u32 = 0u; cc < 4u; cc++) {
      let s = k0 + kg * 4u + cc;
      fas[(kg * 4u + cc) * 17u + rg] = select(vec4<f32>(-3.0e38), vec4<f32>(sc[0][cc], sc[1][cc], sc[2][cc], sc[3][cc]) * scale, s >= lt && s <= pt);
    }
    workgroupBarrier();
    if (t < 64u) {
      let mo = fam[t];
      var mx = mo;
      let r4 = t / 4u; let rc = t % 4u;
      for (var cc: u32 = 0u; cc < 64u; cc++) { mx = max(mx, fas[cc * 17u + r4][rc]); }
      var sm = 0.0;
      for (var cc: u32 = 0u; cc < 64u; cc++) { let v = fas[cc * 17u + r4][rc]; let w = select(0.0, exp(v - mx), v > -1.0e38); fas[cc * 17u + r4][rc] = w; sm += w; }
      let cf = exp(mo - mx);
      fac[t] = cf; fam[t] = mx; fal[t] = fal[t] * cf + sm;
    }
    workgroupBarrier();
    for (var i: u32 = 0u; i < 8u; i++) { acc[i] *= fac[tq * 8u + i]; }
    for (var vp: u32 = 0u; vp < 4u; vp++) {
      workgroupBarrier();
      for (var i = t; i < 512u; i += 256u) {
        let kr = i / 32u; let d4 = i % 32u; let kp = k0 + vp * 16u + kr;
        var v = vec4<f32>(0.0);
        if (kp <= last_pos) { v = kv4(1u, kp, kvh, d4 * 4u); }
        fab[i] = pk4(v);
      }
      workgroupBarrier();
      for (var kr: u32 = 0u; kr < 16u; kr++) {
        let v0 = up4(fab[kr * 32u + dg]);
        let col = vp * 16u + kr;
        let pa = fas[col * 17u + tq * 2u]; let pb = fas[col * 17u + tq * 2u + 1u];
        acc[0] += pa.x * v0; acc[1] += pa.y * v0; acc[2] += pa.z * v0; acc[3] += pa.w * v0;
        acc[4] += pb.x * v0; acc[5] += pb.y * v0; acc[6] += pb.z * v0; acc[7] += pb.w * v0;
      }
    }
  }
  workgroupBarrier();
  let tk = tb + tq;
  if (tk < p.n) {
    for (var i: u32 = 0u; i < 8u; i++) {
      let v = acc[i] / fal[tq * 8u + i];
      let o0 = tk * 4096u + (8u * kvh + i) * 128u + dg * 4u;
      o[o0] = v.x; o[o0 + 1u] = v.y; o[o0 + 2u] = v.z; o[o0 + 3u] = v.w;
    }
  }
}

// ===================== generation: flash-decoding =====================
// attn_sink above has one workgroup per query head: every K/V row is read by the 8 query heads of its KV head, 8 times
// (157 ms of 193 per token at 30K). Here pass 1 = workgroup (KV head g, piece of FD_CH positions) with the 8 query heads
// of g together: each K/V row read once. Partials per (query head, piece) in sc: [max, sum, acc 128] at
// ((head * (ctx / FD_CH)) + piece) * 130. Pass 2 (attn_fd_merge) merges the pieces and adds the sink, as attn_sink:
// the sink joins max and sum, not the output. Same positions as attn_sink: p.lo .. p.lo + p.n - 1, row = pos % p.ring.
const FD_CH: u32 = 256u;
var<workgroup> fq: array<f32, 1024>;       // the 8 query heads of g
var<workgroup> fp: array<f32, 2048>;       // [head 8][position 256]: scores, then exp
var<workgroup> fr: array<f32, 256>;        // reductions, 32 threads per head
@compute @workgroup_size(256)
fn attn_fd(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let g = wid.x; let ch = wid.y; let j0 = ch * FD_CH;
  if (j0 >= p.n) { return; }
  let nj = min(FD_CH, p.n - j0);
  for (var i = t; i < 1024u; i += 256u) { fq[i] = a[g * 1024u + i]; }
  workgroupBarrier();
  let scale = 1.0 / sqrt(128.0);
  // scores: thread t = position j0 + t, the 8 heads
  var s: array<f32, 8>;
  if (t < nj) {
    let k0 = ((p.lo + j0 + t) % p.ring) * 512u + g * 128u;
    for (var e: u32 = 0u; e < 128u; e += 4u) {
      let k = k4((k0 + e) / 4u);
      for (var h: u32 = 0u; h < 8u; h++) {
        s[h] += dot(vec4<f32>(fq[h * 128u + e], fq[h * 128u + e + 1u], fq[h * 128u + e + 2u], fq[h * 128u + e + 3u]), k);
      }
    }
    for (var h: u32 = 0u; h < 8u; h++) { fp[h * 256u + t] = s[h] * scale; }
  } else {
    for (var h: u32 = 0u; h < 8u; h++) { fp[h * 256u + t] = -3.0e38; }
  }
  workgroupBarrier();
  // max and sum per head: 32 threads per head
  let h = t / 32u; let l = t % 32u;
  var mx = -3.0e38;
  for (var j = l; j < 256u; j += 32u) { mx = max(mx, fp[h * 256u + j]); }
  fr[t] = mx;
  workgroupBarrier();
  for (var m: u32 = 16u; m > 0u; m >>= 1u) {
    if (l < m) { fr[t] = max(fr[t], fr[t + m]); }
    workgroupBarrier();
  }
  let mm = fr[h * 32u];
  workgroupBarrier();
  var sm = 0.0;
  for (var j = l; j < 256u; j += 32u) {
    let v = fp[h * 256u + j];
    let e = select(0.0, exp(v - mm), v > -1.0e38);
    fp[h * 256u + j] = e; sm += e;
  }
  fr[t] = sm;
  workgroupBarrier();
  for (var m: u32 = 16u; m > 0u; m >>= 1u) {
    if (l < m) { fr[t] += fr[t + m]; }
    workgroupBarrier();
  }
  let ll = fr[h * 32u];
  // P V: thread t = head h, dims 4 l .. 4 l + 3
  var acc = vec4<f32>(0.0);
  for (var j: u32 = 0u; j < nj; j++) {
    let v0 = ((p.lo + j0 + j) % p.ring) * 512u + g * 128u + 4u * l;
    acc += fp[h * 256u + j] * v4(v0 / 4u);
  }
  let base = (((8u * g + h) * (p.ctx / FD_CH)) + ch) * 130u;
  sc[base + 2u + 4u * l] = acc.x; sc[base + 3u + 4u * l] = acc.y; sc[base + 4u + 4u * l] = acc.z; sc[base + 5u + 4u * l] = acc.w;
  if (l == 0u) { sc[base] = mm; sc[base + 1u] = ll; }
}

// Pass 2: one workgroup per query head, one thread per dimension. d = sinks, o = output (32 x 128).
@compute @workgroup_size(128)
fn attn_fd_merge(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let h = wid.x; let nch = (p.n + FD_CH - 1u) / FD_CH; let step_i = p.ctx / FD_CH;
  let sink = d[h];
  var mm = sink;
  for (var cc: u32 = 0u; cc < nch; cc++) { mm = max(mm, sc[(h * step_i + cc) * 130u]); }
  var ll = exp(sink - mm);
  var acc = 0.0;
  for (var cc: u32 = 0u; cc < nch; cc++) {
    let base = (h * step_i + cc) * 130u;
    let w = exp(sc[base] - mm);
    ll += w * sc[base + 1u];
    acc += w * sc[base + 2u + t];
  }
  o[h * 128u + t] = acc / ll;
}
