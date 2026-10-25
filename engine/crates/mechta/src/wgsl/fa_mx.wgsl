// ===================== attn_fa_mx: attn_fa with Q K^T on the matrix units =====================
// Appended to ops.wgsl (f16 KV only) with `enable f16; enable wgpu_cooperative_matrix;` in front (engine::fa_mx_source).
// Same job and same partials as attn_fa, rows grouped differently: a workgroup = 16 tokens x 4 heads (half hf of the 8
// heads sharing KV head kvh) = 64 rows, row = head * 16 + token. S = Q K^T per tile of 64 keys on the matrix units
// (coopMultiplyAdd 16x16x16, f16 in, f32 sum), reading Q (f16 copy, qh) and K (the f16 cache, kh) straight from memory:
// a 16-row tile = 16 tokens of one head, rows 4096 f16 apart (through shared memory it was slower, bank
// conflicts and 16-bit stores). Subgroup s (4 of 64 lanes) = head s of the half, 4 tiles of 16 keys. Then the online
// softmax and O += P V as attn_fa, each thread = (token t / 16, 16 dims) x 4 heads.
// Grid: x = kvh * 2 + hf, y = chunk (from p.step_o), z = group of 16 tokens (4 per dispatch of 64 tokens).
@group(0) @binding(12) var<storage, read> kh: array<f16>;     // the K cache (f16 format) read as f16
@group(0) @binding(13) var<storage, read> qh: array<f16>;     // Q of the block as f16 (q_f16), [token][4096]
const MX_S: u32 = 65u;                          // row stride of S (64 keys + 1 of padding)
var<workgroup> mxs: array<f32, 4160>;           // S then P: [row 64][MX_S]
var<workgroup> mxv: array<vec2<u32>, 1024>;     // V: [key 16][dim/4 64], packed f16
var<workgroup> mxm: array<f32, 64>;             // running max per row
var<workgroup> mxl: array<f32, 64>;             // running sum per row
var<workgroup> mxc: array<f32, 64>;             // rescale of the accumulators of this tile
var<workgroup> mxq: array<f32, 256>;            // softmax: per-thread partial max, then partial sum
@compute @workgroup_size(256)
fn attn_fa_mx(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let kvh = wid.x / 2u; let hf = wid.x % 2u; let chunk = wid.y + p.step_o; let grp = wid.z;
  let scale = bitcast<f32>(p.f0); let nb = p.rows;
  let tb = p.step_a + 16u * grp;                   // block index of the first token of this group
  let cs = p.extra + chunk * FA_CH;
  let lim = p.n + tb + 15u;                         // positions seen by the last token of the group: [extra, lim)
  if (cs >= lim) { return; }
  let ntile = min(FA_CH / 64u, (lim - cs + 63u) / 64u);
  let phase = p.f1;                                  // measuring only (MECHTA_FA_PHASE): 1 = no QK, 2 = no softmax, 3 = no PV
  if (t < 64u) { mxm[t] = -3.0e38; mxl[t] = 0.0; }
  let s = t / 64u;                                  // subgroup (64 lanes) = head 4 * hf + s
  let head0 = 8u * kvh + 4u * hf;
  let qa = tb * 4096u + (head0 + s) * 256u;
  let tq = t / 16u; let dg = t % 16u;               // PV: token tq, dims dg * 16..
  var acc: array<vec4<f32>, 16>;                    // PV: [head i][4 vec4]
  for (var tile: u32 = 0u; tile < ntile; tile++) {
    let k0 = cs + tile * 64u;
    let kb = k0 * 512u + kvh * 256u;
    var c0 = coop_mat16x16<f32, C>(); var c1 = coop_mat16x16<f32, C>();
    var c2 = coop_mat16x16<f32, C>(); var c3 = coop_mat16x16<f32, C>();
    for (var dd: u32 = 0u; dd < 256u; dd += 16u) {
      if (phase == 1u) { break; }
      let qm = coopLoadT<coop_mat16x16<f16, A>>(&qh[qa + dd], 4096u);
      c0 = coopMultiplyAdd(qm, coopLoad<coop_mat16x16<f16, B>>(&kh[kb + dd], 512u), c0);
      c1 = coopMultiplyAdd(qm, coopLoad<coop_mat16x16<f16, B>>(&kh[kb + 8192u + dd], 512u), c1);
      c2 = coopMultiplyAdd(qm, coopLoad<coop_mat16x16<f16, B>>(&kh[kb + 16384u + dd], 512u), c2);
      c3 = coopMultiplyAdd(qm, coopLoad<coop_mat16x16<f16, B>>(&kh[kb + 24576u + dd], 512u), c3);
    }
    workgroupBarrier();                             // the PV of the previous tile is done with mxs
    let so = s * 16u * MX_S;
    coopStoreT(c0, &mxs[so], MX_S); coopStoreT(c1, &mxs[so + 16u], MX_S);
    coopStoreT(c2, &mxs[so + 32u], MX_S); coopStoreT(c3, &mxs[so + 48u], MX_S);
    workgroupBarrier();
    // online softmax, 4 threads per row (row t / 4, keys (t % 4) * 16..+16): scale and mask, row max through
    // shared memory, exp and partial sums, then one thread per row updates max, sum and rescale. (With one
    // thread per row it cost 1.25 ms/token at 100K against 0.4 of attn_fa.)
    if (phase != 2u) {
      let sr = t / 4u; let sq = t % 4u;
      let vis = p.n + tb + sr % 16u;                // the token of row sr sees positions < vis
      let rb = sr * MX_S + sq * 16u;
      var mq = -3.0e38;
      for (var c: u32 = 0u; c < 16u; c++) {
        let v = select(-3.0e38, mxs[rb + c] * scale, k0 + sq * 16u + c < vis);
        mxs[rb + c] = v; mq = max(mq, v);
      }
      mxq[t] = mq;
      workgroupBarrier();
      let mo = mxm[sr];
      let mx = max(mo, max(max(mxq[sr * 4u], mxq[sr * 4u + 1u]), max(mxq[sr * 4u + 2u], mxq[sr * 4u + 3u])));
      var sm = 0.0;
      for (var c: u32 = 0u; c < 16u; c++) { let v = mxs[rb + c]; let w = select(0.0, exp(v - mx), v > -1.0e38); mxs[rb + c] = w; sm += w; }
      workgroupBarrier();                           // every thread has read mxq (the maxima) and mxm[sr]
      mxq[t] = sm;
      workgroupBarrier();
      if (sq == 0u) {
        let cf = select(0.0, exp(mo - mx), mo > -1.0e38);
        mxc[sr] = cf; mxm[sr] = mx; mxl[sr] = mxl[sr] * cf + (mxq[t] + mxq[t + 1u] + mxq[t + 2u] + mxq[t + 3u]);
      }
    }
    workgroupBarrier();
    for (var i: u32 = 0u; i < 4u; i++) {
      let cf = mxc[i * 16u + tq];
      acc[4u * i] *= cf; acc[4u * i + 1u] *= cf; acc[4u * i + 2u] *= cf; acc[4u * i + 3u] *= cf;
    }
    // O += P V, V in pieces of 16 keys
    for (var vp: u32 = 0u; vp < 4u; vp++) {
      if (phase == 3u) { break; }
      workgroupBarrier();
      for (var i = t; i < 1024u; i += 256u) {
        let kr = i / 64u; let d4 = i % 64u;
        mxv[i] = pk4(vv(((k0 + vp * 16u + kr) * 512u + kvh * 256u) / 4u + d4));
      }
      workgroupBarrier();
      for (var kr: u32 = 0u; kr < 16u; kr++) {
        let col = vp * 16u + kr;
        let v0 = up4(mxv[kr * 64u + dg * 4u]); let v1 = up4(mxv[kr * 64u + dg * 4u + 1u]);
        let v2 = up4(mxv[kr * 64u + dg * 4u + 2u]); let v3 = up4(mxv[kr * 64u + dg * 4u + 3u]);
        for (var i: u32 = 0u; i < 4u; i++) {
          let pr = mxs[(i * 16u + tq) * MX_S + col];
          acc[4u * i] += pr * v0; acc[4u * i + 1u] += pr * v1; acc[4u * i + 2u] += pr * v2; acc[4u * i + 3u] += pr * v3;
        }
      }
    }
  }
  workgroupBarrier();
  // partials: token of the dispatch = 16 * grp + tq, head head0 + i (same format as attn_fa, merged by attn_fd_combine)
  for (var i: u32 = 0u; i < 4u; i++) {
    let r = i * 16u + tq;
    let base = (((16u * grp + tq) * 16u + head0 + i) * nb + chunk) * 258u + 2u + dg * 16u;
    for (var q: u32 = 0u; q < 4u; q++) {
      let v = acc[4u * i + q];
      o[base + 4u * q] = v.x; o[base + 4u * q + 1u] = v.y; o[base + 4u * q + 2u] = v.z; o[base + 4u * q + 3u] = v.w;
    }
    if (dg == 0u) { o[base - 2u] = mxm[r]; o[base - 1u] = mxl[r]; }
  }
}

// ===================== attn_fa_mx2: attn_fa_mx with O += P V on the matrix units too =====================
// Same QK and grid as attn_fa_mx. P goes to shared memory as f16 (rows MX_P wide: 32 f16 rows had bank conflicts)
// and V is read straight from the f16 cache (vh); each subgroup keeps its 16 rows x 256 dims of O in 16
// accumulators. The matrix units cannot scale rows, so the softmax uses a lazy reference max: a row's reference moves
// only on its first valid tile or when the row max grows by more than MX_LAZY (P stays <= e^MX_LAZY, fine in f16);
// when one row of the workgroup moves, the accumulators are scaled through shared memory (mxr), which is rare.
// The partials (m = reference max, l, O) have the same format as attn_fa: attn_fd_combine is unchanged.
@group(0) @binding(14) var<storage, read> vh: array<f16>;     // the V cache (f16 format) read as f16
const MX_P: u32 = 72u;                          // row stride of P in f16 (64 keys + 8 of padding)
const MX_LAZY: f32 = 8.0;
var<workgroup> mxp: array<f16, 4608>;           // P: [row 64][MX_P]
var<workgroup> mxr: array<f32, 1024>;           // rescale round trip: [subgroup 4][16 x 16]
var<workgroup> mxn: array<u32, 64>;             // per row: the reference moved on this tile
var<workgroup> mxf: u32;                        // some row of the workgroup moved
@compute @workgroup_size(256)
fn attn_fa_mx2(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
  let kvh = wid.x / 2u; let hf = wid.x % 2u; let chunk = wid.y + p.step_o; let grp = wid.z;
  let scale = bitcast<f32>(p.f0); let nb = p.rows;
  let tb = p.step_a + 16u * grp;
  let cs = p.extra + chunk * FA_CH;
  let lim = p.n + tb + 15u;
  if (cs >= lim) { return; }
  let ntile = min(FA_CH / 64u, (lim - cs + 63u) / 64u);
  let phase = p.f1;                                  // measuring only (MECHTA_FA_PHASE): 1 = no QK, 2 = no softmax, 3 = no PV
  if (t < 64u) { mxm[t] = -3.0e38; mxl[t] = 0.0; mxn[t] = 0u; }
  let s = t / 64u;                                  // subgroup (64 lanes) = head 4 * hf + s, rows s * 16..
  let head0 = 8u * kvh + 4u * hf;
  let qa = tb * 4096u + (head0 + s) * 256u;
  var o0 = coop_mat16x16<f32, C>();
  var o1 = coop_mat16x16<f32, C>();
  var o2 = coop_mat16x16<f32, C>();
  var o3 = coop_mat16x16<f32, C>();
  var o4 = coop_mat16x16<f32, C>();
  var o5 = coop_mat16x16<f32, C>();
  var o6 = coop_mat16x16<f32, C>();
  var o7 = coop_mat16x16<f32, C>();
  var o8 = coop_mat16x16<f32, C>();
  var o9 = coop_mat16x16<f32, C>();
  var o10 = coop_mat16x16<f32, C>();
  var o11 = coop_mat16x16<f32, C>();
  var o12 = coop_mat16x16<f32, C>();
  var o13 = coop_mat16x16<f32, C>();
  var o14 = coop_mat16x16<f32, C>();
  var o15 = coop_mat16x16<f32, C>();
  for (var tile: u32 = 0u; tile < ntile; tile++) {
    let k0 = cs + tile * 64u;
    let kb = k0 * 512u + kvh * 256u;
    var c0 = coop_mat16x16<f32, C>(); var c1 = coop_mat16x16<f32, C>();
    var c2 = coop_mat16x16<f32, C>(); var c3 = coop_mat16x16<f32, C>();
    for (var dd: u32 = 0u; dd < 256u; dd += 16u) {
      if (phase == 1u) { break; }
      let qm = coopLoadT<coop_mat16x16<f16, A>>(&qh[qa + dd], 4096u);
      c0 = coopMultiplyAdd(qm, coopLoad<coop_mat16x16<f16, B>>(&kh[kb + dd], 512u), c0);
      c1 = coopMultiplyAdd(qm, coopLoad<coop_mat16x16<f16, B>>(&kh[kb + 8192u + dd], 512u), c1);
      c2 = coopMultiplyAdd(qm, coopLoad<coop_mat16x16<f16, B>>(&kh[kb + 16384u + dd], 512u), c2);
      c3 = coopMultiplyAdd(qm, coopLoad<coop_mat16x16<f16, B>>(&kh[kb + 24576u + dd], 512u), c3);
    }
    workgroupBarrier();                             // the PV of the previous tile is done with mxp
    let so = s * 16u * MX_S;
    coopStoreT(c0, &mxs[so], MX_S); coopStoreT(c1, &mxs[so + 16u], MX_S);
    coopStoreT(c2, &mxs[so + 32u], MX_S); coopStoreT(c3, &mxs[so + 48u], MX_S);
    workgroupBarrier();
    // softmax, 4 threads per row (row t / 4, keys (t % 4) * 16..+16), with the lazy reference max
    if (phase != 2u) {
      let sr = t / 4u; let sq = t % 4u;
      let vis = p.n + tb + sr % 16u;                // the token of row sr sees positions < vis
      let rb = sr * MX_S + sq * 16u;
      var mq = -3.0e38;
      for (var c: u32 = 0u; c < 16u; c++) {
        let v = select(-3.0e38, mxs[rb + c] * scale, k0 + sq * 16u + c < vis);
        mxs[rb + c] = v; mq = max(mq, v);
      }
      mxq[t] = mq;
      workgroupBarrier();
      let mo = mxm[sr];
      let mt = max(max(mxq[sr * 4u], mxq[sr * 4u + 1u]), max(mxq[sr * 4u + 2u], mxq[sr * 4u + 3u]));
      var mr = mo;
      if (mo < -1.0e38 || mt > mo + MX_LAZY) { mr = max(mo, mt); }
      var sm = 0.0;
      let pb = sr * MX_P + sq * 16u;
      for (var c: u32 = 0u; c < 16u; c++) {
        let v = mxs[rb + c]; let w = select(0.0, exp(v - mr), v > -1.0e38);
        let wh = f16(w); mxp[pb + c] = wh; sm += f32(wh);   // the sum of the same rounded P that multiplies V
      }
      workgroupBarrier();                           // every thread has read mxq (the maxima) and mxm[sr]
      mxq[t] = sm;
      workgroupBarrier();
      if (sq == 0u) {
        let cf = select(1.0, exp(mo - mr), mo > -1.0e38);   // no valid reference yet: O is still zero
        mxc[sr] = cf; mxm[sr] = mr; mxl[sr] = mxl[sr] * cf + (mxq[t] + mxq[t + 1u] + mxq[t + 2u] + mxq[t + 3u]);
        mxn[sr] = select(0u, 1u, cf != 1.0);
      }
    }
    workgroupBarrier();
    if (t == 0u) { var f = 0u; for (var r: u32 = 0u; r < 64u; r++) { f |= mxn[r]; } mxf = f; }
    if (workgroupUniformLoad(&mxf) != 0u) {
      // rare: some row moved its reference; scale the accumulators row by row through shared memory
      coopStoreT(o0, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o0 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o1, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o1 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o2, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o2 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o3, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o3 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o4, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o4 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o5, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o5 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o6, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o6 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o7, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o7 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o8, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o8 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o9, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o9 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o10, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o10 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o11, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o11 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o12, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o12 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o13, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o13 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o14, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o14 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
      coopStoreT(o15, &mxr[s * 256u], 16u); workgroupBarrier();
      for (var i = t % 64u; i < 256u; i += 64u) { mxr[s * 256u + i] *= mxc[s * 16u + i / 16u]; }
      workgroupBarrier(); o15 = coopLoadT<coop_mat16x16<f32, C>>(&mxr[s * 256u], 16u); workgroupBarrier();
    }
    // O += P V: P tile (16 rows x 16 keys) from shared memory, V tile (16 keys x 16 dims) straight from the cache
    for (var kt: u32 = 0u; kt < 4u; kt++) {
      if (phase == 3u) { break; }
      let pa = coopLoadT<coop_mat16x16<f16, A>>(&mxp[s * 16u * MX_P + kt * 16u], MX_P);
      let vb = (k0 + kt * 16u) * 512u + kvh * 256u;
        o0 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 0u], 512u), o0);
        o1 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 16u], 512u), o1);
        o2 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 32u], 512u), o2);
        o3 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 48u], 512u), o3);
        o4 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 64u], 512u), o4);
        o5 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 80u], 512u), o5);
        o6 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 96u], 512u), o6);
        o7 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 112u], 512u), o7);
        o8 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 128u], 512u), o8);
        o9 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 144u], 512u), o9);
        o10 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 160u], 512u), o10);
        o11 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 176u], 512u), o11);
        o12 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 192u], 512u), o12);
        o13 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 208u], 512u), o13);
        o14 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 224u], 512u), o14);
        o15 = coopMultiplyAdd(pa, coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 240u], 512u), o15);
    }
  }
  // partials: rows of subgroup s = 16 tokens of head head0 + s; one token to the next is 16 * nb * 258 apart.
  // The index goes in a `let`: written inline next to the runtime stride, naga 30.0.1 panics in the SPIR-V writer
  // ("Expression [..] is not cached!").
  let ts = 16u * nb * 258u;
  let ob = ((16u * grp * 16u + head0 + s) * nb + chunk) * 258u + 2u;
  let oi0 = ob + 0u; coopStoreT(o0, &o[oi0], ts);
  let oi1 = ob + 16u; coopStoreT(o1, &o[oi1], ts);
  let oi2 = ob + 32u; coopStoreT(o2, &o[oi2], ts);
  let oi3 = ob + 48u; coopStoreT(o3, &o[oi3], ts);
  let oi4 = ob + 64u; coopStoreT(o4, &o[oi4], ts);
  let oi5 = ob + 80u; coopStoreT(o5, &o[oi5], ts);
  let oi6 = ob + 96u; coopStoreT(o6, &o[oi6], ts);
  let oi7 = ob + 112u; coopStoreT(o7, &o[oi7], ts);
  let oi8 = ob + 128u; coopStoreT(o8, &o[oi8], ts);
  let oi9 = ob + 144u; coopStoreT(o9, &o[oi9], ts);
  let oi10 = ob + 160u; coopStoreT(o10, &o[oi10], ts);
  let oi11 = ob + 176u; coopStoreT(o11, &o[oi11], ts);
  let oi12 = ob + 192u; coopStoreT(o12, &o[oi12], ts);
  let oi13 = ob + 208u; coopStoreT(o13, &o[oi13], ts);
  let oi14 = ob + 224u; coopStoreT(o14, &o[oi14], ts);
  let oi15 = ob + 240u; coopStoreT(o15, &o[oi15], ts);
  if (t < 64u) {
    let mb = (((16u * grp + t % 16u) * 16u + head0 + t / 16u) * nb + chunk) * 258u;
    o[mb] = mxm[t]; o[mb + 1u] = mxl[t];
  }
}

// q_f16: Q of the block (a, f32) to f16 for attn_fa_mx: o[i] holds elements 2i and 2i+1 packed, read as f16 by
// binding 13 (qh). p.n = number of pairs.
@compute @workgroup_size(256)
fn q_f16(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x; if (i >= p.n) { return; }
  o[i] = bitcast<f32>(pack2x16float(vec2<f32>(a[2u * i], a[2u * i + 1u])));
}
