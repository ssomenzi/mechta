//! attn_dmx: decode attention on the matrix units, for a step that checks k tokens (k = 2: the double step, k = 4: the
//! 4-token step of quad.rs). Used with the f16 KV cache.
//!
//! Why: in decode the 8 query heads that share a KV head, times the k tokens of the step, are 8k rows that read the
//! same K and V. With k = 4 that is a 32 x 256 by 256 x positions product: what the matrix units are for. attn_fd2h
//! (k = 2) does it with scalar math, and with k = 4 the math doubles on the same bytes.
//!
//! Layout. Q is packed first (q_dmx) as f16 [kvh][row = token * 8 + head][256], so a tile of 16 rows is one load with
//! stride 256. A workgroup = (KV head, chunk of p.flag positions); R = 8k / 16 row tiles. For every tile of 64 keys:
//!   S = Q K^T: subgroup s (64 lanes) takes keys 16 s..16 s + 15 of the tile, for the R row tiles;
//!   softmax per row with a lazy reference max (the accumulators are matrices and cannot scale one row: when a row's
//!   reference moves, they go through shared memory once), P as f16;
//!   O += P V: subgroup s takes dims 64 s..64 s + 63 (4 tiles of 16) for the R row tiles.
//! So K and V are read once per workgroup. Partials in the format of attn_fd ([max, sum, acc 256] per token, head,
//! chunk), merged by attn_fd_combine with p.flag = chunk size. Token t sees positions < p.n + t.
use crate::engine::{KvFormat, ops_source};

/// The WGSL of attn_dmx{r} (r = 1 or 2 row tiles, i.e. 2 or 4 tokens), with its own workgroup memory.
fn dmx_kernel(r: usize) -> String {
    let m = 16 * r;
    let mut acc = String::new(); let mut zero_c = String::new(); let mut qk = String::new(); let mut store_s = String::new();
    let mut rescale = String::new(); let mut pv = String::new(); let mut out = String::new();
    for rt in 0..r {
        zero_c += &format!("    var c{rt} = coop_mat16x16<f32, C>();\n");
        qk += &format!("      c{rt} = coopMultiplyAdd(coopLoadT<coop_mat16x16<f16, A>>(&qh[qb + {}u + dd], 256u), km, c{rt});\n", rt * 16 * 256);
        store_s += &format!("    coopStoreT(c{rt}, &dms{r}[{}u + 16u * s], DM_SS);\n", rt * 16 * 65);
        pv += &format!("      let pa{rt} = coopLoadT<coop_mat16x16<f16, A>>(&dmp{r}[{}u + kt * 16u], DM_PP);\n", rt * 16 * 72);
        for j in 0..4 {
            acc += &format!("  var o{rt}_{j} = coop_mat16x16<f32, C>();\n");
            rescale += &format!("      coopStoreT(o{rt}_{j}, &dmr{r}[s * 256u], 16u); workgroupBarrier();\n      for (var i = t % 64u; i < 256u; i += 64u) {{ dmr{r}[s * 256u + i] *= dmc{r}[{}u + i / 16u]; }}\n      workgroupBarrier(); o{rt}_{j} = coopLoadT<coop_mat16x16<f32, C>>(&dmr{r}[s * 256u], 16u); workgroupBarrier();\n", rt * 16);
            pv += &format!("      o{rt}_{j} = coopMultiplyAdd(pa{rt}, v{j}, o{rt}_{j});\n");
            out += &format!("  coopStoreT(o{rt}_{j}, &dmr{r}[s * 256u], 16u); workgroupBarrier();\n  for (var i = t % 64u; i < 256u; i += 64u) {{\n    let row = {}u + i / 16u; let tok = row / 8u; let head = 8u * kvh + row % 8u;\n    if (tok < kk) {{ o[((tok * 16u + head) * nb + chunk) * 258u + 2u + 64u * s + {}u + i % 16u] = dmr{r}[s * 256u + i]; }}\n  }}\n  workgroupBarrier();\n", rt * 16, 16 * j);
        }
    }
    format!(r#"
// ===================== attn_dmx{r}: decode attention on the matrix units, {m} rows per KV head =====================
var<workgroup> dms{r}: array<f32, {ms}>;        // S: [row {m}][DM_SS]
var<workgroup> dmp{r}: array<f16, {mp}>;        // P: [row {m}][DM_PP]
var<workgroup> dmr{r}: array<f32, 1024>;        // round trip of one accumulator per subgroup
var<workgroup> dmm{r}: array<f32, {m}>;         // reference max per row
var<workgroup> dml{r}: array<f32, {m}>;         // sum per row
var<workgroup> dmc{r}: array<f32, {m}>;         // rescale of this tile per row
var<workgroup> dmn{r}: array<u32, {m}>;         // the row's reference moved on this tile
var<workgroup> dmq{r}: array<f32, 256>;         // per-thread partial max, then partial sum
var<workgroup> dmf{r}: u32;
@compute @workgroup_size(256)
fn attn_dmx{r}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {{
  let kvh = wid.x; let chunk = wid.y; let nb = p.rows; let ch = p.flag; let kk = p.f1;
  let scale = bitcast<f32>(p.f0);
  let cs = chunk * ch;
  let lim = p.n + kk - 1u;                         // the last token sees positions [0, lim)
  if (cs >= lim) {{ return; }}
  let ntile = min(ch / 64u, (lim - cs + 63u) / 64u);
  if (t < {m}u) {{ dmm{r}[t] = -3.0e38; dml{r}[t] = 0.0; dmn{r}[t] = 0u; }}
  let s = t / 64u;
  let qb = kvh * {m}u * 256u;
  let tpr = 256u / {m}u; let kpt = 64u / tpr;    // softmax: threads per row, keys per thread
  let sr = t / tpr; let sq = t % tpr;
{acc}  for (var tile: u32 = 0u; tile < ntile; tile++) {{
    let k0 = cs + tile * 64u;
    let kb = (k0 + 16u * s) * 512u + kvh * 256u;
{zero_c}    for (var dd: u32 = 0u; dd < 256u; dd += 16u) {{
      let km = coopLoad<coop_mat16x16<f16, B>>(&kh[kb + dd], 512u);
{qk}    }}
    workgroupBarrier();                            // the PV of the previous tile is done with dmp
{store_s}    workgroupBarrier();
    let vis = p.n + sr / 8u;                       // the token of row sr sees positions < vis
    let rb = sr * DM_SS + sq * kpt;
    var mq = -3.0e38;
    for (var c: u32 = 0u; c < kpt; c++) {{
      let v = select(-3.0e38, dms{r}[rb + c] * scale, k0 + sq * kpt + c < vis);
      dms{r}[rb + c] = v; mq = max(mq, v);
    }}
    dmq{r}[t] = mq;
    workgroupBarrier();
    let mo = dmm{r}[sr];
    var mt = -3.0e38;
    for (var i: u32 = 0u; i < tpr; i++) {{ mt = max(mt, dmq{r}[sr * tpr + i]); }}
    var mr = mo;
    if (mo < -1.0e38 || mt > mo + DM_LAZY) {{ mr = max(mo, mt); }}
    var sm = 0.0;
    for (var c: u32 = 0u; c < kpt; c++) {{
      let v = dms{r}[rb + c]; let w = select(0.0, exp(v - mr), v > -1.0e38);
      let wh = f16(w); dmp{r}[sr * DM_PP + sq * kpt + c] = wh; sm += f32(wh);
    }}
    workgroupBarrier();
    dmq{r}[t] = sm;
    workgroupBarrier();
    if (sq == 0u) {{
      var tot = 0.0;
      for (var i: u32 = 0u; i < tpr; i++) {{ tot += dmq{r}[sr * tpr + i]; }}
      let cf = select(1.0, exp(mo - mr), mo > -1.0e38);
      dmc{r}[sr] = cf; dmm{r}[sr] = mr; dml{r}[sr] = dml{r}[sr] * cf + tot;
      dmn{r}[sr] = select(0u, 1u, cf != 1.0);
    }}
    workgroupBarrier();
    if (t == 0u) {{ var f = 0u; for (var i: u32 = 0u; i < {m}u; i++) {{ f |= dmn{r}[i]; }} dmf{r} = f; }}
    if (workgroupUniformLoad(&dmf{r}) != 0u) {{
{rescale}    }}
    for (var kt: u32 = 0u; kt < 4u; kt++) {{
      let vb = (k0 + kt * 16u) * 512u + kvh * 256u + 64u * s;
      let v0 = coopLoadT<coop_mat16x16<f16, B>>(&vh[vb], 512u);
      let v1 = coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 16u], 512u);
      let v2 = coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 32u], 512u);
      let v3 = coopLoadT<coop_mat16x16<f16, B>>(&vh[vb + 48u], 512u);
{pv}    }}
  }}
  workgroupBarrier();
{out}  if (t < {m}u) {{
    let tok = t / 8u;
    if (tok < kk) {{ let base = ((tok * 16u + 8u * kvh + t % 8u) * nb + chunk) * 258u; o[base] = dmm{r}[t]; o[base + 1u] = dml{r}[t]; }}
  }}
}}
"#, ms = m * 65, mp = m * 72)
}

/// ops.wgsl (f16 KV) + q_dmx + attn_dmx1 + attn_dmx2, with the two enables in front.
pub fn dmx_source() -> String {
    format!(r#"enable f16;
enable wgpu_cooperative_matrix;
{ops}
@group(0) @binding(12) var<storage, read> kh: array<f16>;     // the K cache read as f16
@group(0) @binding(13) var<storage, read> qh: array<f16>;     // Q packed by q_dmx
@group(0) @binding(14) var<storage, read> vh: array<f16>;     // the V cache read as f16
const DM_SS: u32 = 65u;                         // row stride of S (64 keys + 1)
const DM_PP: u32 = 72u;                         // row stride of P in f16 (64 keys + 8)
const DM_LAZY: f32 = 8.0;
// q_dmx: Q of the step (a: [token][4096] f32) to f16 [kvh][row = token * 8 + head][256], rows beyond 8 * p.f1 = 0.
// p.rows = rows per KV head (16 or 32). o[i] = pair 2i, 2i + 1, read as f16 by binding 13.
@compute @workgroup_size(256)
fn q_dmx(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = gid.x; let m = p.rows; if (i >= m * 256u) {{ return; }}
  let e = 2u * i; let d = e % 256u; let rr = e / 256u; let kvh = rr / m; let r = rr % m;
  let tok = r / 8u; let head = 8u * kvh + r % 8u;
  var v = vec2<f32>(0.0, 0.0);
  if (tok < p.f1) {{ let q0 = tok * 4096u + head * 256u + d; v = vec2<f32>(a[q0], a[q0 + 1u]); }}
  o[i] = bitcast<f32>(pack2x16float(v));
}}
{k1}
{k2}"#, ops = ops_source(KvFormat::F16), k1 = dmx_kernel(1), k2 = dmx_kernel(2))
}
