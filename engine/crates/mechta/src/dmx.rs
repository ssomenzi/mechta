//! attn_dmx: decode attention on the matrix units, for a step that checks k tokens (k = 2 or 4). Kernel benchmark:
//! `mechta dmx-bench`.
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

/// `mechta dmx-bench <positions,...> <chunk,...>`: random f16 K/V (deterministic), random Q; at each context n:
/// attn_fd2h + merge (k = 2, today) against q_dmx + attn_dmx + merge for k = 2 and 4 and each chunk size. Checks the
/// outputs against a CPU reference (f64) for every token and head, then times 12 runs (2 dropped), median per dispatch.
pub fn dmx_bench(positions: &str, chunks: &str) {
    use mechta_gpu::Gpu;
    let gpu = Gpu::new();
    let ns: Vec<usize> = positions.split(',').map(|s| s.trim().parse().unwrap()).collect();
    let chs: Vec<usize> = chunks.split(',').map(|s| s.trim().parse().unwrap()).collect();
    let ctx = 131072usize;
    assert!(ns.iter().all(|&n| n + 64 < ctx));
    // deterministic values: K, V ~ uniform(-1, 1) as f16, Q ~ uniform(-1, 1) f32 (scale 1/16 as the model)
    let mut st = 0x9E3779B97F4A7C15u64;
    let mut rnd = || { st ^= st << 13; st ^= st >> 7; st ^= st << 17; (st >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0 };
    let h16 = |x: f64| half::f16::from_f64(x);
    let kv_words = ctx * 256;
    let k16: Vec<half::f16> = (0..ctx * 512).map(|_| h16(rnd())).collect();
    let v16: Vec<half::f16> = (0..ctx * 512).map(|_| h16(rnd())).collect();
    let q: Vec<f32> = (0..4 * 4096).map(|_| rnd() as f32).collect();
    let bytes16 = |v: &[half::f16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_bits().to_le_bytes()).collect() };
    let kc = gpu.buffer_byte("k", &bytes16(&k16));
    let vc = gpu.buffer_byte("v", &bytes16(&v16));
    assert_eq!(k16.len() / 2, kv_words);
    let qr = gpu.buffer_f32("qr", &q);
    let qp = gpu.empty_buffer("qp", (2 * 32 * 256 * 2) as u64);
    let nb_max = ctx / chs.iter().min().unwrap().min(&256);
    let att_p = gpu.empty_buffer("att_p", (4 * 16 * nb_max * 258 * 4) as u64);
    let att = gpu.empty_buffer("att", (4 * 4096 * 4) as u64);
    let src = dmx_source();
    let pipe = |e: &str| gpu.pipeline(&src, e);
    let (p_fd2h, p_comb, p_q, p_d1, p_d2) = (pipe("attn_fd2h"), pipe("attn_fd_combine"), pipe("q_dmx"), pipe("attn_dmx1"), pipe("attn_dmx2"));
    let fb = |x: f32| x.to_bits();
    // CPU reference: token t, head h, positions < n + t
    let reference = |n: usize, k: usize| -> Vec<f64> {
        let mut out = vec![0f64; k * 4096];
        for tok in 0..k { for h in 0..16 {
            let kvh = h / 8; let vis = n + tok;
            let qv = &q[tok * 4096 + h * 256..][..256];
            let s: Vec<f64> = (0..vis).map(|i| (0..256).map(|d| qv[d] as f64 * k16[i * 512 + kvh * 256 + d].to_f64()).sum::<f64>() / 16.0).collect();
            let mx = s.iter().cloned().fold(f64::MIN, f64::max);
            let w: Vec<f64> = s.iter().map(|x| (x - mx).exp()).collect(); let z: f64 = w.iter().sum();
            for d in 0..256 { out[tok * 4096 + h * 256 + d] = (0..vis).map(|i| w[i] * v16[i * 512 + kvh * 256 + d].to_f64()).sum::<f64>() / z; }
        } }
        out
    };
    let err = |got: &[f32], want: &[f64]| -> f64 { got.iter().zip(want).map(|(a, b)| (*a as f64 - b).abs()).fold(0.0, f64::max) / want.iter().map(|x| x.abs()).fold(0.0, f64::max) };
    let time = |cmds: &[mechta_gpu::Command]| -> Vec<f64> {
        let mut per: Vec<Vec<f64>> = vec![vec![]; cmds.len()];
        for it in 0..12 { let m = gpu.run_commands_measured(cmds); if it >= 2 { for (i, t) in m.gpu_ms.iter().enumerate() { per[i].push(*t); } } }
        per.into_iter().map(|mut v| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] }).collect()
    };
    println!("dmx-bench · ctx {ctx} · K/V {:.0} MB each · times = median of 10 (one pass per dispatch), ms", (ctx * 1024) as f64 / 1e6);
    for &n in &ns {
        let kv_mb = (n * 2 * 2 * 256 * 2) as f64 / 1e6;   // K + V of the 2 KV heads read, f16
        // today: attn_fd2h + merge, k = 2 (blocks of 256, merge flag 256)
        let u = gpu.uniform(&[n as u32, (ctx / 128) as u32, fb(1.0 / 16.0), 0, 0, 0, 256, 0]);
        let c_fd = gpu.command(&p_fd2h, &[(0, &qr), (5, &att_p), (7, &u), (9, &kc), (10, &vc)], (2, (ctx / 256) as u32, 1), "attn_fd2h".into(), 0);
        let c_cm = gpu.command(&p_comb, &[(0, &att_p), (5, &att), (7, &u)], (16, 2, 1), "merge".into(), 0);
        let want2 = reference(n, 2);
        let tm = time(&[c_fd, c_cm]);
        let got: Vec<f32> = gpu.read_f32(&att, 2 * 4096);
        println!("n={n} k=2 attn_fd2h: {:.2} + merge {:.2} = {:.2} ms · {:.0} GB/s · error vs CPU {:.1e}", tm[0], tm[1], tm[0] + tm[1], kv_mb / tm[0], err(&got, &want2));
        let want4 = reference(n, 4);
        for &k in &[2usize, 4] {
            for &ch in &chs {
                let (m, pd) = if k == 2 { (16u32, &p_d1) } else { (32, &p_d2) };
                let nb = (ctx / ch) as u32;
                let uq = gpu.uniform(&[n as u32, m, 0, k as u32, 0, 0, 0, 0]);
                let ud = gpu.uniform(&[n as u32, nb, fb(1.0 / 16.0), k as u32, 0, 0, ch as u32, 0]);
                let cq = gpu.command(&p_q, &[(0, &qr), (5, &qp), (7, &uq)], (m, 1, 1), "q_dmx".into(), 0);
                let cd = gpu.command(pd, &[(5, &att_p), (7, &ud), (12, &kc), (13, &qp), (14, &vc)], (2, nb, 1), "attn_dmx".into(), 0);
                let cm = gpu.command(&p_comb, &[(0, &att_p), (5, &att), (7, &ud)], (16, k as u32, 1), "merge".into(), 0);
                let tm = time(&[cq, cd, cm]);
                let got: Vec<f32> = gpu.read_f32(&att, k * 4096);
                let want = if k == 2 { &want2[..] } else { &want4[..] };
                println!("n={n} k={k} chunk {ch}: q {:.2} + attn_dmx {:.2} + merge {:.2} = {:.2} ms · {:.0} GB/s · error vs CPU {:.1e}",
                    tm[0], tm[1], tm[2], tm[0] + tm[1] + tm[2], kv_mb / tm[1], err(&got, want));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn dmx_source_validates() {
        let src = super::dmx_source();
        let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module).unwrap_or_else(|e| panic!("{}", e.emit_to_string(&src)));
    }
}

/// `mechta dmx-ab <text> <checkpoints> <steps> <rounds>` (measurement, real model and real KV): prefill of the text up to
/// each checkpoint; then (1) 8 double steps where, from the same state (photo), attn_fd2h and attn_dmx give their logits
/// (KL and top-1 of both tokens); (2) speed: `rounds` x (block of `steps` double steps with attn_fd2h, block with
/// attn_dmx), order swapped every round, the true tokens of the text as drafts (always accepted). Wall time of each
/// double step (CPU preparation, submit, GPU, read back), the first 2 of each block left out.
pub fn dmx_ab(text: &str, checkpoints: &str, steps: usize, rounds: usize) {
    use crate::engine::{Engine, VOCAB};
    use std::time::Instant;
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(text).expect("text"), false).expect("encode").get_ids().to_vec();
    let soft = |l: &[f32]| -> Vec<f64> { let mx = l.iter().cloned().fold(f32::MIN, f32::max) as f64;
        let e: Vec<f64> = l.iter().map(|x| (*x as f64 - mx).exp()).collect(); let z: f64 = e.iter().sum(); e.iter().map(|x| x / z).collect() };
    let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    m.reset();
    let mut pos = 0;
    for p in checkpoints.split(',').map(|s| s.trim().parse::<usize>().expect("checkpoint")) {
        assert!(p >= pos && p + 16 + 2 * steps * 2 * rounds < ids.len(), "text too short");
        let t = Instant::now();
        if p > pos { m.prefill(&ids[pos..p], pos); }
        pos = p;
        println!("prefill to {p} in {:.0} s", t.elapsed().as_secs_f64());
        let (mut kls, mut same) = (vec![], 0);
        for _ in 0..8 {
            m.photograph();
            m.set_dmx(false); m.step2(ids[pos], ids[pos + 1], pos); let a = m.read("logits", 2 * VOCAB);
            m.restore_photo();
            m.set_dmx(true); m.step2(ids[pos], ids[pos + 1], pos); let b = m.read("logits", 2 * VOCAB);
            for tok in 0..2 {
                let (la, lb) = (&a[tok * VOCAB..(tok + 1) * VOCAB], &b[tok * VOCAB..(tok + 1) * VOCAB]);
                let (pa, pb) = (soft(la), soft(lb));
                kls.push(pa.iter().zip(&pb).filter(|(x, _)| **x > 0.0).map(|(x, y)| x * (x.ln() - y.max(1e-300).ln())).sum::<f64>());
                same += (argmax(la) == argmax(lb)) as usize;
            }
            pos += 2;
        }
        kls.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("  checkpoint {p}: attn_dmx against attn_fd2h on the same state, 16 tokens: KL median {:.2e} max {:.2e} · same top-1 {same}/16", kls[8], kls[15]);
        let mut per = [vec![], vec![]];
        let mut rounds_med = [vec![], vec![]];
        for r in 0..rounds {
            let order = if r % 2 == 0 { [false, true] } else { [true, false] };
            for v in order {
                m.set_dmx(v);
                let mut blk = vec![];
                for i in 0..steps {
                    let t = Instant::now();
                    m.step2(ids[pos], ids[pos + 1], pos);
                    if i >= 2 { blk.push(t.elapsed().as_secs_f64() * 1e3); }
                    pos += 2;
                }
                blk.sort_by(|a, b| a.partial_cmp(b).unwrap());
                rounds_med[v as usize].push(blk[blk.len() / 2]);
                per[v as usize].extend(blk);
            }
        }
        let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
        let (a, b) = (med(&mut per[0]), med(&mut per[1]));
        let range = |v: &Vec<f64>| { let mn = v.iter().cloned().fold(f64::MAX, f64::min); let mx = v.iter().cloned().fold(f64::MIN, f64::max); format!("{mn:.1}-{mx:.1}") };
        println!("  checkpoint {p} (positions {p}..{pos}): double step attn_fd2h {a:.2} ms (round medians {}) · attn_dmx {b:.2} ms (round medians {}) · {:+.1}%",
            range(&rounds_med[0]), range(&rounds_med[1]), 100.0 * (b - a) / a);
    }
}
