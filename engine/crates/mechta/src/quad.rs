//! The k-token verify step (K = 4: the sure token + 3 MTP drafts), designed from what one step must read and write.
//!
//! One pass over the weights for the 4 rows: dense matvecs with 4 columns (matvec_nh), the experts of the union of the
//! 4 rows read once (merge4 + matvec_u4), attention on the matrix units for 32 rows per KV head (attn_dmx2).
//! DeltaNet: the state is read and written once per step. `conv_q` / `delta_q` first apply the rows the PREVIOUS step
//! had accepted (inputs saved by that step), write the committed state, then run the 4 new rows in registers for their
//! outputs only and save their inputs. So a refused draft costs nothing to undo; before any other kind of step the
//! pending rows go in with `flush` (commit only).
//! Drafts: the MTP layer on the 1 + accepted rows (their K/V into its cache), head on the last row; then 2 more MTP
//! drafts in a chain.
//! Child module of `engine`: it uses the engine's private parts.
use super::*;
use std::time::Instant;

/// Rows of the step.
pub const K: usize = 4;
/// Values saved per row by `delta_q` (DQ_ROW of ops.wgsl): k [2048], v [4096], beta [32], gate [32].
const DQ_ROW: usize = 6208;

/// Programs and buffers of the k-token step.
pub(super) struct Quad {
    /// the step, per conv parity (the parity also selects which save buffer is read and which is written)
    prog: [Vec<Command>; 2],
    /// commit of the pending rows only, per conv parity
    flush: [Vec<Command>; 2],
    /// MTP draft on c = 1..=K rows (index c - 1), head on the last row
    mtp: Vec<Vec<Command>>,
    /// RoPE tables of the K rows of the step; position uniforms of the attention layers and of the MTP layer (index N_LAYER)
    tab: wgpu::Buffer, u_pos: Vec<Option<[wgpu::Buffer; 3]>>,
    /// RoPE table of the MTP rows (`mtp`)
    tab_m: wgpu::Buffer,
    /// embedding rows of the step (K rows)
    emb4: wgpu::Buffer,
    /// attn_dmx2 uniform per attention layer
    u_dmx: Vec<Option<wgpu::Buffer>>,
    /// conv_q / delta_q uniforms: step (K new rows) and flush (0 new rows); both carry the pending count
    u_dq: wgpu::Buffer, u_dq0: wgpu::Buffer,
    /// rows of the previous step still to commit (0 = state committed)
    pub pend: std::cell::Cell<usize>,
}

/// Fixed uniforms of a block of k rows.
struct Uk { norm: wgpu::Buffer, normh: wgpu::Buffer, qn: wgpu::Buffer, kn: wgpu::Buffer, rq: wgpu::Buffer, bg: wgpu::Buffer,
            l2: wgpu::Buffer, gnh: wgpu::Buffer, n_x: wgpu::Buffer, n_ag: wgpu::Buffer, n_esw: wgpu::Buffer, n_ssw: wgpu::Buffer,
            moe: wgpu::Buffer, moe_last: wgpu::Buffer, route: wgpu::Buffer }
fn uk(of: &Factory, k: usize) -> Uk {
    let k = k as u32;
    Uk {
        norm: of.u([2048, k, fb(EPS), fb(1.0), 2048, 2048, 1, 0]), normh: of.u([2048, k, fb(EPS), fb(1.0), 2048, 2048, 1 | 4, 0]),
        qn: of.u([256, 16 * k, fb(EPS), fb(1.0), 512, 256, 1, 0]), kn: of.u([256, 2 * k, fb(EPS), fb(1.0), 256, 256, 1, 0]),
        rq: of.u([256, 16 * k, 0, 0, 256, 256, 16, 0]),
        bg: of.u([32, k, 0, 0, 0, 0, 0, 0]),
        l2: of.u([128, 32 * k, fb(EPS / 128.0), fb(1.0 / 128f32.sqrt()), 128, 128, 0, 0]),
        gnh: of.u([128, 32 * k, fb(EPS), fb(1.0), 128, 128, 3 | 4, 0]),
        n_x: of.u([2048 * k, 0, 0, 0, 0, 0, 0, 0]), n_ag: of.u([4096 * k, 0, 0, 0, 0, 0, 0, 0]),
        n_esw: of.u([4096 * k, 0, 0, 0, 0, 0, 0, 0]), n_ssw: of.u([512 * k, 0, 0, 0, 0, 0, 0, 0]),
        moe: of.u([2048, 8, 0, 0, 0, 0, 0, k]),
        moe_last: of.u([2048, 8, 0, 0, 0, 0, 2048 * (k - 1), 0]),   // one token, flag = offset of the last row
        route: of.u([k, 0, 0, 0, 0, 0, 0, 0]),
    }
}

impl Factory<'_> {
    /// y = W X with X of n columns (n = 1..4, token t at t * k), dense weights read once. f16: x in f16 (matvec_nh).
    fn mvk(&self, name: &str, x: &str, y: &str, n: usize, f16: bool) -> Command {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}"));
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        assert!(!name.contains("_exps") && (1..=4).contains(&n));
        let e = match t.ttype { TType::Q8_0 => "mv_q8_0", TType::Q4_K => "mv_q4_k", TType::Q6_K => "mv_q6_k", TType::F32 | TType::BF16 => "mv_f32", x => panic!("{x:?}") };
        let (tpr, nx, ny) = mv_piece_geometry(rows, k, 32);
        let u = self.u([rows as u32, k as u32, nx, 0, 0, tpr, n as u32, k as u32]);
        let nb = if t.ttype == TType::BF16 { 2 * t.nbytes } else { t.nbytes };
        let p = if f16 { &self.mvh[e] } else { &self.mvn[e] };
        self.gpu.command(p, &[(0, &self.w[name]), (1, self.a(x)), (2, self.a(y)), (3, &self.local_ids), (4, &u)], (nx, ny, 1), format!("{} x{n}{}", short(name), if f16 { " h" } else { "" }), nb)
    }
    /// Experts of the union of K rows (list from merge4 in "u4"), x in f16: row t at t * x_tok + slot * x_step, y at t * y_tok.
    fn mv_u4(&self, name: &str, x: &str, y: &str, x_step: u32, x_tok: u32) -> Command {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}"));
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        let e = match t.ttype { TType::Q4_K => "mv_q4_k", TType::Q6_K => "mv_q6_k", TType::Q3_K => "mv_q3_k", TType::Q8_0 => "mv_q8_0", x => panic!("{name}: {x:?}") };
        let (tpr, nx, ny) = mv_piece_geometry(rows, k, 32);
        let u = self.u([rows as u32, k as u32, nx, 8 * rows as u32, x_step, tpr, K as u32, x_tok]);
        self.gpu.command(&self.mvu4[e], &[(0, &self.w[name]), (1, self.a(x)), (2, self.a(y)), (3, self.a("u4")), (4, &u)],
                         (nx, ny, (8 * K) as u32), format!("experts {} ({e}) union x{K}", short(name)), 20 * t.nbytes / 256)
    }
}

/// What the k-row block is for: the main model (x in f16, DeltaNet with fused commit, attention on the matrix units,
/// MoE for every row) or the MTP layer (x in f32, attention with attn_fd, MoE for the last row only).
enum Mode<'a> {
    Main { save_c: &'a [Option<[wgpu::Buffer; 2]>], save_d: &'a [Option<[wgpu::Buffer; 2]>], u_dq: &'a wgpu::Buffer, u_dmx: &'a [Option<wgpu::Buffer>] },
    Mtp,
}

/// A layer for k rows: reads "x" = [row 0 | .. | row k-1] (f32, 2048 each), leaves the new residual in "x".
#[allow(clippy::too_many_arguments)]
fn block_k(of: &Factory, il: usize, c: &mut Vec<Command>, u: &Uk, k: usize, par: usize, conv: &[[wgpu::Buffer; 2]], ssm: &[wgpu::Buffer],
           kc: &[Option<wgpu::Buffer>], vc: &[Option<wgpu::Buffer>], u_pos: &[Option<[wgpu::Buffer; 3]>], tab: &wgpu::Buffer, mode: &Mode) {
    let a = |n: &str| of.a(n); let w = |n: String| &of.w[&n];
    let l = |s: &str| format!("blk.{il}.{s}");
    let main = matches!(mode, Mode::Main { .. });
    let k32 = k as u32;
    let mv = |n: &str, x: &str, y: &str| of.mvk(n, x, y, k, main);
    let norm = if main { &u.normh } else { &u.norm };
    c.push(of.op("rmsnorm", &[(0, a("x")), (1, w(l("attn_norm.weight")))], a("h"), None, norm, (k32, 1, 1)));
    let a0 = c.len();
    if attention(il) || il == N_LAYER {
        let (kc, vc) = (kc[il].as_ref().unwrap(), vc[il].as_ref().unwrap());
        let up = u_pos[il].as_ref().unwrap();
        c.push(mv(&l("attn_q.weight"), "h", "qf"));
        c.push(mv(&l("attn_k.weight"), "h", "kb"));
        c.push(mv(&l("attn_v.weight"), "h", "vb"));
        c.push(of.op("rmsnorm", &[(0, a("qf")), (1, w(l("attn_q_norm.weight")))], a("qn"), None, &u.qn, (16 * k32, 1, 1)));
        c.push(of.op("rmsnorm", &[(0, a("kb")), (1, w(l("attn_k_norm.weight")))], a("kn"), None, &u.kn, (2 * k32, 1, 1)));
        c.push(of.op("rope2", &[(0, a("qn")), (1, tab)], a("qr"), None, &u.rq, (16 * k32, 1, 1)));
        c.push(of.op(kv_writer("rope2"), &[(0, a("kn")), (1, tab), (11, kc)], kc, None, &up[0], (2 * k32, 1, 1)));
        c.push(of.op(kv_writer("copy"), &[(0, a("vb")), (11, vc)], vc, None, &up[1], (8 * k32, 1, 1)));
        match mode {
            Mode::Main { u_dmx, .. } => {
                assert_eq!(k, 4, "attn_dmx2 is for 4 rows");
                let ud = u_dmx[il].as_ref().unwrap();
                c.push(of.op("q_dmx", &[(0, a("qr"))], a("qdm"), None, &of.u([0, 32, 0, 4, 0, 0, 0, 0]), (32, 1, 1)));
                c.push(of.op("attn_dmx2", &[(12, kc), (13, a("qdm")), (14, vc)], a("att_p"), None, ud, (2, ctx().div_ceil(DMX_CH) as u32, 1)));
                c.push(of.op("attn_fd_combine", &[(0, a("att_p"))], a("att"), None, ud, (16, k32, 1)));
                c.push(of.op("attn_gate16", &[(0, a("att")), (1, a("qf"))], a("ag"), None, &u.n_ag, (32 * k32, 1, 1)));
            }
            Mode::Mtp => {
                // short window (the MTP cache starts at mtp_start): one token per grid z, as the single draft
                c.push(of.op("attn_fd", &[(0, a("qr")), (9, kc), (10, vc)], a("att_p"), None, &up[2], (2, (ctx() / 128) as u32, k32)));
                c.push(of.op("attn_fd_combine", &[(0, a("att_p"))], a("att"), None, &up[2], (16, k32, 1)));
                c.push(of.op("attn_gate", &[(0, a("att")), (1, a("qf"))], a("ag"), None, &u.n_ag, (64 * k32, 1, 1)));
            }
        }
        c.push(mv(&l("attn_output.weight"), "ag", "att_out"));
    } else {
        let Mode::Main { save_c, save_d, u_dq, .. } = mode else { panic!("layer {il}: DeltaNet only in the main model") };
        let (sc, sd) = (save_c[il].as_ref().unwrap(), save_d[il].as_ref().unwrap());
        c.push(mv(&l("attn_qkv.weight"), "h", "qkv"));
        c.push(mv(&l("attn_gate.weight"), "h", "z"));
        c.push(mv(&l("ssm_beta.weight"), "h", "braw"));
        c.push(mv(&l("ssm_alpha.weight"), "h", "araw"));
        c.push(of.op("beta_gate2", &[(0, a("braw")), (1, a("araw")), (2, w(l("ssm_dt.bias"))), (3, w(l("ssm_a")))], a("bg"), None, &u.bg, ((32 * k32).div_ceil(64), 1, 1)));
        // binding 8: this step's raw qkv, saved for the commit in the next step
        c.push(of.op("conv_q", &[(0, a("qkv")), (1, w(l("ssm_conv1d.weight"))), (2, &conv[il][par]), (3, &sc[1 - par]), (8, &sc[par])], a("cv"), Some(&conv[il][1 - par]), u_dq, (128, 1, 1)));
        c.push(of.op("rmsnorm", &[(0, a("cv"))], a("qk"), None, &u.l2, (32 * k32, 1, 1)));
        c.push(of.op("delta_q", &[(0, a("qk")), (1, a("cv")), (2, a("bg")), (3, &sd[1 - par]), (8, a("ao"))], &ssm[il], Some(&sd[par]), u_dq, (32, 1, 1)));
        c.push(of.op("rmsnorm", &[(0, a("ao")), (1, w(l("ssm_norm.weight"))), (2, a("z"))], a("y"), None, &u.gnh, (32 * k32, 1, 1)));
        c.push(mv(&l("ssm_out.weight"), "y", "att_out"));
    }
    c.push(of.op("sum", &[(0, a("att_out")), (1, a("x"))], a("x2"), None, &u.n_x, (32 * k32, 1, 1)));
    if attention(il) || il == N_LAYER { mark_attention(&mut c[a0..], il); }
    c.push(of.op("rmsnorm", &[(0, a("x2")), (1, w(l("post_attention_norm.weight")))], a("h2"), None, norm, (k32, 1, 1)));
    if !main {
        // MTP: the MoE only for the last row (the draft comes from it); its result into x at that row
        let xo = 2048 * (k32 - 1);
        c.push(of.mv_x(&l("ffn_gate_inp.weight"), "h2", "lg", xo));
        c.push(of.gpu.command(&of.rt, &[(0, a("lg")), (1, a("probs")), (2, a("ids")), (3, a("weights"))], (1, 1, 1), "routing".into(), 0));
        c.push(of.mv_tok(&l("ffn_gate_exps.weight"), "h2", "eg", 0, xo, 0, "ids"));
        c.push(of.mv_tok(&l("ffn_up_exps.weight"), "h2", "eu", 0, xo, 0, "ids"));
        c.push(of.op("swiglu", &[(0, a("eg")), (1, a("eu"))], a("esw"), None, &of.u([4096, 0, 0, 0, 0, 0, 0, 0]), (64, 1, 1)));
        c.push(of.mv_tok(&l("ffn_down_exps.weight"), "esw", "ed", 512, 0, 0, "ids"));
        c.push(of.mv_x(&l("ffn_gate_shexp.weight"), "h2", "sg", xo));
        c.push(of.mv_x(&l("ffn_up_shexp.weight"), "h2", "su", xo));
        c.push(of.op("swiglu", &[(0, a("sg")), (1, a("su"))], a("ssw"), None, &of.u([512, 0, 0, 0, 0, 0, 0, 0]), (8, 1, 1)));
        c.push(of.mv_x(&l("ffn_down_shexp.weight"), "ssw", "sd", 0));
        c.push(of.mv_x(&l("ffn_gate_inp_shexp.weight"), "h2", "sgt", xo));
        c.push(of.op("moe_sum", &[(0, a("ed")), (1, a("weights")), (2, a("sd")), (3, a("sgt")), (4, a("x2"))], a("fo"), Some(a("x")), &u.moe_last, (32, 1, 1)));
        return;
    }
    c.push(mv(&l("ffn_gate_inp.weight"), "h2", "lg"));
    c.push(of.gpu.command(&of.rtn, &[(0, a("lg")), (1, a("probs")), (2, a("ids4")), (3, a("weights")), (4, &u.route)], (k32, 1, 1), "routing x4".into(), 0));
    c.push(of.gpu.command(&of.un4, &[(0, a("ids4")), (2, a("u4"))], (1, 1, 1), "merge x4".into(), 0));
    c.push(of.mv_u4(&l("ffn_gate_exps.weight"), "h2", "eg", 0, 2048));
    c.push(of.mv_u4(&l("ffn_up_exps.weight"), "h2", "eu", 0, 2048));
    c.push(of.op("swiglu16", &[(0, a("eg")), (1, a("eu"))], a("esw"), None, &u.n_esw, (32 * k32, 1, 1)));
    c.push(of.mv_u4(&l("ffn_down_exps.weight"), "esw", "ed", 512, 4096));
    c.push(mv(&l("ffn_gate_shexp.weight"), "h2", "sg"));
    c.push(mv(&l("ffn_up_shexp.weight"), "h2", "su"));
    c.push(of.op("swiglu16", &[(0, a("sg")), (1, a("su"))], a("ssw"), None, &u.n_ssw, (4 * k32, 1, 1)));
    c.push(mv(&l("ffn_down_shexp.weight"), "ssw", "sd"));
    c.push(mv(&l("ffn_gate_inp_shexp.weight"), "h2", "sgt"));
    c.push(of.op("moe_sum", &[(0, a("ed")), (1, a("weights")), (2, a("sd")), (3, a("sgt")), (4, a("x2"))], a("fo"), Some(a("x")), &u.moe, (32 * k32, 1, 1)));
}

impl Quad {
    /// Builds the buffers and the programs. Needs the f16 KV cache (attn_dmx2) and the MTP layer.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(of: &Factory, conv: &[[wgpu::Buffer; 2]], ssm: &[wgpu::Buffer], kc: &[Option<wgpu::Buffer>], vc: &[Option<wgpu::Buffer>]) -> Self {
        let gpu = of.gpu;
        let tab = gpu.empty_buffer("rope-k", (K * 64 * 4) as u64);
        let tab_m = gpu.empty_buffer("rope-k-mtp", (K * 64 * 4) as u64);
        let u3 = || [gpu.uniform(&[0u32; 8]), gpu.uniform(&[0u32; 8]), gpu.uniform(&[0u32; 8])];
        let u_pos: Vec<Option<[wgpu::Buffer; 3]>> = (0..=N_LAYER).map(|il| (attention(il) || il == N_LAYER).then(u3)).collect();
        let emb_row = emb_format(of.g).1 as usize;
        let emb4 = gpu.empty_buffer("emb-k", (K * emb_row) as u64);
        let u_dmx: Vec<Option<wgpu::Buffer>> = (0..N_LAYER).map(|il| attention(il).then(|| gpu.uniform(&[0u32; 8]))).collect();
        let pair = |n: &str, words: usize| [gpu.empty_buffer(n, (words * 4) as u64), gpu.empty_buffer(n, (words * 4) as u64)];
        let save_c: Vec<Option<[wgpu::Buffer; 2]>> = (0..N_LAYER).map(|il| (!attention(il)).then(|| pair("save-conv", K * 8192))).collect();
        let save_d: Vec<Option<[wgpu::Buffer; 2]>> = (0..N_LAYER).map(|il| (!attention(il)).then(|| pair("save-delta", K * DQ_ROW))).collect();
        let u_dq = gpu.uniform(&[0u32; 8]); let u_dq0 = gpu.uniform(&[0u32; 8]);
        let a = |n: &str| of.a(n);
        let u = uk(of, K);
        let main = Mode::Main { save_c: &save_c, save_d: &save_d, u_dq: &u_dq, u_dmx: &u_dmx };
        let step = |par: usize| {
            let mut c = vec![];
            let u_e = gpu.uniform(&[(K as u32) * of.emb_bpr, 0, 0, 0]);
            c.push(gpu.command(&of.dq, &[(0, &emb4), (1, a("x")), (2, &u_e)], (((K as u32) * of.emb_bpr).div_ceil(64), 1, 1), format!("embedding x{K}"), 0));
            for il in 0..N_LAYER { block_k(of, il, &mut c, &u, K, par, conv, ssm, kc, vc, &u_pos, &tab, &main); }
            c.push(of.op("rmsnorm", &[(0, a("x")), (1, &of.w["output_norm.weight"])], a("xn"), None, &u.norm, (K as u32, 1, 1)));   // h for the MTP, f32
            c.push(of.op("rmsnorm", &[(0, a("x")), (1, &of.w["output_norm.weight"])], a("xn16"), None, &u.normh, (K as u32, 1, 1)));
            c.push(of.mvk("output.weight", "xn16", "logits", K, true));
            argmax_rows(of, &mut c, K);
            c
        };
        // flush: conv_q and delta_q with 0 new rows (u_dq0), only on the DeltaNet layers
        let flush = |par: usize| {
            let mut c = vec![];
            for il in 0..N_LAYER { if !attention(il) {
                let (sc, sd) = (save_c[il].as_ref().unwrap(), save_d[il].as_ref().unwrap());
                c.push(of.op("conv_q", &[(0, a("qkv")), (1, &of.w[&format!("blk.{il}.ssm_conv1d.weight")]), (2, &conv[il][par]), (3, &sc[1 - par]), (8, &sc[par])],
                             a("cv"), Some(&conv[il][1 - par]), &u_dq0, (128, 1, 1)));
                c.push(of.op("delta_q", &[(0, a("qk")), (1, a("cv")), (2, a("bg")), (3, &sd[1 - par]), (8, a("ao"))], &ssm[il], Some(&sd[par]), &u_dq0, (32, 1, 1)));
            } }
            c
        };
        // MTP on `rows` rows: embedding from `emb`, h from `h` (row t at t * 2048), K/V at the positions of `u_pos`/`tab`
        let mtp_prog = |rows: usize, emb: &wgpu::Buffer, h: &wgpu::Buffer, u_pos: &[Option<[wgpu::Buffer; 3]>], tab: &wgpu::Buffer| {
            let r = rows as u32;
            let ur = uk(of, rows);
            let mut c = vec![];
            let u_e = gpu.uniform(&[r * of.emb_bpr, 0, 0, 0]);
            c.push(gpu.command(&of.dq, &[(0, emb), (1, a("m_e")), (2, &u_e)], ((r * of.emb_bpr).div_ceil(64), 1, 1), format!("mtp embedding x{rows}"), 0));
            let u_en = of.u([2048, r, fb(EPS), fb(1.0), 2048, 4096, 1, 0]);      // row t -> m_cat[t*4096]
            let u_hn = of.u([2048, r, fb(EPS), fb(1.0), 2048, 4096, 1, 2048]);   // row t -> m_cat[t*4096 + 2048]
            c.push(of.op("rmsnorm", &[(0, a("m_e")), (1, &of.w["blk.40.nextn.enorm.weight"])], a("m_cat"), None, &u_en, (r, 1, 1)));
            c.push(of.op("rmsnorm", &[(0, h), (1, &of.w["blk.40.nextn.hnorm.weight"])], a("m_cat"), None, &u_hn, (r, 1, 1)));
            c.push(of.mvk("blk.40.nextn.eh_proj.weight", "m_cat", "x", rows, false));
            block_k(of, N_LAYER, &mut c, &ur, rows, 0, conv, ssm, kc, vc, u_pos, tab, &Mode::Mtp);
            c.push(of.op("rmsnorm", &[(0, a("x")), (1, &of.w["blk.40.nextn.shared_head_norm.weight"])], a("xn"), None, &ur.norm, (r, 1, 1)));
            of.draft_head(&mut c, 2048 * (r - 1));
            c
        };
        let mtp: Vec<Vec<Command>> = (1..=K).map(|rows| mtp_prog(rows, a("emb_raw"), a("xn"), &u_pos, &tab_m)).collect();
        let prog = [step(0), step(1)];
        Quad { prog, flush: [flush(0), flush(1)], mtp, tab, u_pos, tab_m, emb4, u_dmx, u_dq, u_dq0, pend: std::cell::Cell::new(0) }
    }
}

/// Argmax of each of the k rows of "logits": am[row], am2[row].
fn argmax_rows(of: &Factory, c: &mut Vec<Command>, k: usize) {
    let a = |n: &str| of.a(n);
    for tok in 0..k as u32 {
        let u1 = of.u([VOCAB as u32, tok * (VOCAB / 256) as u32, 0, 0, 0, 0, 0, tok * VOCAB as u32]);
        let u2 = of.u([(VOCAB / 256) as u32, tok, 0, 0, 0, 0, 0, tok * (VOCAB / 256) as u32]);
        c.push(of.op("argmax1", &[(0, a("logits"))], a("am_v"), Some(a("am_i")), &u1, ((VOCAB / 256) as u32, 1, 1)));
        c.push(of.op("argmax2", &[(0, a("am_v")), (1, a("am_i"))], a("am"), Some(a("am2")), &u2, (1, 1, 1)));
    }
}

impl Engine {
    fn quad(&self) -> &Quad { self.quad.as_ref().expect("the 4-token step needs the f16 KV cache and the MTP layer") }
    /// Whether the 4-token step is available.
    pub fn has_step4(&self) -> bool { self.quad.is_some() }
    /// Rows waiting to be committed into the DeltaNet state (0 = committed).
    pub fn pending4(&self) -> usize { self.quad.as_ref().map_or(0, |q| q.pend.get()) }

    /// Embedding rows of `toks` into `emb`, RoPE tables of `n` positions from `pos` into `tab`.
    fn rows_inputs(&self, toks: &[u32], emb: &wgpu::Buffer, pos: usize, n: usize, tab: &wgpu::Buffer) {
        let te = self.g.tensor("token_embd.weight").unwrap();
        let mut e = Vec::new();
        for &t in toks { let mut r = Vec::new(); self.g.read(te, t as u64 * self.emb_row, self.emb_row, &mut r).unwrap(); e.extend_from_slice(&r); }
        self.gpu.write(emb, 0, &e);
        let tb: Vec<u8> = (0..n).flat_map(|t| rope_table((pos + t) as i32)).flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.write(tab, 0, &tb);
    }
    /// Inputs of the step at `pos`: the embedding rows of the K tokens `toks`, uniforms.
    fn step_inputs(&self, toks: &[u32], pos: usize) {
        assert!(pos + K < ctx());
        let q = self.quad();
        self.rows_inputs(toks, &q.emb4, pos, K, &q.tab);
        let ub = |v: [u32; 8]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        for il in 0..N_LAYER { if attention(il) {
            let u = q.u_pos[il].as_ref().unwrap();
            self.gpu.write(&u[0], 0, &ub([256, 2 * K as u32, 0, 0, 256, 256, 2, (pos * 512) as u32]));
            self.gpu.write(&u[1], 0, &ub([512 * K as u32, 0, 0, 0, 0, 0, 0, (pos * 512) as u32]));
            self.gpu.write(q.u_dmx[il].as_ref().unwrap(), 0, &ub([pos as u32 + 1, ctx().div_ceil(DMX_CH) as u32, fb(1.0 / 16.0), K as u32, 0, 0, DMX_CH as u32, 0]));
        } }
        self.gpu.write(&q.u_dq, 0, &ub([8192, K as u32, 0, 0, 0, 0, K as u32, q.pend.get() as u32]));
    }
    /// Position uniforms of the MTP layer for `c` rows from `q0`.
    fn mtp_uniforms(&self, u: &[wgpu::Buffer; 3], q0: usize, c: usize) {
        let ub = |v: [u32; 8]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        self.gpu.write(&u[0], 0, &ub([256, 2 * c as u32, 0, 0, 256, 256, 2, (q0 * 512) as u32]));
        self.gpu.write(&u[1], 0, &ub([512 * c as u32, 0, 0, 0, 0, 0, 0, (q0 * 512) as u32]));
        self.gpu.write(&u[2], 0, &ub([q0 as u32 + 1, (ctx() / 128) as u32, fb(1.0 / 16.0), 0, 0, 0, 0, self.mtp_start.get()]));
    }
    /// Inputs of the MTP on the rows `toks` at q0.. (`draft_rows`).
    fn mtp_inputs(&self, toks: &[u32], q0: usize) {
        let c = toks.len(); assert!((1..=K).contains(&c));
        let q = self.quad();
        self.rows_inputs(toks, &self.a["emb_raw"], q0, c, &q.tab_m);
        self.mtp_uniforms(q.u_pos[N_LAYER].as_ref().unwrap(), q0, c);
    }

    /// The 4-token step: `toks[0]` at `pos` (sure), toks[1..4] drafts at pos+1..pos+3. Applies first the rows accepted by
    /// the previous step (`accept4`). Returns the argmax of the 4 rows (row i = candidate for pos+i+1); the 4 rows of
    /// logits stay in "logits". Call `accept4` before the next step or `flush4`.
    pub fn step4(&self, toks: [u32; K], pos: usize) -> [u32; K] {
        trace::set_ctx("quad", pos, K, 0);
        let ti = trace::start();
        let q = self.quad();
        self.step_inputs(&toks, pos);
        trace::span(trace::Kind::Inputs, ti, |_| {});
        let par = self.par.get(); self.par.set(1 - par);
        let (prep, wait, gpu, b) = self.gpu.run_commands_read(&q.prog[par], &self.a["am2"], 4 * K as u64);
        if let Some(v) = self.span.borrow_mut().as_mut() { v.push((prep, wait, gpu)); }
        std::array::from_fn(|i| u32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()))
    }
    /// After `step4`: its first `rows` rows (the sure token and the accepted drafts) are the ones that enter the state.
    pub fn accept4(&self, rows: usize) { assert!((1..=K).contains(&rows)); self.quad().pend.set(rows); }
    /// Commits the pending rows into the DeltaNet and conv states (before a step of another kind, a root, a photo).
    pub fn flush4(&self) {
        let Some(q) = self.quad.as_ref() else { return };
        let n = q.pend.get(); if n == 0 { return; }
        trace::set_ctx("flush4", 0, n, 0);
        let ub = |v: [u32; 8]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        self.gpu.write(&q.u_dq0, 0, &ub([8192, 0, 0, 0, 0, 0, K as u32, n as u32]));
        let par = self.par.get(); self.par.set(1 - par);
        self.gpu.run_commands(&q.flush[par]);
        q.pend.set(0);
    }
    /// MTP draft over the rows `toks` at positions q0.. (the tokens after the main rows 0..toks.len()-1 of the last
    /// `step4`, whose h are in "xn"): their K/V into the MTP cache; returns the draft for q0 + toks.len().
    pub fn draft_rows(&self, toks: &[u32], q0: usize) -> u32 {
        let c = toks.len(); assert!((1..=K).contains(&c));
        trace::set_ctx("draft4", q0, c, 0); trace::set_kv((q0 + c).saturating_sub(self.mtp_start.get() as usize));
        self.mtp_inputs(toks, q0);
        let (_, _, _, b) = self.gpu.run_commands_read(&self.quad().mtp[c - 1], &self.a["am2"], 4);
        self.draft_id(u32::from_le_bytes(b[..4].try_into().unwrap()))
    }

}

/// `mechta quad-check <text> <prefill> <cycles>` (deterministic test): from the same state after the
/// prefill, the logits of single steps on the true tokens (reference), of double steps (today's path), and of 4-token
/// steps whose drafts are the true tokens made wrong at a chosen row (accepted 3, 0, 1, 2, ...: every way back), with
/// only the valid rows compared; at the end flush4 and one single step (the committed state). Criterion, written before
/// the run: the 4-token rows no worse than the double step against the same reference (KL median <= 2x and <= 1e-5
/// allowed, KL max <= 2x + 1e-3, same top-1 >= double's - 2 points), and the row after the flush likewise.
pub fn quad_check(text: &str, prefill: usize, cycles: usize) {
    let m = Engine::new(&crate::model_path());
    assert!(m.has_step4(), "quad-check needs the 4-token step (not MECHTA_STEP4=0), the f16 KV cache and the MTP layer");
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(text).expect("text"), false).expect("encode").get_ids().to_vec();
    let n = 4 * cycles + 8;
    assert!(prefill + n < ids.len(), "text too short");
    let t = Instant::now();
    m.reset(); m.prefill(&ids[..prefill], 0);
    println!("prefill {prefill} in {:.0} s", t.elapsed().as_secs_f64());
    m.photograph();
    let refl: Vec<Vec<f32>> = (prefill..prefill + n).map(|q| { m.step(ids[q], q); m.read("logits", VOCAB) }).collect();
    let kl = |q: usize, l: &[f32]| -> (f64, bool) {
        let soft = |l: &[f32]| -> Vec<f64> { let mx = l.iter().cloned().fold(f32::MIN, f32::max) as f64;
            let e: Vec<f64> = l.iter().map(|x| (*x as f64 - mx).exp()).collect(); let z: f64 = e.iter().sum(); e.iter().map(|x| x / z).collect() };
        let am = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let r = &refl[q - prefill];
        let (pa, pb) = (soft(r), soft(l));
        (pa.iter().zip(&pb).filter(|(x, _)| **x > 0.0).map(|(x, y)| x * (x.ln() - y.max(1e-300).ln())).sum(), am(r) == am(l))
    };
    let summary = |name: &str, v: &mut Vec<(f64, bool)>| -> (f64, f64, f64) {
        let same = 100.0 * v.iter().filter(|x| x.1).count() as f64 / v.len() as f64;
        let mut k: Vec<f64> = v.iter().map(|x| x.0).collect(); k.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let (med, max) = (k[k.len() / 2], k[k.len() - 1]);
        println!("{name}: {} rows · KL median {med:.2e} max {max:.2e} · same top-1 {same:.1}%", v.len());
        (med, max, same)
    };
    // today's double step, true tokens (always accepted)
    m.restore_photo();
    let mut dbl = vec![];
    let mut q = prefill;
    while q + 2 <= prefill + 4 * cycles { m.step2(ids[q], ids[q + 1], q); let l = m.read("logits", 2 * VOCAB); dbl.push(kl(q, &l[..VOCAB])); dbl.push(kl(q + 1, &l[VOCAB..])); q += 2; }
    // the 4-token step: draft na + 1 wrong (na = accepted drafts), the rows 0..=na compared
    m.restore_photo();
    let pattern = [3usize, 0, 1, 2, 3, 1, 0, 2];
    let (mut quad, mut by_na) = (vec![], vec![vec![]; 4]);
    let mut q = prefill;
    for c in 0..cycles {
        let na = pattern[c % pattern.len()];
        let mut toks = [ids[q], ids[q + 1], ids[q + 2], ids[q + 3]];
        if na < 3 { toks[na + 1] = (toks[na + 1] + 1) % VOCAB as u32; }
        m.step4(toks, q);
        let l = m.read("logits", K * VOCAB);
        for i in 0..=na { let r = kl(q + i, &l[i * VOCAB..(i + 1) * VOCAB]); quad.push(r); by_na[na].push(r); }
        m.accept4(na + 1);
        q += na + 1;
    }
    m.flush4();
    m.step(ids[q], q);
    let mut fl = vec![kl(q, &m.read("logits", VOCAB))];
    let (dm, dx, ds) = summary("double step (today)", &mut dbl);
    let (qm, qx, qs) = summary("4-token step", &mut quad);
    for (na, v) in by_na.iter_mut().enumerate() { if !v.is_empty() { summary(&format!("  cycles with {na} drafts accepted"), v); } }
    let (fm, _, fs) = summary("single step after flush4", &mut fl);
    let ok = qm <= (2.0 * dm).max(1e-5) && qx <= 2.0 * dx + 1e-3 && qs >= ds - 2.0 && fm <= (2.0 * dm).max(1e-5).max(2.0 * dx + 1e-3) && fs >= 100.0;
    println!("{}", if ok { "PASS" } else { "FAIL" });
    // MTP on c rows (draft_rows) against today's path on the same tokens: single steps, each followed by its draft;
    // the last draft must be the same token, its draft-head logits close. Every c from the same photo.
    let p0 = q + 1;
    m.photograph();
    let heads = if m.head_ids.is_empty() { VOCAB } else { m.head_ids.len() };
    let mut mtp_ok = true;
    for c in 1..=K {
        m.restore_photo();
        let mut dr = 0;
        for i in 0..c { m.step(ids[p0 + i], p0 + i); dr = m.draft(ids[p0 + i + 1], p0 + i + 1); }
        let lr = m.read("logits", heads);
        m.restore_photo();
        m.step4([ids[p0], ids[p0 + 1], ids[p0 + 2], ids[p0 + 3]], p0);
        m.accept4(c);
        let dq = m.draft_rows(&ids[p0 + 1..p0 + 1 + c], p0 + 1);
        let lq = m.read("logits", heads);
        m.flush4();
        let nan = lq.iter().filter(|x| !x.is_finite()).count();
        let dmax = lr.iter().zip(&lq).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        println!("MTP on {c} rows: draft {dq} vs {dr} (today's path) · max |logit difference| {dmax:.3} · non-finite {nan}");
        mtp_ok &= dq == dr && nan == 0;
    }
    println!("MTP {}", if mtp_ok { "PASS" } else { "FAIL" });
}

#[cfg(test)]
mod tests {
    fn validate(src: &str) {
        let module = naga::front::wgsl::parse_str(src).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module).unwrap_or_else(|e| panic!("{}", e.emit_to_string(src)));
    }
    #[test]
    fn kernels_validate() {
        validate(&super::q3_source("matvec_u4"));
        validate(include_str!("wgsl/merge.wgsl"));
        validate(&super::ops_source(super::KvFormat::F16));
        validate(&super::ops_source(super::KvFormat::F32));
    }
}
