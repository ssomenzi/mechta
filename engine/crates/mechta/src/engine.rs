//! The engine: one token at a time, 40 layers, everything in f32 (KV cache included).
//! M2: all the weights on the GPU, experts included (~21 GB). The dispatch list of a token is
//! built only once (two copies: the conv state alternates between two buffers, even and odd
//! positions); at every token the small inputs are written and everything is sent in a single submit.
//! The GPU reads the chosen experts by itself (matvec with `ids` written by the routing).
use mechta_gguf::{Gguf, TType};
use mechta_gpu::{Command, Gpu, wgpu};
use std::cell::RefCell;
use std::collections::HashMap;
use std::time::Instant;

pub const EPS: f32 = 1e-6;
pub const N_LAYER: usize = 40;
pub const VOCAB: usize = 248_320;
/// M3: tokens per block in the prefill. The buffers are sized on NB (the maximum); the programs exist for the blocks
/// in `BLOCKS` (largest first): the prefill uses blocks of 512 while it can, then of 128 (the last one can be shorter).
/// Large blocks = fewer expert reads per token (the union at 128 tokens already touches ~252 experts of 256).
pub const NB: usize = 512;
pub const BLOCKS: [usize; 4] = [512, 128, 32, 8];   // 32 and 8: the following turns have few new tokens
/// Context size in tokens, chosen at start (before: the constant CTX = 65536). Order: `set_ctx` (the
/// `--ctx` option), then MECHTA_CTX, then the default. Fixed for the life of the process: the KV cache, the flash-decoding
/// partials and the dispatch sizes are built on it. The attn v1 kernels (`punt[1024]`) only work up to 1024 positions.
pub const CTX_DEFAULT: usize = 65536;
pub const CTX_MIN: usize = 4096;
pub const CTX_MAX: usize = 262_144;   // native context of Qwen3.6-35B-A3B
static CTX_CELL: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
/// Checks a context size: a multiple of 1024 (blocks of 64/128/256 positions in the attention kernels), in [CTX_MIN, CTX_MAX].
pub fn check_ctx(n: usize) -> Result<usize, String> {
    if n % 1024 != 0 || !(CTX_MIN..=CTX_MAX).contains(&n) {
        return Err(format!("context {n}: must be a multiple of 1024 between {CTX_MIN} and {CTX_MAX}"));
    }
    Ok(n)
}
/// Sets the context size before the engine is built. Panics if already set to another value.
pub fn set_ctx(n: usize) {
    let n = check_ctx(n).unwrap_or_else(|e| panic!("{e}"));
    let v = *CTX_CELL.get_or_init(|| n);
    assert_eq!(v, n, "context already fixed at {v}");
}
pub fn ctx() -> usize {
    *CTX_CELL.get_or_init(|| match std::env::var("MECHTA_CTX") {
        Ok(v) if !v.is_empty() => check_ctx(v.parse().unwrap_or(0)).unwrap_or_else(|e| panic!("MECHTA_CTX={v}: {e}")),
        _ => CTX_DEFAULT,
    })
}

fn fb(v: f32) -> u32 { v.to_bits() }

/// Format of the KV cache. MECHTA_KV=f16 (default) | f32 | q8 (int8 + one f32 scale per 32 values).
/// The attention reads the whole cache at every generated token: f16 halves those bytes, q8 cuts them to 1.125/4.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvFormat { F32, F16, Q8, F16Sim }
pub fn kv_format() -> KvFormat {
    static F: std::sync::OnceLock<KvFormat> = std::sync::OnceLock::new();
    *F.get_or_init(|| match std::env::var("MECHTA_KV").unwrap_or_default().as_str() {
        "f32" => KvFormat::F32, "" | "f16" => KvFormat::F16, "q8" => KvFormat::Q8,
        "f16sim" => KvFormat::F16Sim,   // test only: f32 cache, each value rounded to f16 when read
        x => panic!("MECHTA_KV={x}: use f32, f16 or q8"),
    })
}
impl KvFormat {
    pub fn name(self) -> &'static str { match self { KvFormat::F32 => "f32", KvFormat::F16 => "f16", KvFormat::Q8 => "q8", KvFormat::F16Sim => "f16sim" } }
    /// u32 words of data for `n` cache elements (q8: the scales are after the data, see `scale_words`)
    pub fn data_words(self, n: usize) -> usize { match self { KvFormat::F32 | KvFormat::F16Sim => n, KvFormat::F16 => n / 2, KvFormat::Q8 => n / 4 } }
    /// q8 only: words of scales for `n` elements, stored from word `data_words(ctx() * 512)`
    pub fn scale_words(self, n: usize) -> usize { if self == KvFormat::Q8 { n / 32 } else { 0 } }
    /// bytes of one KV buffer (K or V of one layer) for the whole context
    pub fn buffer_bytes(self) -> u64 { self.buffer_bytes_for(ctx()) }
    /// the same for a context of `ctx` tokens (memory check before the start)
    pub fn buffer_bytes_for(self, ctx: usize) -> u64 { 4 * (self.data_words(ctx * 512) + self.scale_words(ctx * 512)) as u64 }
}
/// Format of the token embedding table: (dequant.wgsl entry, bytes of one row
/// of 2048 values, blocks per row); Ornith has token_embd in Q4_K. Before, Q8_0 was assumed everywhere (2176 bytes, 64 blocks).
pub fn emb_format(g: &Gguf) -> (&'static str, u64, u32) {
    match g.tensor("token_embd.weight").expect("token_embd").ttype {
        TType::Q8_0 => ("q8_0", 2176, 64), TType::Q4_K => ("q4_k", 1152, 8), TType::Q6_K => ("q6_k", 1680, 8),
        x => panic!("token_embd.weight: {x:?} not supported"),
    }
}
/// The four shader files that use the Q3_K decoding (q3k.wgsl), with it appended. Every pipeline built from
/// them must take the source from here: compiled alone they do not parse.
pub fn q3_source(name: &str) -> String {
    let s = match name {
        "matvec4" => include_str!("wgsl/matvec4.wgsl"), "matvec_u" => include_str!("wgsl/matvec_u.wgsl"),
        "matvec_uh" => include_str!("wgsl/matvec_uh.wgsl"), "gemm" => include_str!("wgsl/gemm.wgsl"), x => panic!("q3_source({x})"),
    };
    format!("{s}\n{}", include_str!("wgsl/q3k.wgsl"))
}
/// ops.wgsl with the KV access block of the chosen format (kv.wgsl), plus the writer kernels kv_rope / kv_rope2 / kv_copy.
pub fn ops_source(f: KvFormat) -> String {
    let ops = include_str!("wgsl/ops.wgsl");
    if f == KvFormat::F32 { return ops.to_string(); }
    if f == KvFormat::F16Sim {
        let r = |x: &str| format!("let v = {x}; return vec4<f32>(unpack2x16float(pack2x16float(v.xy)), unpack2x16float(pack2x16float(v.zw)));");
        return ops.replace("fn kk(i: u32) -> vec4<f32> { return b4[i]; }", &format!("fn kk(i: u32) -> vec4<f32> {{ {} }}", r("b4[i]")))
                  .replace("fn vv(i: u32) -> vec4<f32> { return c4[i]; }", &format!("fn vv(i: u32) -> vec4<f32> {{ {} }}", r("c4[i]")));
    }
    let kv = include_str!("wgsl/kv.wgsl");
    let block = |name: &str| -> &str {
        let start = kv.find(&format!("// FORMAT {name}")).expect("kv.wgsl block");
        let rest = &kv[start..]; let end = rest[1..].find("// FORMAT ").map(|e| e + 1).unwrap_or(rest.len());
        &rest[..end]
    };
    let chosen = block(f.name()).replace("KV_SC_VALUE", &f.data_words(ctx() * 512).to_string());
    let (b, e) = (ops.find("// KV-ACCESS-BEGIN").expect("KV-ACCESS-BEGIN"), ops.find("// KV-ACCESS-END").expect("KV-ACCESS-END"));
    format!("{}{}\n{}{}", &ops[..b], chosen, block("common"), &ops[e..])
}
/// The kernel that writes K or V into the cache: rope / rope2 / copy (f32), kv_rope / kv_rope2 / kv_copy otherwise.
fn kv_writer(e: &'static str) -> &'static str {
    if matches!(kv_format(), KvFormat::F32 | KvFormat::F16Sim) { return e; }
    match e { "rope" => "kv_rope", "rope2" => "kv_rope2", "copy" => "kv_copy", _ => unreachable!() }
}
fn attention(il: usize) -> bool { (il + 1) % 4 == 0 }

/// cos|sin RoPE table for one position, as ggml_mrope_cache_init (text: same positions).
pub fn rope_table(pos: i32) -> Vec<f32> {
    let scale = 1e7f32.powf(-2.0 / 64.0);
    let mut th = pos as f32; let (mut c, mut s) = (vec![0f32; 32], vec![0f32; 32]);
    for i in 0..32 { c[i] = th.cos(); s[i] = th.sin(); th *= scale; }
    [c, s].concat()
}

/// Bindings used by each entry point of ops.wgsl (the automatic layout contains only those).
pub fn binding_op(e: &str) -> &'static [u32] {
    match e { "rmsnorm" | "attn" => &[0, 1, 2, 5, 7], "conv" => &[0, 1, 2, 5, 6, 7], "delta" => &[0, 1, 2, 5, 6],
              "argmax1" => &[0, 5, 6, 7], "argmax2" => &[0, 1, 5, 6, 7],
              "delta2" => &[0, 1, 2, 5, 6, 8], "beta_gate2" => &[0, 1, 2, 3, 5, 7], "conv2" => &[0, 1, 2, 5, 6, 7, 8], "attn2" => &[0, 1, 2, 5, 7],
              "swiglu16" | "attn_gate16" => &[0, 1, 5, 7], "attn_v2" | "attn2_v2" => &[0, 1, 2, 5, 7], "attn_v3" | "attn2_v3" => &[0, 2, 5, 7, 9],   // K only as vec4 (binding 9), not from b
              "attn_fd" | "attn_fd2" | "attn_fd2b" | "attn_fd2c" | "attn_fd2d" | "attn_fd2e" | "attn_fd2_read" | "attn_fd2_read2" | "attn_fd2f" | "attn_fd2g" | "attn_fd2cm" | "attn_fd2h" | "attn_fd2i" => &[0, 5, 7, 9, 10], "attn_fd_combine" => &[0, 5, 7],
              "conv_n" => &[0, 1, 2, 5, 6, 7], "delta_n" => &[0, 1, 2, 5, 7, 8],   // M3
              "attn_fd4" | "attn_fa" => &[0, 5, 7, 9, 10],
              "attn_fa_mx" => &[5, 7, 10, 12, 13], "attn_fa_mx2" => &[5, 7, 12, 13, 14], "q_f16" => &[0, 5, 7],   // matrix units (fa_mx.wgsl)
              "beta_gate" => &[0, 1, 2, 3, 5, 7], "moe_sum" => &[0, 1, 2, 3, 4, 5, 6, 7], "copy" => &[0, 5, 7],
              "kv_rope" | "kv_rope2" => &[0, 1, 7, 11], "kv_copy" => &[0, 7, 11],   // compact KV cache
              _ => &[0, 1, 5, 7] }
}

/// MECHTA_Q4K=64: Q4_K in pieces of 64 values (mv_q4_k64 of matvec4.wgsl), on trial.
pub fn q4k_64() -> bool {
    static Q: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *Q.get_or_init(|| std::env::var("MECHTA_Q4K").map(|v| v == "64").unwrap_or(false))
}

/// Attention: `attn_v2`/`attn2_v2` parallel over the positions (default); MECHTA_ATTN=v1 for the old ones.
fn attn_entry(double: bool) -> &'static str {
    let v = std::env::var("MECHTA_ATTN").unwrap_or_default();
    assert!(v.is_empty() || kv_format() == KvFormat::F32, "MECHTA_ATTN={v} reads the cache as f32: only the default attention reads MECHTA_KV={}", kv_format().name());
    match (double, v.as_str()) {
        (false, "v1") => "attn", (true, "v1") => "attn2",
        (false, "v2") => "attn_v2", (true, "v2") => "attn2_v2",
        (false, "v3") => "attn_v3", (true, "v3") => "attn2_v3",
        _ => "fd",   // default: flash-decoding with K/V read once for the 8 Q heads
    }
}
/// Attention in the program: `fd` = two dispatches (blocks + merge), otherwise one dispatch. `ntok` = 1 or 2.
#[allow(clippy::too_many_arguments)]
fn attn_op(of: &Factory, c: &mut Vec<Command>, double: bool, kc: &wgpu::Buffer, vc: &wgpu::Buffer, u_att: &wgpu::Buffer) {
    let a = |n: &str| of.a(n); let ntok = if double { 2 } else { 1 };
    let e = attn_entry(double);
    if e == "fd" {
        // double step with K/V read once for both tokens (attn_fd2); MECHTA_ATTN2=old for one workgroup per token
        if double && attn2_b256() {
            let e2 = match std::env::var("MECHTA_ATTN2").unwrap_or_default().as_str() { "b256" => "attn_fd2b", "d" => "attn_fd2d", "e" => "attn_fd2e",
                // measure only (output wrong): the grid of attn_fd2c that only reads K and V
                "read" => "attn_fd2_read", "read2" => "attn_fd2_read2",
                // trials: f, g = K read in order (slower), cm = measuring copy
                "f" => "attn_fd2f", "g" => "attn_fd2g", "cm" => "attn_fd2cm", "c" => "attn_fd2c", "i" => "attn_fd2i",
                // default: attn_fd2h, 4 rows of V in flight, -18% on attn_fd2c at 45K-124K with
                // a numeric PASS; MECHTA_ATTN2=c for attn_fd2c (the previous default)
                _ => "attn_fd2h" };
            c.push(of.op(e2, &[(0, a("qr")), (9, kc), (10, vc)], a("att_p"), None, u_att, (2, (ctx() / 256) as u32, 1)));
        } else if double && std::env::var("MECHTA_ATTN2").map(|v| v != "old").unwrap_or(true) {
            c.push(of.op("attn_fd2", &[(0, a("qr")), (9, kc), (10, vc)], a("att_p"), None, u_att, (2, (ctx() / 128) as u32, 1)));
        } else {
            c.push(of.op("attn_fd", &[(0, a("qr")), (9, kc), (10, vc)], a("att_p"), None, u_att, (2, (ctx() / 128) as u32, ntok)));
        }
        c.push(of.op("attn_fd_combine", &[(0, a("att_p"))], a("att"), None, u_att, (16, ntok, 1)));   // same uniform: n, extra, rows
    } else {
        c.push(of.op(e, &[(0, a("qr")), (1, kc), (2, vc), (9, kc)], a("att"), None, u_att, (16, ntok, 1)));
    }
}

/// MECHTA_KERNEL=m1: matvec and routing of M1 (in wgsl/m1/), to compare the versions in M2.
/// Measurements only: MECHTA_FD_PHASE=1 (attn_fd without the V phase) or 2 (without the K phase). Output wrong.
/// Prefill: tokens of the lot done in sequence by the same attention workgroup (8, default); MECHTA_ATTN_BATCH=0 for
/// one workgroup per token (the previous way).
/// Positions per workgroup of attn_fd4 in the prefill: 64 (default), 128, 256 or 512, MECHTA_PREFILL_BLOCK.
/// Measured: 256 cuts the merge from 4.3 to 1.2 ms/token at 100K but attn_fd4 grows by 3.9:
/// no gain, so the default stays 64.
/// How often a prefill asks whether the client is still there (`prefill_cancellable`).
pub const KEEPALIVE_EVERY: std::time::Duration = std::time::Duration::from_millis(500);

pub fn prefill_block() -> usize {
    static B: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *B.get_or_init(|| match std::env::var("MECHTA_PREFILL_BLOCK").unwrap_or_default().as_str() {
        "" | "64" => 64, "128" => 128, "256" => 256, "512" => 512,
        x => panic!("MECHTA_PREFILL_BLOCK={x}: use 64, 128, 256 or 512"),
    })
}
/// Attention of the prefill: `fa` = attn_fa, flash attention on groups of 64 tokens and chunks of
/// FA_CH positions (blocks of 64 tokens or more; default); `fd4` = attn_fd4 on
/// lots of 8 tokens (before). MECHTA_PREFILL_ATTN.
pub const FA_CH: usize = 2048;   // = FA_CH of ops.wgsl
/// chunks per dispatch of attn_fa: 8 x 2048 = 16K positions, ~160 ms at 100K (as one lot of attn_fd4 was)
pub const FA_SPLIT: usize = 8;
/// measuring only: MECHTA_FA_PHASE=1..4 skips one phase of attn_fa (output wrong)
fn fa_phase() -> u32 { std::env::var("MECHTA_FA_PHASE").ok().and_then(|v| v.parse().ok()).unwrap_or(0) }
fn fa_splits() -> usize { (ctx() / FA_CH).div_ceil(FA_SPLIT) }
pub fn prefill_fa() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(|| match std::env::var("MECHTA_PREFILL_ATTN").unwrap_or_default().as_str() {
        "" | "fa" | "mx" | "mx2" => true, "fd4" => false,
        x => panic!("MECHTA_PREFILL_ATTN={x}: use fa, mx, mx2 or fd4"),
    })
}
/// MECHTA_PREFILL_ATTN=mx: attn_fa_mx, attn_fa with Q K^T on the matrix units
/// (cooperative matrix, an experimental feature of wgpu asked by crates/gpu). f16 KV cache only.
/// Default with the f16 KV cache; MECHTA_PREFILL_ATTN=fa for attn_fa. Same choice as crates/gpu.
pub fn prefill_mx() -> bool {
    static F: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F.get_or_init(mechta_gpu::prefill_mx_requested)
}
/// The matrix-unit kernel: attn_fa_mx (QK on the matrix units) or, with MECHTA_PREFILL_ATTN=mx2, attn_fa_mx2 (PV too).
fn mx_entry() -> &'static str { if std::env::var("MECHTA_PREFILL_ATTN").map(|v| v == "mx2").unwrap_or(false) { "attn_fa_mx2" } else { "attn_fa_mx" } }
/// The source of attn_fa_mx and q_f16: ops.wgsl (f16 KV) + fa_mx.wgsl, with the two enables in front.
pub fn fa_mx_source() -> String {
    format!("enable f16;\nenable wgpu_cooperative_matrix;\n{}\n{}", ops_source(KvFormat::F16), include_str!("wgsl/fa_mx.wgsl"))
}
fn attn_batch() -> u32 { std::env::var("MECHTA_ATTN_BATCH").ok().and_then(|v| v.parse().ok()).unwrap_or(8) }
/// Double-step attention with blocks of 256 positions: attn_fd2h (default; attn_fd2c before, now
/// MECHTA_ATTN2=c), or the
/// variants b256 / d / e; MECHTA_ATTN2=fd2 for attn_fd2 (blocks of 128), =old for attn_fd per token. The merge needs p.flag = 256.
fn attn2_b256() -> bool { std::env::var("MECHTA_ATTN2").map(|v| v != "fd2" && v != "old").unwrap_or(true) }
fn attn2_flag() -> u32 { if attn2_b256() { 256 } else { 0 } }
fn fd_phase() -> u32 { std::env::var("MECHTA_FD_PHASE").ok().and_then(|v| v.parse().ok()).unwrap_or(0) }
pub fn kernel_m1() -> bool {
    static M1: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *M1.get_or_init(|| std::env::var("MECHTA_KERNEL").map(|v| v == "m1").unwrap_or(false))
}

/// Matvec geometry: (threads per row, workgroups x, workgroups y). A row has min(64, k/32)
/// threads, a workgroup of 64 does 64 / tpr rows; beyond 65535 workgroups it goes onto the y axis.
pub fn geometry_mv(rows: usize, k: usize) -> (u32, u32, u32) { mv_piece_geometry(rows, k, 32) }
/// As `geometry_mv`, with pieces of `piece` values (32, or 64 for the Q4_K with double pieces).
pub fn mv_piece_geometry(rows: usize, k: usize, piece: usize) -> (u32, u32, u32) {
    if kernel_m1() { return (64, rows.min(65535) as u32, rows.div_ceil(65535) as u32); }
    let tpr = (k / piece).min(64);
    assert!(tpr.is_power_of_two() && k % piece == 0, "k = {k}");
    let groups = rows.div_ceil(64 / tpr);
    (tpr as u32, groups.min(65535) as u32, groups.div_ceil(65535) as u32)
}

/// Tensor name without `blk.N.` and without `.weight`.
fn short(n: &str) -> String {
    let s = n.strip_prefix("blk.").map(|r| r.split_once('.').unwrap().1).unwrap_or(n);
    s.trim_end_matches(".weight").to_string()
}

#[derive(Default)]
pub struct Profile { pub gpu: HashMap<String, (f64, u64, u32)>, pub cpu: HashMap<&'static str, f64> }

/// Model state after the first `p` tokens (root of the prefix across sessions), in RAM.
pub struct Root { pub p: usize, pub ssm: Vec<Vec<f32>>, pub conv: Vec<Vec<f32>>, pub kv: Vec<(Vec<f32>, Vec<f32>)> }

pub struct Engine {
    pub g: Gguf, pub gpu: Gpu,
    /// bytes of one row of token_embd (emb_format)
    emb_row: u64,
    a: HashMap<&'static str, wgpu::Buffer>,
    conv: Vec<[wgpu::Buffer; 2]>, ssm: Vec<wgpu::Buffer>,
    /// uniforms that change with the position, for the attention layers: (rope k, copy v, attn)
    u_pos: Vec<Option<[wgpu::Buffer; 3]>>,
    tab: wgpu::Buffer,
    /// the dispatches of one token, for even [0] and odd [1] position; `end_layer[il]` = index after layer il
    prog: [Vec<Command>; 2], end_layer: Vec<usize>,
    pub byte_gpu: u64,
    /// Measurements (M2): None = no measurement, the normal path.
    /// MTP layer (M4): program of one draft, segment signatures (after eh_proj, attention, MoE, final norm, logits),
    /// its own RoPE table (position + 1). The position uniforms are u_pos[N_LAYER].
    prog_mtp: Vec<Command>, pub end_mtp: Vec<usize>, m_tab: wgpu::Buffer,
    /// Double step (M4): program for 2 tokens (per conv parity), end of each layer, DeltaNet state after the
    /// first token, RoPE tables of the 2 tokens, position uniforms for 2 tokens.
    prog2: [Vec<Command>; 2], pub end2: Vec<usize>, ssm1: Vec<wgpu::Buffer>, conv1: Vec<wgpu::Buffer>, tab2: wgpu::Buffer, u_pos2: Vec<Option<[wgpu::Buffer; 3]>>,
    /// 2-token MTP (MTP cycle: after an accepted draft the MTP cache must receive the two new tokens); parity
    /// used by the last double step (to know which conv state to restore on rejection)
    prog_mtp2: Vec<Command>, last_par: std::cell::Cell<usize>,
    /// true ids of the rows of the reduced head (empty = full head): the argmax of the draft is an index into this
    pub head_ids: Vec<u32>,
    /// M3: prefill in blocks of NB tokens: its own buffers, RoPE tables of the block, uniforms rewritten at every block
    /// (conv_n, delta_n, routing), position uniforms per attention layer (rope K, copy V, attention in lots of 8),
    /// program per conv parity.
    ap: HashMap<&'static str, wgpu::Buffer>, tab_n: wgpu::Buffer, u_n: [wgpu::Buffer; 3],
    u_pos_n: Vec<Option<(wgpu::Buffer, wgpu::Buffer, Vec<wgpu::Buffer>)>>, prog_n: Vec<[Vec<Command>; 2]>, end_n: Vec<Vec<usize>>,
    /// Where the prefill splits its submits: the layer ends of `end_n` plus cuts inside the attention (see `attn_cuts`)
    cuts_n: Vec<Vec<usize>>,
    /// First position of the MTP layer's KV cache that the MTP attention looks at (default 1). The Session
    /// fills the MTP cache only with the tail of the prompt and raises this value: the drafts are attempts, the output stays exact.
    pub mtp_start: std::cell::Cell<u32>,
    /// Photo of the DeltaNet + conv states (for reusing the prefix across requests): (ssm, conv) per DeltaNet layer.
    photo: RefCell<Vec<(wgpu::Buffer, wgpu::Buffer)>>,
    /// KV cache per layer (Some for the attention layers and for the MTP): needed by the root (prefix across sessions)
    kv_buf: (Vec<Option<wgpu::Buffer>>, Vec<Option<wgpu::Buffer>>),
    /// Parity of the conv state (which of the two buffers is read): it flips at every step, single or double.
    /// Before it was pos % 2: wrong after a double step (a profiling run found 13 wrong argmax of 27).
    par: std::cell::Cell<usize>,
    pub prof: RefCell<Option<Profile>>,
    /// If Some: every submit without per-dispatch measurements records (preparation, wait, GPU time from the first to the last dispatch).
    pub span: RefCell<Option<Vec<(f64, f64, f64)>>>,
}

/// What is needed to build the commands (pipelines, weights). It does not stay in the engine: the bind groups
/// keep alive the buffers they use.
struct Factory<'a> {
    gpu: &'a Gpu, g: &'a Gguf,
    mv: HashMap<&'static str, wgpu::ComputePipeline>, op: HashMap<&'static str, wgpu::ComputePipeline>,
    /// M4: n-column matvec (matvec_n4.wgsl) and routing of the second token
    mvn: HashMap<&'static str, wgpu::ComputePipeline>, rt2: wgpu::ComputePipeline,
    /// M4: experts in union (matvec_u.wgsl) and the kernel that computes the union
    mvu: HashMap<&'static str, wgpu::ComputePipeline>, un: wgpu::ComputePipeline,
    /// M4: x in f16 in the double step: 2-column matvec (matvec_nh) and experts in union (matvec_uh)
    mvh: HashMap<&'static str, wgpu::ComputePipeline>, mvuh: HashMap<&'static str, wgpu::ComputePipeline>,
    /// F32 of matvec4 (with x_off): the engine's `mv_f32` comes from matvec.wgsl, which does NOT have x_off (a past bug: mtp2 19/27)
    mv4_f32: wgpu::ComputePipeline,
    /// rows of the reduced head for the drafts (0 = full head)
    head_k: usize,
    rt: wgpu::ComputePipeline, dq: wgpu::ComputePipeline,
    /// blocks of one token_embd row for `dq` (emb_format)
    emb_bpr: u32,
    w: HashMap<String, wgpu::Buffer>, a: &'a HashMap<&'static str, wgpu::Buffer>, empty: wgpu::Buffer, local_ids: wgpu::Buffer,
    /// M3: NB-column GEMM (gemm.wgsl), per-token routing and grouping by expert (prefill)
    gm: HashMap<&'static str, wgpu::ComputePipeline>, rtn: wgpu::ComputePipeline, rg: wgpu::ComputePipeline,
}

impl Factory<'_> {
    fn a(&self, n: &str) -> &wgpu::Buffer { &self.a[n] }
    fn op(&self, e: &'static str, inputs: &[(u32, &wgpu::Buffer)], o: &wgpu::Buffer, o2: Option<&wgpu::Buffer>, u: &wgpu::Buffer, g: (u32, u32, u32)) -> Command {
        let mut b = vec![];
        for &i in binding_op(e) {
            let buf = match i { 5 => o, 6 => o2.expect("o2"), 7 => u, _ => inputs.iter().find(|(j, _)| *j == i).map(|(_, x)| *x).unwrap_or(&self.empty) };
            b.push((i, buf));
        }
        self.gpu.command(&self.op[e], &b, g, e.to_string(), 0)
    }
    fn u(&self, v: [u32; 8]) -> wgpu::Buffer { self.gpu.uniform(&v) }
    /// M3: y = W X with X of NB columns (prefill), dense weights read once per block. x f32 (token t at t * k), y[t * rows + r].
    fn gemm(&self, name: &str, x: &str, y: &str, nb: usize) -> Command {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}"));
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        assert!(!name.contains("_exps"));
        // tile 128 x 64 (gemm2, default); MECHTA_GEMM_DENSE=old for the 64 x 64 tile (gemm)
        let old = std::env::var("MECHTA_GEMM_DENSE").map(|v| v == "old").unwrap_or(false);
        let suf = match t.ttype { TType::Q8_0 => "q8_0", TType::Q4_K => "q4_k", TType::Q6_K => "q6_k", TType::F32 | TType::BF16 => "f32", x => panic!("{x:?}") };
        let (e, bm): (&'static str, usize) = if old { (Box::leak(format!("gemm_{suf}").into_boxed_str()), 64) } else { (Box::leak(format!("gemm2_{suf}").into_boxed_str()), 128) };
        let u = self.u([rows as u32, k as u32, 0, 0, 0, 0, nb as u32, k as u32]);
        let byte = if t.ttype == TType::BF16 { 2 * t.nbytes } else { t.nbytes };
        self.gpu.command(&self.gm[e], &[(0, &self.w[name]), (1, self.a(x)), (2, self.a(y)), (3, &self.local_ids), (4, &u)], ((rows as u32).div_ceil(bm as u32), (nb as u32).div_ceil(64), 1), format!("{} x{nb}", short(name)), byte)
    }
    /// M3: expert GEMM in the prefill: grid z = 256 experts, columns = the tokens that use the expert (list `grp` of the
    /// gather kernel); x of token t, slot s at t * x_tok + s * x_step; y at (t * 8 + s) * rows (as `ed`/`eg` per token).
    fn gemm_e(&self, name: &str, x: &str, y: &str, x_step: u32, x_tok: u32, nb: usize) -> Command {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}"));
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        // weights in registers, 2 rows per thread (gemmr5, default); MECHTA_GEMM_EXPERTS=r (1 row), tile16, narrow, wide
        let var = std::env::var("MECHTA_GEMM_EXPERTS").unwrap_or_default();
        let (pre, bn) = match var.as_str() { "wide" => ("gemm", 64), "narrow" => ("gemm8", 8), "tile16" => ("gemm16", 16), "r" => ("gemmr", 16), _ => ("gemmr5", 16) };
        let suf = match t.ttype { TType::Q4_K => "q4_k", TType::Q6_K => "q6_k", TType::Q8_0 => "q8_0", TType::Q3_K => "q3_k", x => panic!("{x:?}") };
        // Q3_K only with the default expert GEMM (gemmr5)
        if t.ttype == TType::Q3_K { assert_eq!(pre, "gemmr5", "{name}: Q3_K experts need the default MECHTA_GEMM_EXPERTS"); }
        let e: &'static str = Box::leak(format!("{pre}_{suf}").into_boxed_str());
        let bm = if pre == "gemm16" || pre == "gemmr5" { 128 } else { 64 };
        let phase: u32 = std::env::var("MECHTA_GEMM_PHASE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);   // measurements only, output wrong
        let u = self.u([rows as u32, k as u32, phase, 2, x_step, 0, nb as u32, x_tok]);
        let ny = (nb as u32).div_ceil(bn as u32);
        self.gpu.command(&self.gm[e], &[(0, &self.w[name]), (1, self.a(x)), (2, self.a(y)), (3, self.a("grp")), (4, &u)], ((rows as u32).div_ceil(bm), ny, 256), format!("experts {} x{nb}", short(name)), t.nbytes)
    }
    /// M4: y = W X with X of 2 columns (two tokens in a row in `x`, outputs in a row in `y`), dense weights read once.
    fn mv2(&self, name: &str, x: &str, y: &str) -> Command { self.mv2_(name, x, y, false) }
    /// As mv2 with x in f16 (written by the producers with flag 4 / swiglu16 / attn_gate16).
    fn mv2h(&self, name: &str, x: &str, y: &str) -> Command { self.mv2_(name, x, y, true) }
    fn mv2_(&self, name: &str, x: &str, y: &str, f16: bool) -> Command {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}"));
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        assert!(!name.contains("_exps"));
        let e = match t.ttype { TType::Q8_0 => "mv_q8_0", TType::Q4_K => "mv_q4_k", TType::Q6_K => "mv_q6_k", TType::F32 | TType::BF16 => "mv_f32", x => panic!("{x:?}") };
        let (tpr, nx, ny) = mv_piece_geometry(rows, k, 32);
        let u = self.u([rows as u32, k as u32, nx, 0, 0, tpr, 2, k as u32]);
        let nb = if t.ttype == TType::BF16 { 2 * t.nbytes } else { t.nbytes };
        let p = if f16 { &self.mvh[e] } else { &self.mvn[e] };
        self.gpu.command(p, &[(0, &self.w[name]), (1, self.a(x)), (2, self.a(y)), (3, &self.local_ids), (4, &u)], (nx, ny, 1), format!("{} x2{}", short(name), if f16 { " h" } else { "" }), nb)
    }
    /// M4: single-column dense matvec with x starting at `x_off` (double draft: output head only for the second token).
    fn mv_x(&self, name: &str, x: &str, y: &str, x_off: u32) -> Command {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}"));
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        let e = match t.ttype { TType::Q8_0 => "mv_q8_0", TType::Q4_K => "mv_q4_k", TType::Q6_K => "mv_q6_k", TType::F32 | TType::BF16 => "mv_f32", x => panic!("{x:?}") };
        let (tpr, nx, ny) = mv_piece_geometry(rows, k, 32);
        let u = self.u([rows as u32, k as u32, nx, 0, 0, tpr, x_off, 0]);
        // x_off exists only in matvec4: for F32/BF16 the engine's kernel (matvec.wgsl) would ignore it
        let p = if e == "mv_f32" { &self.mv4_f32 } else { &self.mv[e] };
        self.gpu.command(p, &[(0, &self.w[name]), (1, self.a(x)), (2, self.a(y)), (3, &self.local_ids), (4, &u)], (nx, ny, 1), short(name), t.nbytes)
    }
    /// M4: output head for the DRAFTS: the reduced one if present (head_k rows, Q6_K, buffer "testa_ridotta"), otherwise
    /// `output.weight`; x = xn from `x_off`; logits in "logits"; then argmax (am2[0] = index, to be remapped if reduced).
    fn draft_head(&self, c: &mut Vec<Command>, x_off: u32) {
        let a = |n: &str| self.a(n);
        if self.head_k > 0 {
            let (rows, k) = (self.head_k, 2048usize);
            let (tpr, nx, ny) = mv_piece_geometry(rows, k, 32);
            let u = self.u([rows as u32, k as u32, nx, 0, 0, tpr, x_off, 0]);
            c.push(self.gpu.command(&self.mv["mv_q6_k"], &[(0, &self.w["testa_ridotta"]), (1, a("xn")), (2, a("logits")), (3, &self.local_ids), (4, &u)], (nx, ny, 1), "reduced head".into(), (rows * 1680) as u64));
        } else {
            c.push(self.mv_x("output.weight", "xn", "logits", x_off));
        }
        let n = if self.head_k > 0 { self.head_k as u32 } else { VOCAB as u32 };
        let g = n.div_ceil(256);
        let u_am1 = self.u([n, 0, 0, 0, 0, 0, 0, 0]);
        let u_am2 = self.u([g, 0, 0, 0, 0, 0, 0, 0]);
        c.push(self.op("argmax1", &[(0, a("logits"))], a("am_v"), Some(a("am_i")), &u_am1, (g, 1, 1)));
        c.push(self.op("argmax2", &[(0, a("am_v")), (1, a("am_i"))], a("am"), Some(a("am2")), &u_am2, (1, 1, 1)));
    }
    /// M4: expert matvec in union for the double step (see matvec_u.wgsl). x_tok/y_tok = stride per token.
    fn mv_u(&self, name: &str, x: &str, y: &str, x_step: u32, x_tok: u32) -> Command { self.mv_u_(name, x, y, x_step, x_tok, false) }
    fn mv_uh(&self, name: &str, x: &str, y: &str, x_step: u32, x_tok: u32) -> Command { self.mv_u_(name, x, y, x_step, x_tok, true) }
    fn mv_u_(&self, name: &str, x: &str, y: &str, x_step: u32, x_tok: u32, f16: bool) -> Command {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}"));
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        let e = match t.ttype { TType::Q4_K => "mv_q4_k", TType::Q6_K => "mv_q6_k", TType::Q3_K => "mv_q3_k", TType::Q8_0 => "mv_q8_0", x => panic!("{name}: {x:?}") };
        let (tpr, nx, ny) = mv_piece_geometry(rows, k, 32);
        let u = self.u([rows as u32, k as u32, nx, 8 * rows as u32, x_step, tpr, 2, x_tok]);
        let p = if f16 { &self.mvuh[e] } else { &self.mvu[e] };
        // bytes: ~14 experts of 16 on average (2.2 in common): only for the GB/s of the profile
        self.gpu.command(p, &[(0, &self.w[name]), (1, self.a(x)), (2, self.a(y)), (3, self.a("uniforms")), (4, &u)], (nx, ny, 16), format!("experts {} ({e}) union{}", short(name), if f16 { " h" } else { "" }), 14 * t.nbytes / 256)
    }
    /// M4: expert matvec for token `tok` of the double step: x of the token at `x_off`, output at `y_off`, experts from `ids`.
    fn mv_tok(&self, name: &str, x: &str, y: &str, x_step: u32, x_off: u32, y_off: u32, ids: &str) -> Command {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}"));
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        let e = match t.ttype { TType::Q4_K => "mv_q4_k", TType::Q6_K => "mv_q6_k", TType::Q3_K => "mv_q3_k", TType::Q8_0 => "mv_q8_0", x => panic!("{name}: {x:?}") };
        let (tpr, nx, ny) = mv_piece_geometry(rows, k, 32);
        let u = self.u([rows as u32, k as u32, nx, 1, x_step, tpr, x_off, y_off]);
        self.gpu.command(&self.mv[e], &[(0, &self.w[name]), (1, self.a(x)), (2, self.a(y)), (3, self.a(ids)), (4, &u)], (nx, ny, 8), format!("experts {} ({e}) tok", short(name)), 8 * t.nbytes / 256)
    }
    /// y = W x with the weight `name`. Experts (`_exps`): 8 matrices chosen by `ids` (written by the routing).
    fn mv(&self, name: &str, x: &str, y: &str, x_step: u32) -> Command {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}"));
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        let experts = name.contains("_exps");
        let mut e = match t.ttype { TType::Q8_0 => "mv_q8_0", TType::Q4_K => "mv_q4_k", TType::Q6_K => "mv_q6_k", TType::Q3_K => "mv_q3_k", TType::F32 | TType::BF16 => "mv_f32", x => panic!("{x:?}") };
        let mut piece = 32;
        if t.ttype == TType::Q4_K && q4k_64() { e = "mv_q4_k64"; piece = 64; }
        let (tpr, nx, ny) = mv_piece_geometry(rows, k, piece);
        let u = self.u([rows as u32, k as u32, nx, experts as u32, x_step, tpr, 0, 0]);
        let ids = if experts { self.a("ids") } else { &self.local_ids };
        let nb = if t.ttype == TType::BF16 { 2 * t.nbytes } else { t.nbytes };   // BF16 is converted to F32
        let (nz, byte) = if experts { (8, 8 * nb / 256) } else { (1, nb) };
        let nm = if experts { format!("experts {} ({e})", short(name)) } else { short(name) };
        self.gpu.command(&self.mv[e], &[(0, &self.w[name]), (1, self.a(x)), (2, self.a(y)), (3, ids), (4, &u)], (nx, ny, nz), nm, byte)
    }
}

impl Engine {
    pub fn new(path: &str) -> Self { Self::new_with_reserve(path, None) }
    /// As `new`, with the VRAM reserve during weight loading given by the caller (None = MECHTA_RESERVE_GB, default 0).
    pub fn new_with_reserve(path: &str, reserve: Option<u64>) -> Self {
        let g = Gguf::open(path).expect("GGUF");
        // MemoryUsage: with the default (Performance) a 150 MiB buffer takes a 256 MiB block
        // (measured with `mechta memtest`): 32 GB for the experts alone instead of 19.
        let gpu = Gpu::with_memory(wgpu::MemoryHints::MemoryUsage);
        let src_mv = if kernel_m1() { include_str!("wgsl/m1/matvec.wgsl") } else { include_str!("wgsl/matvec.wgsl") };
        let src_op_s = ops_source(kv_format()); let src_op = src_op_s.as_str();
        // Matvec: Q8_0, Q4_K and Q6_K from matvec4.wgsl (x as vec4; Q4_K: d1 (q . x) - m1 sum x, 46 → 62 GB/s
        // in the engine; Q8_0 and Q6_K: realigned words + unpack4xI8/U8, at the read ceiling of 81-83 GB/s on the bench).
        // F32 from matvec.wgsl. MECHTA_MATVEC, only for comparisons: base = everything from matvec.wgsl; q4k3 = Q4_K from matvec3
        // and the rest from matvec; q8 = Q8_0 and Q4_K from matvec4, Q6_K from matvec (the default before the new Q6_K);
        // mixed = Q8_0 and Q4_K from matvec2 (4-byte words; it is not faster).
        let choice = std::env::var("MECHTA_MATVEC").unwrap_or_default();
        let src_mv2 = include_str!("wgsl/matvec2.wgsl");
        let src_mv3 = include_str!("wgsl/matvec3.wgsl");
        let src_mv4_s = q3_source("matvec4"); let src_mv4 = src_mv4_s.as_str();
        let mv = ["mv_q8_0", "mv_q4_k", "mv_q6_k", "mv_f32", "mv_q4_k64", "mv_q3_k"].into_iter().map(|e| {
            let src = match choice.as_str() {
                _ if e == "mv_q4_k64" || e == "mv_q3_k" => src_mv4,
                "base" => src_mv,
                "q4k3" if e == "mv_q4_k" => src_mv3,
                "q4k3" => src_mv,
                "mixed" if e == "mv_q8_0" || e == "mv_q4_k" => src_mv2,
                "q8" if e == "mv_q8_0" || e == "mv_q4_k" => src_mv4,
                "q8" => src_mv,
                _ if e != "mv_f32" && !kernel_m1() => src_mv4,
                _ => src_mv,
            };
            (e, gpu.pipeline(src, e))
        }).collect();
        if !choice.is_empty() { eprintln!("matvec: MECHTA_MATVEC={choice}"); }
        // MECHTA_DELTA=v1: the old delta (state as ggml, two passes), only for comparisons.
        let delta_v1 = std::env::var("MECHTA_DELTA").map(|v| v == "v1").unwrap_or(false);
        if delta_v1 { eprintln!("delta: MECHTA_DELTA=v1"); }
        let op = ["rmsnorm", "conv", "beta_gate", "delta", "sum", "swiglu", "moe_sum", "rope", "attn", "attn_gate", "copy", "argmax1", "argmax2",
                  "beta_gate2", "conv2", "delta2", "rope2", "attn2", "swiglu16", "attn_gate16", "attn_v2", "attn2_v2", "attn_v3", "attn2_v3", "attn_fd", "attn_fd_combine", "conv_n", "delta_n", "attn_fd4", "attn_fa", "attn_fd2", "attn_fd2b", "attn_fd2c", "attn_fd2d", "attn_fd2e", "attn_fd2_read", "attn_fd2_read2", "attn_fd2f", "attn_fd2g", "attn_fd2cm", "attn_fd2h", "attn_fd2i"].into_iter()
            .chain(if matches!(kv_format(), KvFormat::F32 | KvFormat::F16Sim) { vec![] } else { vec!["kv_rope", "kv_rope2", "kv_copy"] })
            .map(|e| (e, gpu.pipeline(src_op, if e == "delta" && delta_v1 { "delta1" } else { e }))).collect::<HashMap<_, _>>();
        let mut op = op;
        if prefill_mx() {
            assert!(kv_format() == KvFormat::F16, "MECHTA_PREFILL_ATTN=mx needs MECHTA_KV=f16");
            let src_mx = fa_mx_source();
            for e in [mx_entry(), "q_f16"] { op.insert(e, gpu.pipeline(&src_mx, e)); }
            eprintln!("prefill attention: {} (matrix units)", mx_entry());
        }
        let rt = gpu.pipeline(if kernel_m1() { include_str!("wgsl/m1/routing.wgsl") } else { include_str!("wgsl/routing.wgsl") }, "routing");
        let rt2 = gpu.pipeline(if kernel_m1() { include_str!("wgsl/m1/routing.wgsl") } else { include_str!("wgsl/routing.wgsl") }, "routing_b");
        let src_mvu = q3_source("matvec_u");
        let mvu = ["mv_q4_k", "mv_q6_k", "mv_q3_k", "mv_q8_0"].into_iter().map(|e| (e, gpu.pipeline(&src_mvu, e))).collect();
        let un = gpu.pipeline(include_str!("wgsl/merge.wgsl"), "merge");
        // M3: prefill GEMM, per-token routing, grouping by expert
        let src_gm_q3 = q3_source("gemm"); let src_gm = src_gm_q3.as_str();
        let gm = ["gemm_q8_0", "gemm_q4_k", "gemm_q6_k", "gemm_f32", "gemm8_q8_0", "gemm8_q4_k", "gemm8_q6_k", "gemm16_q8_0", "gemm16_q4_k", "gemm16_q6_k", "gemmr_q8_0", "gemmr_q4_k", "gemmr_q6_k", "gemmr5_q8_0", "gemmr5_q4_k", "gemmr5_q6_k", "gemm2_q8_0", "gemm2_q4_k", "gemm2_q6_k", "gemm2_f32", "gemmr5_q3_k"].into_iter().map(|e| (e, gpu.pipeline(src_gm, e))).collect();
        let rtn = gpu.pipeline(include_str!("wgsl/routing.wgsl"), "routing_n");
        let rg = gpu.pipeline(include_str!("wgsl/routing.wgsl"), "gather");
        let mv4_f32 = gpu.pipeline(src_mv4, "mv_f32");
        let src_mvh = include_str!("wgsl/matvec_nh.wgsl");
        let mvh = ["mv_q8_0", "mv_q4_k", "mv_q6_k", "mv_f32"].into_iter().map(|e| (e, gpu.pipeline(src_mvh, e))).collect();
        let src_mvuh = q3_source("matvec_uh");
        let mvuh = ["mv_q4_k", "mv_q6_k", "mv_q3_k", "mv_q8_0"].into_iter().map(|e| (e, gpu.pipeline(&src_mvuh, e))).collect();
        let src_mvn = include_str!("wgsl/matvec_n4.wgsl");
        let mvn = ["mv_q8_0", "mv_q4_k", "mv_q6_k", "mv_f32"].into_iter().map(|e| (e, gpu.pipeline(src_mvn, e))).collect();
        let (emb_e, emb_row, emb_bpr) = emb_format(&g);
        let dq = gpu.pipeline(include_str!("wgsl/dequant.wgsl"), emb_e);
        // Trial with 16 GB of VRAM. Normal order: first the buffers the GPU WRITES (states, KV cache,
        // outputs, ~0.11 GB), while the VRAM is still empty, so they end up in VRAM; then, with MECHTA_RESERVE_GB=N,
        // a reserve of N GB during weight loading, freed afterwards. With MECHTA_ORDER=weights-first the old
        // order: weights first, written buffers after (with VRAM full they end up in GTT). Used to separate the two causes.
        let create_buffer = || {
        let mut a = HashMap::new();
        // sizes for TWO tokens (M4, double step): token t is at t * single size
        for (n, k) in [("x", 2 * 2048), ("x2", 2 * 2048), ("h", 2 * 2048), ("qkv", 2 * 8192), ("z", 2 * 4096), ("braw", 64), ("araw", 64), ("bg", 128),
                       ("cv", 2 * 8192), ("qk", 2 * 4096), ("ao", 2 * 4096), ("y", 2 * 4096), ("att_out", 2 * 2048), ("qf", 2 * 8192), ("kb", 2 * 512), ("vb", 2 * 512),
                       ("qn", 2 * 4096), ("kn", 2 * 512), ("qr", 2 * 4096), ("att", 2 * 4096), ("ag", 2 * 4096), ("h2", 2 * 2048), ("lg", 512), ("probs", 512),
                       ("ids", 8), ("ids2", 8), ("weights", 16), ("eg", 2 * 4096), ("eu", 2 * 4096), ("esw", 2 * 4096), ("ed", 2 * 16384), ("sg", 2 * 512), ("su", 2 * 512),
                       ("ssw", 2 * 512), ("sd", 2 * 2048), ("sgt", 2), ("fo", 2 * 2048), ("xn", 2 * 2048), ("logits", 2 * VOCAB), ("emb_raw", 2 * 544),
                       ("am_v", 2 * VOCAB / 256), ("am_i", 2 * VOCAB / 256), ("am", 2), ("am2", 2),
                       ("m_emb_raw", 544), ("m_e", 2 * 2048), ("m_cat", 2 * 4096),   // MTP layer (M4)
                       ("uniforms", 1 + 3 * 16), ("xn16", 2 * 2048),                 // union of the experts; xn in f16 (double step)
                       ("att_p", 2 * 16 * (ctx() / 128) * 258)] {                  // flash-decoding partials (2 tokens)
            a.insert(n, gpu.empty_buffer(n, k as u64 * 4));
        }
        // M3: buffers of the prefill in blocks of NB tokens (same names, separate map)
        let mut ap = HashMap::new();
        for (n, k) in [("x", NB * 2048), ("x2", NB * 2048), ("h", NB * 2048), ("h2", NB * 2048), ("att_out", NB * 2048), ("fo", NB * 2048),
                       ("qkv", NB * 8192), ("z", NB * 4096), ("braw", NB * 32), ("araw", NB * 32), ("bg", NB * 64), ("cv", NB * 8192), ("qk", NB * 4096),
                       ("ao", NB * 4096), ("y", NB * 4096), ("qf", NB * 8192), ("kb", NB * 512), ("vb", NB * 512), ("kn", NB * 512), ("qn", NB * 4096),
                       ("qr", NB * 4096), ("qh", NB * 2048), ("att", NB * 4096), ("ag", NB * 4096), ("lg", NB * 256), ("probs", NB * 256), ("ids", NB * 8), ("weights", NB * 8),
                       ("grp", 512 + NB * 8), ("eg", NB * 8 * 512), ("eu", NB * 8 * 512), ("esw", NB * 8 * 512), ("ed", NB * 8 * 2048),
                       ("sg", NB * 512), ("su", NB * 512), ("ssw", NB * 512), ("sd", NB * 2048), ("sgt", NB), ("emb_raw", NB * 544),
                       ("att_p", (8 * 16 * (ctx() / prefill_block())).max(64 * 16 * (ctx() / FA_CH)) * 258)] {   // attention in lots of 8 tokens (attn_fd4: blocks of prefill_block())
            ap.insert(n, gpu.empty_buffer(n, k as u64 * 4));
        }
        let tab_n = gpu.empty_buffer("rope-n", (NB * 64 * 4) as u64);
        // uniforms rewritten at every block: conv_n, delta_n, routing/gather; per attention layer: rope K, copy V, attention in lots of 8
        let u_n = [gpu.uniform(&[0u32; 8]), gpu.uniform(&[0u32; 8]), gpu.uniform(&[0u32; 4])];
        let u_pos_n: Vec<Option<(wgpu::Buffer, wgpu::Buffer, Vec<wgpu::Buffer>)>> = (0..N_LAYER).map(|il| if attention(il) {
            Some((gpu.uniform(&[0u32; 8]), gpu.uniform(&[0u32; 8]), (0..NB / 8 + NB / 64 * fa_splits()).map(|_| gpu.uniform(&[0u32; 8])).collect()))
        } else { None }).collect();
        let tab = gpu.empty_buffer("rope", 64 * 4);
        let (mut conv, mut ssm, mut kc, mut vc, mut u_pos) = (vec![], vec![], vec![], vec![], vec![]);
        // the MTP layer (index N_LAYER) is an attention block with its own KV cache and its own RoPE table (position + 1)
        let m_tab = gpu.empty_buffer("rope-mtp", 64 * 4);
        // double step (M4): RoPE tables of the 2 tokens, position uniforms for 2 tokens, DeltaNet state after the first token
        let tab2 = gpu.empty_buffer("rope2", 128 * 4);
        let (mut ssm1, mut conv1, mut u_pos2) = (vec![], vec![], vec![]);
        for il in 0..=N_LAYER {
            if attention(il) || il == N_LAYER {
                ssm1.push(gpu.empty_buffer("-", 4)); conv1.push(gpu.empty_buffer("-", 4));
                u_pos2.push(Some([gpu.uniform(&[0; 8]), gpu.uniform(&[0; 8]), gpu.uniform(&[0; 8])]));
            } else {
                ssm1.push(gpu.empty_buffer("ssm1", 32 * 128 * 128 * 4)); conv1.push(gpu.empty_buffer("conv1", 24576 * 4)); u_pos2.push(None);
            }
            if attention(il) || il == N_LAYER {
                kc.push(Some(gpu.empty_buffer("kc", kv_format().buffer_bytes()))); vc.push(Some(gpu.empty_buffer("vc", kv_format().buffer_bytes())));
                conv.push([gpu.empty_buffer("-", 4), gpu.empty_buffer("-", 4)]); ssm.push(gpu.empty_buffer("-", 4));
                u_pos.push(Some([gpu.uniform(&[0; 8]), gpu.uniform(&[0; 8]), gpu.uniform(&[0; 8])]));
            } else {
                conv.push([gpu.empty_buffer("conv", 24576 * 4), gpu.empty_buffer("conv", 24576 * 4)]);
                ssm.push(gpu.empty_buffer("ssm", 32 * 128 * 128 * 4));
                kc.push(None); vc.push(None); u_pos.push(None);
            }
        }
            (a, tab, m_tab, conv, ssm, kc, vc, u_pos, tab2, ssm1, conv1, u_pos2, ap, tab_n, u_n, u_pos_n)
        };
        let with_mtp = std::env::var("MECHTA_MTP").map(|v| v != "no").unwrap_or(true);
        // MECHTA_DRAFT_HEAD=frequencies.tsv (id<TAB>count): reduced output head for the DRAFTS, with only the rows of those tokens
        // (M4: the outputs of the sessions use 32.6K tokens of 248K; 417 MB → 55 MB per draft). The argmax of the
        // draft is on the reduced index and is remapped onto the true id. The verification always uses the full head.
        // Default (+5-10% with MTP, same acceptance): engine/draft-head.tsv.
        // MECHTA_DRAFT_HEAD=no for the full head in the drafts too.
        let head_file = match std::env::var("MECHTA_DRAFT_HEAD") {
            Ok(v) if v == "no" => None,
            Ok(v) => Some(v),
            Err(_) => Some(concat!(env!("CARGO_MANIFEST_DIR"), "/../../draft-head.tsv").to_string()),
        };
        let head_ids: Vec<u32> = head_file.map(|f| {
            let mut v: Vec<u32> = std::fs::read_to_string(&f).expect("MECHTA_DRAFT_HEAD").lines().filter_map(|l| l.split('\t').next()?.parse().ok()).collect();
            v.sort_unstable(); v.dedup(); v
        }).unwrap_or_default();
        let load_weights = |base: (f64, f64)| {
        let mut w = HashMap::new(); let mut byte_gpu = 0u64;
                for t in g.tensors.clone() {
            if t.name == "token_embd.weight" { continue; }
            if t.name.starts_with("blk.40.") && !with_mtp { continue; }   // the MTP layer (M4)
            let mut v = Vec::new(); g.read(&t, 0, t.nbytes, &mut v).expect("read");
            // BF16 (router of the MTP layer): converted to F32 at load time, the kernel reads it as F32
            if t.ttype == TType::BF16 { v = v.chunks_exact(2).flat_map(|c| (u32::from(u16::from_le_bytes([c[0], c[1]])) << 16).to_le_bytes()).collect(); }
            byte_gpu += v.len() as u64;
            w.insert(t.name.clone(), gpu.buffer_byte(&t.name, &v));
            // the temporary copy of the data is freed only after a submit and a wait: right away, tensor by tensor
            gpu.queue.submit([]);
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            // safety: GPU memory must not grow more than the weights + 3 GB
            let (v1, g1) = crate::profile::gpu_memory();
            let grown = (v1 - base.0) + (g1 - base.1);
            assert!(grown < byte_gpu as f64 / 1e9 + 3.0, "GPU memory grew {grown:.2} GB for {:.2} GB of weights: stopping", byte_gpu as f64 / 1e9);
        }
            if !head_ids.is_empty() {
                let t = g.tensor("output.weight").expect("output.weight");
                assert_eq!(t.ttype, TType::Q6_K); let row = 2048 / 256 * 210;   // 8 Q6_K blocks of 210 bytes
                let mut all = Vec::new(); g.read(&t, 0, t.nbytes, &mut all).expect("read");
                let mut v = Vec::with_capacity(head_ids.len() * row);
                for &id in &head_ids { v.extend_from_slice(&all[id as usize * row..(id as usize + 1) * row]); }
                byte_gpu += v.len() as u64;
                w.insert("testa_ridotta".to_string(), gpu.buffer_byte("testa_ridotta", &v));
                gpu.queue.submit([]); gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
                eprintln!("reduced head for the drafts: {} rows of {VOCAB} ({:.0} MB)", head_ids.len(), v.len() as f64 / 1e6);
            }
            (w, byte_gpu)
        };
        let weights_before = std::env::var("MECHTA_ORDER").map(|v| v == "weights-first").unwrap_or(false);
        let mem0 = crate::profile::gpu_memory();
        let ((a, tab, m_tab, conv, ssm, kc, vc, u_pos, tab2, ssm1, conv1, u_pos2, ap, tab_n, u_n, u_pos_n), (w, byte_gpu)) = if weights_before {
            let (w, byte_gpu) = load_weights(mem0);
            let mem1 = crate::profile::gpu_memory();
            eprintln!("order pesi-prima · weights {:.2} GB: VRAM +{:.2} GTT +{:.2} GB", byte_gpu as f64 / 1e9, mem1.0 - mem0.0, mem1.1 - mem0.1);
            let b = create_buffer();
            let mem2 = crate::profile::gpu_memory();
            eprintln!("buffers written by the GPU: VRAM +{:.3} GTT +{:.3} GB", mem2.0 - mem1.0, mem2.1 - mem1.1);
            (b, (w, byte_gpu))
        } else {
            let b = create_buffer();
            let mem1 = crate::profile::gpu_memory();
            eprintln!("buffers written by the GPU: VRAM +{:.3} GTT +{:.3} GB", mem1.0 - mem0.0, mem1.1 - mem0.1);
            let reserve_gb: u64 = reserve.unwrap_or_else(|| std::env::var("MECHTA_RESERVE_GB").ok().and_then(|v| v.parse().ok()).unwrap_or(0));
            let reserve: Vec<wgpu::Buffer> = (0..reserve_gb * 4).map(|_| gpu.empty_buffer("reserve", 256 << 20)).collect();
            let mem2 = crate::profile::gpu_memory();
            if reserve_gb > 0 { eprintln!("reserve {reserve_gb} GB: VRAM +{:.2} GTT +{:.2} GB", mem2.0 - mem1.0, mem2.1 - mem1.1); }
            let (w, byte_gpu) = load_weights(mem2);
            let mem3 = crate::profile::gpu_memory();
            eprintln!("weights {:.2} GB: VRAM +{:.2} GTT +{:.2} GB", byte_gpu as f64 / 1e9, mem3.0 - mem2.0, mem3.1 - mem2.1);
            drop(reserve);
            gpu.queue.submit([]); gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            let mem4 = crate::profile::gpu_memory();
            if reserve_gb > 0 { eprintln!("reserve released: VRAM in use {:.2} GB, GTT in use {:.2} GB", mem4.0, mem4.1); }
            (b, (w, byte_gpu))
        };
        let empty = gpu.empty_buffer("vuoto", 4);
        let local_ids = gpu.buffer_u32("id_locali", &[0, 1, 2, 3, 4, 5, 6, 7]);
        let (prog, end_layer, pm, fm, prog2, end2, pm2, prog_n, end_n) = {
            let of = Factory { gpu: &gpu, g: &g, mv, op, mvn, rt2, mvu, un, mvh, mvuh, mv4_f32, head_k: head_ids.len(), rt, dq, emb_bpr, w, a: &a, empty, local_ids, gm, rtn, rg };
            let (p0, f) = program(&of, 0, &conv, &ssm, &kc, &vc, &u_pos, &tab);
            let (p1, _) = program(&of, 1, &conv, &ssm, &kc, &vc, &u_pos, &tab);
            let (pm, fm) = if with_mtp { program_mtp(&of, &conv, &ssm, &kc, &vc, &u_pos, &m_tab) } else { (vec![], vec![]) };
            let (d0, f2) = program2(&of, 0, &conv, &conv1, &ssm, &ssm1, &kc, &vc, &u_pos2, &tab2);
            let (d1, _) = program2(&of, 1, &conv, &conv1, &ssm, &ssm1, &kc, &vc, &u_pos2, &tab2);
            let pm2 = if with_mtp { program_mtp2(&of, &conv, &conv1, &ssm, &ssm1, &kc, &vc, &u_pos2, &tab2) } else { vec![] };
            // M3: the prefill program uses the `ap` buffers (same names): the same factory with the map swapped
            let ofn = Factory { a: &ap, ..of };
            // M3: one program per block size and conv parity, and its segments per layer (`end_n[ib]`). The segments
            // change with the size: the attention puts nb/8 dispatches per layer. Before, `end_n` was a single one (the one
            // of the last size, 8 tokens) and in the 512 blocks the tail of the layers ended up in a 1-3 s submit: GPU blocked
            // for the other programs (mouse and browser frozen).
            let mut prog_n = vec![]; let mut fn_ = vec![];
            for nb in BLOCKS {
                let (n0, f0) = program_n(&ofn, 0, &conv, &ssm, &kc, &vc, &u_n, &u_pos_n, &tab_n, nb);
                let (n1, _) = program_n(&ofn, 1, &conv, &ssm, &kc, &vc, &u_n, &u_pos_n, &tab_n, nb);
                prog_n.push([n0, n1]); fn_.push(f0);
            }
            ([p0, p1], f, pm, fm, [d0, d1], f2, pm2, prog_n, fn_)
        };
        Engine { g, gpu, emb_row, a, conv, ssm, u_pos, tab, prog, end_layer, byte_gpu, prog_mtp: pm, end_mtp: fm, m_tab, prog2, end2, ssm1, conv1, tab2, u_pos2, prog_mtp2: pm2,
                 ap, tab_n, u_n, u_pos_n, cuts_n: (0..BLOCKS.len()).map(|ib| attn_cuts(&prog_n[ib][0], &end_n[ib])).collect(), prog_n, end_n, last_par: std::cell::Cell::new(0), head_ids, par: std::cell::Cell::new(0), mtp_start: std::cell::Cell::new(1), photo: RefCell::new(vec![]), kv_buf: (kc, vc), prof: RefCell::new(None), span: RefCell::new(None) }
    }

    /// Resets the states (conv, DeltaNet) for a new sequence. The KV cache gets overwritten.
    pub fn reset(&self) {
        self.par.set(0);
        for il in 0..N_LAYER { if !attention(il) {
            for b in &self.conv[il] { self.gpu.write(b, 0, &vec![0u8; 24576 * 4]); }
            self.gpu.write(&self.ssm[il], 0, &vec![0u8; 32 * 128 * 128 * 4]);
        } }
    }

    /// One token at position `pos`. Returns the logits. With `l_out`, x is read at the end of every layer
    /// (one submit per layer: only for the verifications).
    pub fn token(&self, tok: u32, pos: usize, l_out: Option<&mut dyn FnMut(usize, Vec<f32>)>) -> Vec<f32> {
        assert!(pos < ctx());
        self.inputs(tok, pos);
        let prog = &self.prog[self.par.get()]; self.par.set(1 - self.par.get());
        match l_out {
            None => self.run(prog),
            Some(f) => {
                let mut s = 0;
                for il in 0..N_LAYER {
                    let e = self.end_layer[il];
                    self.run(&prog[s..e]);
                    f(il, self.gpu.read_f32(&self.a["x"], 2048));
                    s = e;
                }
                self.run(&prog[s..]);
            }
        }
        let tc = Instant::now();
        let v = self.gpu.read_f32(&self.a["logits"], VOCAB);
        self.cpu("logit read (1 MB)", tc);
        v
    }

    /// The inputs of one token: embedding row (from the file, in page cache), RoPE table, position uniforms.
    fn inputs(&self, tok: u32, pos: usize) {
        let tc = Instant::now();
        let te = self.g.tensor("token_embd.weight").unwrap();
        let mut row = Vec::new(); self.g.read(te, tok as u64 * self.emb_row, self.emb_row, &mut row).unwrap();
        self.gpu.write(&self.a["emb_raw"], 0, &row);
        let tab: Vec<u8> = rope_table(pos as i32).iter().flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.write(&self.tab, 0, &tab);
        let ub = |v: [u32; 8]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        for u in self.u_pos[..N_LAYER].iter().flatten() {
            self.gpu.write(&u[0], 0, &ub([256, 2, 0, 0, 256, 256, 0, (pos * 512) as u32]));
            self.gpu.write(&u[1], 0, &ub([512, 0, 0, 0, 0, 0, 0, (pos * 512) as u32]));
            self.gpu.write(&u[2], 0, &ub([pos as u32 + 1, (ctx() / 128) as u32, fb(1.0 / 16.0), fd_phase(), 0, 0, 0, 0]));
        }
        self.cpu("inputs (embedding, RoPE, uniform)", tc);
    }

    /// Inputs of an MTP draft at position `pos` (= position of token `tok`, which is t_{i+1}): embedding row,
    /// RoPE table of `pos`, position uniforms of layer 40 (attention from position 1: the MTP cache does not have 0).
    fn inputs_mtp(&self, tok: u32, pos: usize) {
        let te = self.g.tensor("token_embd.weight").unwrap();
        let mut row = Vec::new(); self.g.read(te, tok as u64 * self.emb_row, self.emb_row, &mut row).unwrap();
        self.gpu.write(&self.a["m_emb_raw"], 0, &row);
        let tab: Vec<u8> = rope_table(pos as i32).iter().flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.write(&self.m_tab, 0, &tab);
        let ub = |v: [u32; 8]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let u = self.u_pos[N_LAYER].as_ref().expect("MTP layer not loaded");
        self.gpu.write(&u[0], 0, &ub([256, 2, 0, 0, 256, 256, 0, (pos * 512) as u32]));
        self.gpu.write(&u[1], 0, &ub([512, 0, 0, 0, 0, 0, 0, (pos * 512) as u32]));
        self.gpu.write(&u[2], 0, &ub([pos as u32 + 1, (ctx() / 128) as u32, fb(1.0 / 16.0), fd_phase(), 0, 0, 0, self.mtp_start.get()]));
    }

    /// An MTP draft for the verifications: h = h_nextn (2048 f32) written into "xn", then the segmented program; after each
    /// segment `f(segment)` can read the buffers ("x", "x2", "xn", "logits" with `read`). Returns the GPU argmax.
    pub fn draft_verify(&self, h: &[f32], tok: u32, pos: usize, f: &mut dyn FnMut(usize)) -> u32 {
        assert_eq!(h.len(), 2048);
        self.gpu.write(&self.a["xn"], 0, &h.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        self.inputs_mtp(tok, pos);
        let mut s = 0;
        for (k, &e) in self.end_mtp.iter().enumerate() { self.gpu.run_commands(&self.prog_mtp[s..e]); f(k); s = e; }
        self.gpu.run_commands(&self.prog_mtp[s..]);
        self.argmax_gpu().0
    }

    /// Inputs of the double step: embedding rows of the 2 tokens (in a row in emb_raw), RoPE tables of pos and pos+1,
    /// position uniforms for 2 tokens (K/V written at pos and pos+1; attention up to pos+tok).
    fn inputs2(&self, tok0: u32, tok1: u32, pos: usize) {
        let te = self.g.tensor("token_embd.weight").unwrap();
        let mut row = Vec::new(); self.g.read(te, tok0 as u64 * self.emb_row, self.emb_row, &mut row).unwrap();
        let mut row1 = Vec::new(); self.g.read(te, tok1 as u64 * self.emb_row, self.emb_row, &mut row1).unwrap();
        row.extend_from_slice(&row1);
        self.gpu.write(&self.a["emb_raw"], 0, &row);
        let mut tab: Vec<u8> = rope_table(pos as i32).iter().flat_map(|v| v.to_le_bytes()).collect();
        tab.extend(rope_table(pos as i32 + 1).iter().flat_map(|v| v.to_le_bytes()));
        self.gpu.write(&self.tab2, 0, &tab);
        let ub = |v: [u32; 8]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        for u in self.u_pos2[..N_LAYER].iter().flatten() {
            self.gpu.write(&u[0], 0, &ub([256, 4, 0, 0, 256, 256, 2, (pos * 512) as u32]));      // rope2 of K: 2 rows (KV heads) per token
            self.gpu.write(&u[1], 0, &ub([1024, 0, 0, 0, 0, 0, 0, (pos * 512) as u32]));
            self.gpu.write(&u[2], 0, &ub([pos as u32 + 1, (ctx() / 128) as u32, fb(1.0 / 16.0), fd_phase(), 0, 0, attn2_flag(), 0]));
        }
    }

    /// Double step for the verifications: token tok0 at `pos` and tok1 at `pos + 1`, segmented program (one per layer):
    /// after layer `il` it calls `f(il)`, which can read "x" = [l_out tok0 | l_out tok1]. Returns the argmax of the two tokens.
    pub fn step2_verify(&self, tok0: u32, tok1: u32, pos: usize, f: &mut dyn FnMut(usize)) -> (u32, u32) {
        self.inputs2(tok0, tok1, pos);
        let prog = &self.prog2[self.par.get()]; self.par.set(1 - self.par.get());
        let mut s = 0;
        for (il, &e) in self.end2.iter().enumerate() { self.gpu.run_commands(&prog[s..e]); f(il); s = e; }
        self.gpu.run_commands(&prog[s..]);
        let i = self.gpu.read_f32(&self.a["am2"], 2);
        (i[0].to_bits(), i[1].to_bits())
    }

    /// Fast double step (M4): one submit, reading back the two argmax (8 bytes) in the same command buffer.
    /// Records into `span` if active. Returns (argmax of token 0 = candidate for pos+1, argmax of token 1 = for pos+2).
    pub fn step2(&self, tok0: u32, tok1: u32, pos: usize) -> (u32, u32) {
        assert!(pos + 1 < ctx());
        self.inputs2(tok0, tok1, pos);
        let par = self.par.get(); self.par.set(1 - par); self.last_par.set(par);
        if self.prof.borrow().is_some() { let b = self.run_profiled(&self.prog2[par], "double", 8); return (b[0], b[1]); }
        let (prep, wait, gpu, b) = self.gpu.run_commands_read(&self.prog2[par], &self.a["am2"], 8);
        if let Some(v) = self.span.borrow_mut().as_mut() { v.push((prep, wait, gpu)); }
        (u32::from_le_bytes(b[..4].try_into().unwrap()), u32::from_le_bytes(b[4..8].try_into().unwrap()))
    }

    /// Double step with the timestamps of every dispatch (slow: one compute pass per dispatch). Returns (name, ms, bytes) per dispatch.
    pub fn step2_measured(&self, tok0: u32, tok1: u32, pos: usize) -> Vec<(String, f64, u64)> {
        self.inputs2(tok0, tok1, pos);
        let par = self.par.get(); self.par.set(1 - par);
        let prog = &self.prog2[par];
        let m = self.gpu.run_commands_measured(prog);
        prog.iter().zip(&m.gpu_ms).map(|(c, t)| (c.n.clone(), *t, c.byte)).collect()
    }

    /// MTP draft (one token) with h = "xn" as it is (the state after output_norm left by the last single step, or of
    /// token 0 of the last double step): token `tok` at position `pos`. One submit, 4 bytes read. Returns the draft.
    pub fn draft(&self, tok: u32, pos: usize) -> u32 {
        self.inputs_mtp(tok, pos);
        let (_, _, _, b) = self.gpu.run_commands_read(&self.prog_mtp, &self.a["am2"], 4);
        self.draft_id(u32::from_le_bytes(b[..4].try_into().unwrap()))
    }
    /// Speed path 4 (measurement): a second, chained draft. The MTP block output of the last draft ("x", row
    /// `row`: 0 after `draft`, 1 after `draft2`) becomes the hidden state of a new MTP step on token `tok` at `pos`.
    /// Its MTP cache entry at `pos` is speculative, but always rewritten (by `draft` / `draft2`) before it is read.
    pub fn draft_chain(&self, tok: u32, pos: usize, row: usize) -> u32 {
        let src = &self.a["x"]; let dst = &self.a["xn"];
        let mut enc = self.gpu.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(src, (row * 2048 * 4) as u64, dst, 0, 2048 * 4);
        self.gpu.queue.submit([enc.finish()]);
        self.draft(tok, pos)
    }
    /// Index of the draft argmax → token id (reduced head: remaps).
    fn draft_id(&self, i: u32) -> u32 { if self.head_ids.is_empty() { i } else { self.head_ids[i as usize] } }

    /// Two-token MTP draft, after an accepted draft: "xn" = [h(pos) | h(pos+1)] from the double step; tokens tok1 (at pos+1)
    /// and tok2 (at pos+2) enter the MTP cache. Returns the draft for pos+3.
    pub fn draft2(&self, tok1: u32, tok2: u32, pos: usize) -> u32 {
        let p1 = pos + 1;
        let te = self.g.tensor("token_embd.weight").unwrap();
        let mut row = Vec::new(); self.g.read(te, tok1 as u64 * self.emb_row, self.emb_row, &mut row).unwrap();
        let mut row2 = Vec::new(); self.g.read(te, tok2 as u64 * self.emb_row, self.emb_row, &mut row2).unwrap();
        row.extend_from_slice(&row2);
        self.gpu.write(&self.a["emb_raw"], 0, &row);
        let mut tab: Vec<u8> = rope_table(p1 as i32).iter().flat_map(|v| v.to_le_bytes()).collect();
        tab.extend(rope_table(p1 as i32 + 1).iter().flat_map(|v| v.to_le_bytes()));
        self.gpu.write(&self.tab2, 0, &tab);
        let ub = |v: [u32; 8]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let u = self.u_pos2[N_LAYER].as_ref().expect("MTP layer not loaded");
        self.gpu.write(&u[0], 0, &ub([256, 4, 0, 0, 256, 256, 2, (p1 * 512) as u32]));
        self.gpu.write(&u[1], 0, &ub([1024, 0, 0, 0, 0, 0, 0, (p1 * 512) as u32]));
        self.gpu.write(&u[2], 0, &ub([p1 as u32 + 1, (ctx() / 128) as u32, fb(1.0 / 16.0), fd_phase(), 0, 0, attn2_flag(), self.mtp_start.get()]));   // attention from mtp_start up to p1 + tok
        if self.prof.borrow().is_some() { let b = self.run_profiled(&self.prog_mtp2, "draft2", 4); return self.draft_id(b[0]); }
        let (_, _, _, b) = self.gpu.run_commands_read(&self.prog_mtp2, &self.a["am2"], 4);
        self.draft_id(u32::from_le_bytes(b[..4].try_into().unwrap()))
    }

    /// Double draft for the verifications: h0, h1 (2048 f32 each) written into "xn", then `draft2`.
    pub fn draft2_verify(&self, h0: &[f32], h1: &[f32], tok1: u32, tok2: u32, pos: usize) -> u32 {
        let b: Vec<u8> = h0.iter().chain(h1).flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.write(&self.a["xn"], 0, &b);
        self.draft2(tok1, tok2, pos)
    }

    /// MECHTA_GEN_PROFILE: runs a program with a timestamp per dispatch (one compute pass each, so slower)
    /// and adds the GPU time of every dispatch to `prof` under "<tag>: <kernel>". Returns the first `n / 4` words of "am2".
    fn run_profiled(&self, prog: &[Command], tag: &str, n: usize) -> Vec<u32> {
        for c in prog.chunks(2000) {   // a QuerySet holds 4096 timestamps
            let m = self.gpu.run_commands_measured(c);
            let mut p = self.prof.borrow_mut(); let p = p.as_mut().unwrap();
            for (x, t) in c.iter().zip(&m.gpu_ms) { let e = p.gpu.entry(format!("{tag}: {}", x.n)).or_default(); e.0 += t; e.1 += x.byte; e.2 += 1; }
        }
        self.gpu.read_f32(&self.a["am2"], n / 4).iter().map(|x| x.to_bits()).collect()
    }
    /// Rejected draft: the DeltaNet and conv states go back to those after the first token of the last double step.
    pub fn restore(&self) {
        let par = self.last_par.get();
        let mut c = vec![];
        for il in 0..N_LAYER { if !attention(il) {
            c.push((&self.ssm1[il], &self.ssm[il], (32 * 128 * 128 * 4) as u64));
            c.push((&self.conv1[il], &self.conv[il][1 - par], (24576 * 4) as u64));
        } }
        self.gpu.copies(&c);
    }

    /// Takes a photo of the current DeltaNet and conv states (those the next step will read). One submit of copies, no wait.
    pub fn photograph(&self) {
        let mut f = self.photo.borrow_mut();
        if f.is_empty() {
            for il in 0..N_LAYER { if !attention(il) {
                f.push((self.gpu.empty_buffer("foto-ssm", (32 * 128 * 128 * 4) as u64), self.gpu.empty_buffer("foto-conv", (24576 * 4) as u64)));
            } }
        }
        let par = self.par.get();
        let mut c = vec![]; let mut j = 0;
        for il in 0..N_LAYER { if !attention(il) {
            c.push((&self.ssm[il], &f[j].0, (32 * 128 * 128 * 4) as u64));
            c.push((&self.conv[il][par], &f[j].1, (24576 * 4) as u64));
            j += 1;
        } }
        self.gpu.copies(&c);
    }
    /// Root (prefix across sessions): copies into RAM the current DeltaNet and conv states and the KV cache of
    /// positions [0, p) of the attention layers (not the MTP). To be called when the state is the one after exactly p tokens.
    pub fn read_root(&self, p: usize) -> Root {
        let par = self.par.get();
        let (mut ssm, mut conv, mut kv) = (vec![], vec![], vec![]);
        for il in 0..N_LAYER {
            if attention(il) {
                // the cache words as they are (f32 values, or the raw bits of f16 / q8 data followed by the q8 scales)
                let f = kv_format(); let sc0 = f.data_words(ctx() * 512);
                let read = |buf: &wgpu::Buffer| -> Vec<f32> {
                    let mut w = self.gpu.read_f32(buf, f.data_words(p * 512));
                    if f.scale_words(p * 512) > 0 { w.extend(self.gpu.read_f32_from(buf, sc0, f.scale_words(p * 512))); }
                    w
                };
                let (k, v) = (self.kv_buf.0[il].as_ref().unwrap(), self.kv_buf.1[il].as_ref().unwrap());
                kv.push((read(k), read(v)));
            } else {
                ssm.push(self.gpu.read_f32(&self.ssm[il], 32 * 128 * 128));
                conv.push(self.gpu.read_f32(&self.conv[il][par], 24576));
            }
        }
        Root { p, ssm, conv, kv }
    }
    /// Puts the root back: DeltaNet and conv states (into the conv buffer the next step will read) and KV [0, p).
    pub fn write_root(&self, r: &Root) {
        let par = self.par.get();
        let b = |v: &Vec<f32>| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let (mut j, mut a) = (0, 0);
        for il in 0..N_LAYER {
            if attention(il) {
                let f = kv_format(); let nd = f.data_words(r.p * 512); let sc0 = 4 * f.data_words(ctx() * 512) as u64;
                for (buf, w) in [(self.kv_buf.0[il].as_ref().unwrap(), &r.kv[a].0), (self.kv_buf.1[il].as_ref().unwrap(), &r.kv[a].1)] {
                    self.gpu.write(buf, 0, &b(&w[..nd].to_vec()));
                    if w.len() > nd { self.gpu.write(buf, sc0, &b(&w[nd..].to_vec())); }
                }
                a += 1;
            } else {
                self.gpu.write(&self.ssm[il], 0, &b(&r.ssm[j]));
                self.gpu.write(&self.conv[il][par], 0, &b(&r.conv[j]));
                j += 1;
            }
        }
        self.gpu.queue.submit([]);
    }
    /// Puts back the DeltaNet and conv states of the photo (into the conv buffer the next step will read).
    pub fn restore_photo(&self) {
        let f = self.photo.borrow();
        assert!(!f.is_empty(), "no photo taken");
        let par = self.par.get();
        let mut c = vec![]; let mut j = 0;
        for il in 0..N_LAYER { if !attention(il) {
            c.push((&f[j].0, &self.ssm[il], (32 * 128 * 128 * 4) as u64));
            c.push((&f[j].1, &self.conv[il][par], (24576 * 4) as u64));
            j += 1;
        } }
        self.gpu.copies(&c);
    }

    /// Reads a work buffer (for the verifications).
    pub fn read(&self, name: &str, n: usize) -> Vec<f32> { self.gpu.read_f32(&self.a[name], n) }
    /// Like `read`, for the buffers of the prefill (`ap`: "x", "h2", "esw", ...).
    pub fn read_p(&self, name: &str, n: usize) -> Vec<f32> { self.gpu.read_f32(&self.ap[name], n) }
    /// For `probe-experts`: prefill in blocks of 512 with one submit per layer, as `prefill_verify` but for the
    /// whole text; `f(block, segment)` after every segment (0 = embedding, 1 + layer), when the buffers can be read.
    pub fn prefill_probe(&self, toks: &[u32], f: &mut dyn FnMut(usize, usize)) {
        assert!(toks.len() % BLOCKS[0] == 0 && toks.len() <= ctx());
        for (b, block_toks) in toks.chunks(BLOCKS[0]).enumerate() {
            self.inputs_n(block_toks, b * BLOCKS[0], BLOCKS[0]);
            let prog = &self.prog_n[0][self.par.get()]; self.par.set(1 - self.par.get());
            let mut s = 0;
            for (i, &e) in self.end_n[0].iter().enumerate() { self.run(&prog[s..e]); f(b, i); s = e; }
            self.run(&prog[s..]);
        }
    }

    /// The argmax computed by the GPU in the last token (index, value): 8 bytes read. For the verifications of `token`.
    pub fn argmax_gpu(&self) -> (u32, f32) {
        let i = self.gpu.read_f32(&self.a["am2"], 1)[0].to_bits(); let v = self.gpu.read_f32(&self.a["am"], 1)[0];
        (i, v)
    }

    /// One token as `token`, but returns only the chosen token (argmax on the GPU, 8 bytes read) with ONE submit and one
    /// poll: the copy of the result is in the same command buffer as the computation. Records into `span` if active.
    pub fn step(&self, tok: u32, pos: usize) -> u32 {
        assert!(pos < ctx());
        self.inputs(tok, pos);
        let prog = &self.prog[self.par.get()]; self.par.set(1 - self.par.get());
        let (prep, wait, gpu, b) = self.gpu.run_commands_read(prog, &self.a["am2"], 4);
        if let Some(v) = self.span.borrow_mut().as_mut() { v.push((prep, wait, gpu)); }
        u32::from_le_bytes(b[..4].try_into().unwrap())
    }

    /// M3: block prefill. Processes `toks` from positions pos0.., NB at a time, without logits (the caller then does
    /// `step` on the last token of the prompt). Conv/DeltaNet states and KV cache as after the same `step` one by one.
    pub fn prefill(&self, toks: &[u32], pos0: usize) { self.prefill_cancellable(toks, pos0, &mut || true); }
    /// As `prefill`, asking `alive` between submits (at most every KEEPALIVE_EVERY) whether the client is still there; false =
    /// stop at once and return false (an abandoned request used to hold the GPU for minutes). The recurrent states
    /// are then half way through a block: the session never reuses them (next request: photo, root or zero).
    pub fn prefill_cancellable(&self, toks: &[u32], pos0: usize, alive: &mut dyn FnMut() -> bool) -> bool {
        assert!(pos0 + toks.len() <= ctx());
        let mut last = Instant::now();
        let mut go_on = |last: &mut Instant| -> bool { if last.elapsed() < KEEPALIVE_EVERY { return true; } *last = Instant::now(); alive() };
        // choice of the blocks: the sequence of sizes that costs least, with the costs measured per block
        // (ms, at ~14K of context, server profile): 512 → 3500, 128 → 1100, 32 → 840, 8 → 390. A block can be
        // partial (the last one). Dynamic programming on the number of tokens left.
        const COST: [f64; 4] = [3500.0, 1100.0, 840.0, 390.0];
        let n = toks.len();
        let mut best = vec![(0.0f64, 0usize); n + 1];   // (minimum cost to cover the first k tokens, index of the size of the last block)
        for k in 1..=n {
            best[k] = (f64::MAX, 0);
            for (j, &b) in BLOCKS.iter().enumerate() {
                let before = k.saturating_sub(b);
                let c = best[before].0 + COST[j];
                if c < best[k].0 { best[k] = (c, j); }
            }
        }
        let mut sequence = vec![]; let mut k = n;
        while k > 0 { let j = best[k].1; let b = BLOCKS[j].min(k); sequence.push((j, b)); k -= b; }
        sequence.reverse();   // the partial block (if any) is the first one found from the right: it is fine at any point
        let mut i = 0;
        for (ib, nb) in sequence {
            let block_toks = &toks[i..i + nb];
            self.inputs_n(block_toks, pos0 + i, BLOCKS[ib]);
            let prog = &self.prog_n[ib][self.par.get()]; self.par.set(1 - self.par.get());
            // STABILITY: one submit per layer, not one per block. At long context the attention of a block
            // of 512 tokens takes more than seconds: a single command buffer would risk the GPU timeout (10 s) and the reset.
            let mut s = 0;
            for &e in &self.cuts_n[ib] { self.run(&prog[s..e]); s = e; if !go_on(&mut last) { return false; } }
            self.run(&prog[s..]);
            i += nb;
        }
        true
    }
    /// Calibration: exactly one block of size BLOCKS[ib] (`toks` may be shorter: partial
    /// block), with the same submits as `prefill`. For measuring the cost of each block size (`block-cost`).
    pub fn single_block(&self, toks: &[u32], pos0: usize, ib: usize) {
        assert!(toks.len() <= BLOCKS[ib] && pos0 + toks.len() <= ctx());
        self.inputs_n(toks, pos0, BLOCKS[ib]);
        let prog = &self.prog_n[ib][self.par.get()]; self.par.set(1 - self.par.get());
        let mut s = 0;
        for &e in &self.cuts_n[ib] { self.run(&prog[s..e]); s = e; }
        self.run(&prog[s..]);
    }
    /// M3 (verifications): a single block, one submit per segment; `f(il, x)` with x = NB rows of "x" after the embedding (il = 0)
    /// and after every layer (il = 1 + layer), to compare them with the dump position by position.
    pub fn prefill_verify(&self, toks: &[u32], pos: usize, f: &mut dyn FnMut(usize, Vec<f32>)) {
        assert!(toks.len() <= NB);
        let ib = (0..BLOCKS.len()).rev().find(|&j| BLOCKS[j] >= toks.len()).unwrap_or(0);   // the smallest one that is enough
        self.inputs_n(toks, pos, BLOCKS[ib]);
        let prog = &self.prog_n[ib][self.par.get()]; self.par.set(1 - self.par.get());
        let mut s = 0;
        for (i, &e) in self.end_n[ib].iter().enumerate() { self.run(&prog[s..e]); f(i, self.gpu.read_f32(&self.ap["x"], toks.len() * 2048)); s = e; }
        self.run(&prog[s..]);
    }
    /// M3 (verifications): l_out of the last layer for the tokens of the last block of the prefill (row t = token t of the block).
    pub fn prefill_outputs(&self, n: usize) -> Vec<f32> { self.gpu.read_f32(&self.ap["x"], n * 2048) }
    /// MECHTA_PREFILL=blocks: the prompt goes through in blocks of NB tokens (M3) instead of one token at a time.
    pub fn prefill_in_blocks() -> bool { std::env::var("MECHTA_PREFILL").map(|v| v == "blocks").unwrap_or(false) }
    fn inputs_n(&self, toks: &[u32], pos: usize, nb: usize) {
        let tc = Instant::now();
        let n = toks.len(); assert!(n >= 1 && n <= nb && nb <= NB);
        let te = self.g.tensor("token_embd.weight").unwrap();
        // `Gguf::read` OVERWRITES the Vec (resize + read_exact): one row at a time, then in a row
        let mut rows = Vec::with_capacity(NB * self.emb_row as usize); let mut row = Vec::new();
        for &tok in toks { self.g.read(te, tok as u64 * self.emb_row, self.emb_row, &mut row).unwrap(); rows.extend_from_slice(&row); }
        self.gpu.write(&self.ap["emb_raw"], 0, &rows);
        let tab: Vec<u8> = (0..nb).flat_map(|t| rope_table((pos + t) as i32)).flat_map(|v| v.to_le_bytes()).collect();
        self.gpu.write(&self.tab_n, 0, &tab);
        let ub = |v: &[u32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        self.gpu.write(&self.u_n[0], 0, &ub(&[8192, n as u32, 0, 0, 0, 0, 0, NB as u32]));
        self.gpu.write(&self.u_n[1], 0, &ub(&[32, n as u32, 0, 0, 0, 0, 0, NB as u32]));
        self.gpu.write(&self.u_n[2], 0, &ub(&[n as u32, NB as u32, 0, 0]));
        for (rk, cv, at) in self.u_pos_n.iter().flatten() {
            self.gpu.write(rk, 0, &ub(&[256, 2 * nb as u32, 0, 0, 256, 256, 2, (pos * 512) as u32]));
            self.gpu.write(cv, 0, &ub(&[(nb * 512) as u32, 0, 0, 0, 0, 0, 0, (pos * 512) as u32]));
            for (ch, u) in at.iter().take(nb / 8).enumerate() {
                let (nb, bb) = if attn_batch() > 0 { (ctx() / prefill_block(), prefill_block() as u32) } else { (ctx() / 128, 0) };
                self.gpu.write(u, 0, &ub(&[pos as u32 + 1, nb as u32, fb(1.0 / 16.0), 0, (ch * 8) as u32, 0, bb, 0]));
            }
            // attn_fa: one uniform per (group of 64 tokens, split of FA_SPLIT chunks), after the NB / 8 of attn_fd4
            for (i, u) in at.iter().skip(NB / 8).take(nb / 64 * fa_splits()).enumerate() {
                let (g, sp) = (i / fa_splits(), i % fa_splits());
                self.gpu.write(u, 0, &ub(&[pos as u32 + 1, (ctx() / FA_CH) as u32, fb(1.0 / 16.0), fa_phase(), (g * 64) as u32, (sp * FA_SPLIT) as u32, FA_CH as u32, 0]));
            }
        }
        self.cpu("block inputs (embedding, RoPE, uniform)", tc);
    }

    fn cpu(&self, name: &'static str, t: Instant) {
        if let Some(p) = self.prof.borrow_mut().as_mut() { *p.cpu.entry(name).or_default() += t.elapsed().as_secs_f64() * 1e3; }
    }

    fn run(&self, c: &[Command]) {
        let mut pr = self.prof.borrow_mut();
        match pr.as_mut() {
            None => match self.span.borrow_mut().as_mut() {
                None => { self.gpu.run_commands(c); }
                Some(v) => v.push(self.gpu.run_commands_span(c)),
            },
            Some(p) => for c in c.chunks(2000) {   // a QuerySet holds 4096 timestamps: the 512 block has 2321 commands
                let m = self.gpu.run_commands_measured(c);
                *p.cpu.entry("submit: command preparation").or_default() += m.prep_ms;
                let sum: f64 = m.gpu_ms.iter().sum();
                *p.cpu.entry("submit: wait beyond the GPU time of the kernels").or_default() += m.wait_ms - sum;
                for (s, t) in c.iter().zip(&m.gpu_ms) { let e = p.gpu.entry(s.n.clone()).or_default(); e.0 += t; e.1 += s.byte; e.2 += 1; }
            },
        }
    }
}

/// The constant uniforms of the blocks (M4: shared between the token program and the MTP one).
struct Uniforms { u_norm: wgpu::Buffer, u_2048: wgpu::Buffer, u_l2: wgpu::Buffer, u_gn: wgpu::Buffer, u_32: wgpu::Buffer, u_8192: wgpu::Buffer, u_qn: wgpu::Buffer, u_kn: wgpu::Buffer, u_rq: wgpu::Buffer, u_4096: wgpu::Buffer, u_512: wgpu::Buffer, u_moe: wgpu::Buffer }
fn make_uniforms(of: &Factory) -> Uniforms {
    Uniforms {
        u_norm: of.u([2048, 1, fb(EPS), fb(1.0), 2048, 2048, 1, 0]),
        u_2048: of.u([2048, 0, 0, 0, 0, 0, 0, 0]),
        u_l2: of.u([128, 32, fb(EPS / 128.0), fb(1.0 / 128f32.sqrt()), 128, 128, 0, 0]),
        u_gn: of.u([128, 32, fb(EPS), fb(1.0), 128, 128, 3, 0]),
        u_32: of.u([32, 0, 0, 0, 0, 0, 0, 0]),
        u_8192: of.u([8192, 0, 0, 0, 0, 0, 0, 0]),
        u_qn: of.u([256, 16, fb(EPS), fb(1.0), 512, 256, 1, 0]),
        u_kn: of.u([256, 2, fb(EPS), fb(1.0), 256, 256, 1, 0]),
        u_rq: of.u([256, 16, 0, 0, 256, 256, 0, 0]),
        u_4096: of.u([4096, 0, 0, 0, 0, 0, 0, 0]),
        u_512: of.u([512, 0, 0, 0, 0, 0, 0, 0]),
        u_moe: of.u([2048, 8, 0, 0, 0, 0, 0, 0]),
    }
}

/// A block of the model (attention or DeltaNet, then MoE): reads "x", leaves the new residual in "x" (and in "fo" the output
/// of the MoE). `il` = N_LAYER is the MTP layer: an attention block with its own cache (M4).
#[allow(clippy::too_many_arguments)]
fn block_toks(of: &Factory, il: usize, c: &mut Vec<Command>, u: &Uniforms, par: usize, conv: &[[wgpu::Buffer; 2]], ssm: &[wgpu::Buffer],
          kc: &[Option<wgpu::Buffer>], vc: &[Option<wgpu::Buffer>], u_pos: &[Option<[wgpu::Buffer; 3]>], tab: &wgpu::Buffer) {
    let a = |n: &str| of.a(n); let w = |n: String| &of.w[&n];
    let l = |s: &str| format!("blk.{il}.{s}");
        c.push(of.op("rmsnorm", &[(0, a("x")), (1, w(l("attn_norm.weight")))], a("h"), None, &u.u_norm, (1, 1, 1)));
        if attention(il) || il == N_LAYER {
            let (kc, vc) = (kc[il].as_ref().unwrap(), vc[il].as_ref().unwrap());
            let up = u_pos[il].as_ref().unwrap();
            c.push(of.mv(&l("attn_q.weight"), "h", "qf", 0));
            c.push(of.mv(&l("attn_k.weight"), "h", "kb", 0));
            c.push(of.mv(&l("attn_v.weight"), "h", "vb", 0));
            c.push(of.op("rmsnorm", &[(0, a("qf")), (1, w(l("attn_q_norm.weight")))], a("qn"), None, &u.u_qn, (16, 1, 1)));
            c.push(of.op("rmsnorm", &[(0, a("kb")), (1, w(l("attn_k_norm.weight")))], a("kn"), None, &u.u_kn, (2, 1, 1)));
            c.push(of.op("rope", &[(0, a("qn")), (1, tab)], a("qr"), None, &u.u_rq, (16, 1, 1)));
            c.push(of.op(kv_writer("rope"), &[(0, a("kn")), (1, tab), (11, kc)], kc, None, &up[0], (2, 1, 1)));
            c.push(of.op(kv_writer("copy"), &[(0, a("vb")), (11, vc)], vc, None, &up[1], (8, 1, 1)));
            attn_op(of, c, false, kc, vc, &up[2]);
            c.push(of.op("attn_gate", &[(0, a("att")), (1, a("qf"))], a("ag"), None, &u.u_4096, (64, 1, 1)));
            c.push(of.mv(&l("attn_output.weight"), "ag", "att_out", 0));
        } else {
            let (cs_in, cs_out) = (&conv[il][par], &conv[il][1 - par]);
            c.push(of.mv(&l("attn_qkv.weight"), "h", "qkv", 0));
            c.push(of.mv(&l("attn_gate.weight"), "h", "z", 0));
            c.push(of.mv(&l("ssm_beta.weight"), "h", "braw", 0));
            c.push(of.mv(&l("ssm_alpha.weight"), "h", "araw", 0));
            c.push(of.op("beta_gate", &[(0, a("braw")), (1, a("araw")), (2, w(l("ssm_dt.bias"))), (3, w(l("ssm_a")))], a("bg"), None, &u.u_32, (1, 1, 1)));
            c.push(of.op("conv", &[(0, a("qkv")), (1, w(l("ssm_conv1d.weight"))), (2, cs_in)], a("cv"), Some(cs_out), &u.u_8192, (128, 1, 1)));
            c.push(of.op("rmsnorm", &[(0, a("cv"))], a("qk"), None, &u.u_l2, (32, 1, 1)));
            c.push(of.op("delta", &[(0, a("qk")), (1, a("cv")), (2, a("bg"))], &ssm[il], Some(a("ao")), &u.u_32, (32, 1, 1)));
            c.push(of.op("rmsnorm", &[(0, a("ao")), (1, w(l("ssm_norm.weight"))), (2, a("z"))], a("y"), None, &u.u_gn, (32, 1, 1)));
            c.push(of.mv(&l("ssm_out.weight"), "y", "att_out", 0));
        }
        c.push(of.op("sum", &[(0, a("att_out")), (1, a("x"))], a("x2"), None, &u.u_2048, (32, 1, 1)));
        c.push(of.op("rmsnorm", &[(0, a("x2")), (1, w(l("post_attention_norm.weight")))], a("h2"), None, &u.u_norm, (1, 1, 1)));
        c.push(of.mv(&l("ffn_gate_inp.weight"), "h2", "lg", 0));
        c.push(of.gpu.command(&of.rt, &[(0, a("lg")), (1, a("probs")), (2, a("ids")), (3, a("weights"))], (1, 1, 1), "routing".into(), 0));
        c.push(of.mv(&l("ffn_gate_exps.weight"), "h2", "eg", 0));
        c.push(of.mv(&l("ffn_up_exps.weight"), "h2", "eu", 0));
        c.push(of.op("swiglu", &[(0, a("eg")), (1, a("eu"))], a("esw"), None, &u.u_4096, (64, 1, 1)));
        c.push(of.mv(&l("ffn_down_exps.weight"), "esw", "ed", 512));
        c.push(of.mv(&l("ffn_gate_shexp.weight"), "h2", "sg", 0));
        c.push(of.mv(&l("ffn_up_shexp.weight"), "h2", "su", 0));
        c.push(of.op("swiglu", &[(0, a("sg")), (1, a("su"))], a("ssw"), None, &u.u_512, (8, 1, 1)));
        c.push(of.mv(&l("ffn_down_shexp.weight"), "ssw", "sd", 0));
        c.push(of.mv(&l("ffn_gate_inp_shexp.weight"), "h2", "sgt", 0));
        c.push(of.op("moe_sum", &[(0, a("ed")), (1, a("weights")), (2, a("sd")), (3, a("sgt")), (4, a("x2"))], a("fo"), Some(a("x")), &u.u_moe, (32, 1, 1)));
}


/// The program of an MTP draft (M4). Inputs: "xn" = h_nextn of the main model (state after output_norm),
/// "m_emb_raw" = embedding row of token t_{i+1}, m_tab and u_pos[N_LAYER] for position i+1.
/// Computation as `graph_mtp` of llama.cpp (qwen35moe.cpp): x = eh_proj(concat(enorm(e), hnorm(h))); attention block +
/// MoE of layer 40 with its own KV cache (positions from 1: the MTP cache does not have position 0); shared_head_norm;
/// shared output head; argmax. Segments (`seg_ends`): after eh_proj (x), attention (x2), MoE (x), norm (xn), logits.
#[allow(clippy::too_many_arguments)]
fn program_mtp(of: &Factory, conv: &[[wgpu::Buffer; 2]], ssm: &[wgpu::Buffer], kc: &[Option<wgpu::Buffer>], vc: &[Option<wgpu::Buffer>],
                 u_pos: &[Option<[wgpu::Buffer; 3]>], m_tab: &wgpu::Buffer) -> (Vec<Command>, Vec<usize>) {
    let a = |n: &str| of.a(n); let w = |n: &str| &of.w[n];
    let u = make_uniforms(of);
    let mut c = vec![]; let mut seg_ends = vec![];
    let u_dq = of.gpu.uniform(&[of.emb_bpr, 0, 0, 0]);
    c.push(of.gpu.command(&of.dq, &[(0, a("m_emb_raw")), (1, a("m_e")), (2, &u_dq)], (of.emb_bpr.div_ceil(64), 1, 1), "mtp embedding".into(), 0));
    let u_en = of.u([2048, 1, fb(EPS), fb(1.0), 2048, 2048, 1, 0]);
    let u_hn = of.u([2048, 1, fb(EPS), fb(1.0), 2048, 2048, 1, 2048]);   // extra = output offset: second half of the concat
    c.push(of.op("rmsnorm", &[(0, a("m_e")), (1, w("blk.40.nextn.enorm.weight"))], a("m_cat"), None, &u_en, (1, 1, 1)));
    c.push(of.op("rmsnorm", &[(0, a("xn")), (1, w("blk.40.nextn.hnorm.weight"))], a("m_cat"), None, &u_hn, (1, 1, 1)));
    c.push(of.mv("blk.40.nextn.eh_proj.weight", "m_cat", "x", 0));
    seg_ends.push(c.len());                                                  // 0: x = eh_proj
    let n0 = c.len();
    block_toks(of, N_LAYER, &mut c, &u, 0, conv, ssm, kc, vc, u_pos, m_tab);
    // inside the block: the "sum" (attention residual) writes x2; segment 1 ends there
    let i_sum = (n0..c.len()).find(|&i| c[i].n == "sum").expect("sum in the MTP block");
    seg_ends.push(i_sum + 1);                                              // 1: x2 = attn_residual
    seg_ends.push(c.len());                                                  // 2: x = post_ffn
    c.push(of.op("rmsnorm", &[(0, a("x")), (1, w("blk.40.nextn.shared_head_norm.weight"))], a("xn"), None, &u.u_norm, (1, 1, 1)));
    seg_ends.push(c.len());                                                  // 3: xn = h_nextn of the MTP
    if of.head_k > 0 {
        // the `mtp` verification compares the full logits: with the reduced head segment 4 is not comparable
        of.draft_head(&mut c, 0); seg_ends.push(c.len());
    } else {
        c.push(of.mv("output.weight", "xn", "logits", 0));
        seg_ends.push(c.len());                                              // 4: logits of the draft
        let u_am1 = of.u([VOCAB as u32, 0, 0, 0, 0, 0, 0, 0]);
        let u_am2 = of.u([(VOCAB / 256) as u32, 0, 0, 0, 0, 0, 0, 0]);
        c.push(of.op("argmax1", &[(0, a("logits"))], a("am_v"), Some(a("am_i")), &u_am1, ((VOCAB / 256) as u32, 1, 1)));
        c.push(of.op("argmax2", &[(0, a("am_v")), (1, a("am_i"))], a("am"), Some(a("am2")), &u_am2, (1, 1, 1)));
    }
    (c, seg_ends)
}

/// The uniforms of the double step (M4), shared between program2 and program_mtp2.
struct Uniforms2 { norm2: wgpu::Buffer, n4096: wgpu::Buffer, n8192: wgpu::Buffer, n1024: wgpu::Buffer, bg2: wgpu::Buffer, conv2: wgpu::Buffer,
              l2x2: wgpu::Buffer, gn2: wgpu::Buffer, qn2: wgpu::Buffer, kn2: wgpu::Buffer, rq2: wgpu::Buffer, moe2: wgpu::Buffer,
              n512: wgpu::Buffer, moe_tok1: wgpu::Buffer, merge: bool,
              norm2h: wgpu::Buffer, gn2h: wgpu::Buffer, x16: bool }
fn uniformi2(of: &Factory) -> Uniforms2 {
    Uniforms2 {
        norm2: of.u([2048, 2, fb(EPS), fb(1.0), 2048, 2048, 1, 0]),
        n4096: of.u([4096, 0, 0, 0, 0, 0, 0, 0]), n8192: of.u([8192, 0, 0, 0, 0, 0, 0, 0]), n1024: of.u([1024, 0, 0, 0, 0, 0, 0, 0]),
        bg2: of.u([32, 2, 0, 0, 0, 0, 0, 0]), conv2: of.u([8192, 2, 0, 0, 0, 0, 0, 0]),
        l2x2: of.u([128, 64, fb(EPS / 128.0), fb(1.0 / 128f32.sqrt()), 128, 128, 0, 0]),
        gn2: of.u([128, 64, fb(EPS), fb(1.0), 128, 128, 3, 0]),
        qn2: of.u([256, 32, fb(EPS), fb(1.0), 512, 256, 1, 0]), kn2: of.u([256, 4, fb(EPS), fb(1.0), 256, 256, 1, 0]),
        rq2: of.u([256, 32, 0, 0, 256, 256, 16, 0]),             // flag = 16 rows (heads) per token
        moe2: of.u([2048, 8, 0, 0, 0, 0, 0, 2]),                // extra = 2 tokens
        n512: of.u([512, 0, 0, 0, 0, 0, 0, 0]),
        moe_tok1: of.u([2048, 8, 0, 0, 0, 0, 2048, 0]),          // one token, flag = offset 2048 of residual and output
        norm2h: of.u([2048, 2, fb(EPS), fb(1.0), 2048, 2048, 1 | 4, 0]),   // f16 output
        gn2h: of.u([128, 64, fb(EPS), fb(1.0), 128, 128, 3 | 4, 0]),
        // x in f16 in the double step (default); MECHTA_X16=no for x in f32
        x16: std::env::var("MECHTA_X16").map(|v| v != "no").unwrap_or(true),
        // experts in union (default); MECHTA_EXPERTS=token = one pass per token (comparisons)
        merge: std::env::var("MECHTA_EXPERTS").map(|v| v != "token").unwrap_or(true),
    }
}

/// A block for TWO tokens (M4): reads "x" = [tok0 | tok1], leaves the new residual in "x". `il` = N_LAYER: MTP layer.
#[allow(clippy::too_many_arguments)]
fn block2(of: &Factory, il: usize, c: &mut Vec<Command>, u: &Uniforms2, par: usize, conv: &[[wgpu::Buffer; 2]], conv1: &[wgpu::Buffer],
           ssm: &[wgpu::Buffer], ssm1: &[wgpu::Buffer], kc: &[Option<wgpu::Buffer>], vc: &[Option<wgpu::Buffer>],
           u_pos2: &[Option<[wgpu::Buffer; 3]>], tab2: &wgpu::Buffer, moe_only_tok1: bool) {
    let a = |n: &str| of.a(n); let w = |n: String| &of.w[&n];
    let l = |s: &str| format!("blk.{il}.{s}");
    // x in f16 (M4): the producers of x write f16 and the 2-column matvecs read it as f16; the MTP block (moe_only_tok1)
    // stays in f32 because its MoE uses the single-column kernels.
    let h16 = u.x16 && !moe_only_tok1;
    let mv2 = |n: &str, x: &str, y: &str| if h16 { of.mv2h(n, x, y) } else { of.mv2(n, x, y) };
    let norm2 = if h16 { &u.norm2h } else { &u.norm2 }; let gn2 = if h16 { &u.gn2h } else { &u.gn2 };
    let swi = if h16 { "swiglu16" } else { "swiglu" }; let agate = if h16 { "attn_gate16" } else { "attn_gate" };
    let d2 = |n: u32| if h16 { n / 2 } else { n };   // threads per pair in the ...16 kernels
    c.push(of.op("rmsnorm", &[(0, a("x")), (1, w(l("attn_norm.weight")))], a("h"), None, norm2, (2, 1, 1)));
    if attention(il) || il == N_LAYER {
        let (kc, vc) = (kc[il].as_ref().unwrap(), vc[il].as_ref().unwrap());
        let up = u_pos2[il].as_ref().unwrap();
        c.push(mv2(&l("attn_q.weight"), "h", "qf"));
        c.push(mv2(&l("attn_k.weight"), "h", "kb"));
        c.push(mv2(&l("attn_v.weight"), "h", "vb"));
        c.push(of.op("rmsnorm", &[(0, a("qf")), (1, w(l("attn_q_norm.weight")))], a("qn"), None, &u.qn2, (32, 1, 1)));
        c.push(of.op("rmsnorm", &[(0, a("kb")), (1, w(l("attn_k_norm.weight")))], a("kn"), None, &u.kn2, (4, 1, 1)));
        c.push(of.op("rope2", &[(0, a("qn")), (1, tab2)], a("qr"), None, &u.rq2, (32, 1, 1)));
        c.push(of.op(kv_writer("rope2"), &[(0, a("kn")), (1, tab2), (11, kc)], kc, None, &up[0], (4, 1, 1)));
        c.push(of.op(kv_writer("copy"), &[(0, a("vb")), (11, vc)], vc, None, &up[1], (16, 1, 1)));
        attn_op(of, c, true, kc, vc, &up[2]);
        c.push(of.op(agate, &[(0, a("att")), (1, a("qf"))], a("ag"), None, &u.n8192, (d2(128), 1, 1)));
        c.push(mv2(&l("attn_output.weight"), "ag", "att_out"));
    } else {
        let (cs_in, cs_out) = (&conv[il][par], &conv[il][1 - par]);
        c.push(mv2(&l("attn_qkv.weight"), "h", "qkv"));
        c.push(mv2(&l("attn_gate.weight"), "h", "z"));
        c.push(mv2(&l("ssm_beta.weight"), "h", "braw"));
        c.push(mv2(&l("ssm_alpha.weight"), "h", "araw"));
        c.push(of.op("beta_gate2", &[(0, a("braw")), (1, a("araw")), (2, w(l("ssm_dt.bias"))), (3, w(l("ssm_a")))], a("bg"), None, &u.bg2, (1, 1, 1)));
        c.push(of.op("conv2", &[(0, a("qkv")), (1, w(l("ssm_conv1d.weight"))), (2, cs_in), (8, &conv1[il])], a("cv"), Some(cs_out), &u.conv2, (128, 1, 1)));
        c.push(of.op("rmsnorm", &[(0, a("cv"))], a("qk"), None, &u.l2x2, (64, 1, 1)));
        c.push(of.op("delta2", &[(0, a("qk")), (1, a("cv")), (2, a("bg")), (8, a("ao"))], &ssm[il], Some(&ssm1[il]), &u.bg2, (32, 1, 1)));
        c.push(of.op("rmsnorm", &[(0, a("ao")), (1, w(l("ssm_norm.weight"))), (2, a("z"))], a("y"), None, gn2, (64, 1, 1)));
        c.push(mv2(&l("ssm_out.weight"), "y", "att_out"));
    }
    c.push(of.op("sum", &[(0, a("att_out")), (1, a("x"))], a("x2"), None, &u.n4096, (64, 1, 1)));
    c.push(of.op("rmsnorm", &[(0, a("x2")), (1, w(l("post_attention_norm.weight")))], a("h2"), None, norm2, (2, 1, 1)));
    if moe_only_tok1 {
        // 2-token MTP draft: of the first token only K and V are needed (already in cache); the MoE runs only for the second,
        // with the inputs of the second token (x_off 2048) in place of the first and the result written into x[2048..].
        c.push(of.mv_x(&l("ffn_gate_inp.weight"), "h2", "lg", 2048));
        c.push(of.gpu.command(&of.rt, &[(0, a("lg")), (1, a("probs")), (2, a("ids")), (3, a("weights"))], (1, 1, 1), "routing".into(), 0));
        c.push(of.mv_tok(&l("ffn_gate_exps.weight"), "h2", "eg", 0, 2048, 0, "ids"));
        c.push(of.mv_tok(&l("ffn_up_exps.weight"), "h2", "eu", 0, 2048, 0, "ids"));
        c.push(of.op("swiglu", &[(0, a("eg")), (1, a("eu"))], a("esw"), None, &u.n4096, (64, 1, 1)));
        c.push(of.mv_tok(&l("ffn_down_exps.weight"), "esw", "ed", 512, 0, 0, "ids"));
        c.push(of.mv_x(&l("ffn_gate_shexp.weight"), "h2", "sg", 2048));
        c.push(of.mv_x(&l("ffn_up_shexp.weight"), "h2", "su", 2048));
        c.push(of.op("swiglu", &[(0, a("sg")), (1, a("su"))], a("ssw"), None, &u.n512, (8, 1, 1)));
        c.push(of.mv_x(&l("ffn_down_shexp.weight"), "ssw", "sd", 0));
        c.push(of.mv_x(&l("ffn_gate_inp_shexp.weight"), "h2", "sgt", 2048));
        c.push(of.op("moe_sum", &[(0, a("ed")), (1, a("weights")), (2, a("sd")), (3, a("sgt")), (4, a("x2"))], a("fo"), Some(a("x")), &u.moe_tok1, (32, 1, 1)));
        return;
    }
    c.push(mv2(&l("ffn_gate_inp.weight"), "h2", "lg"));
    c.push(of.gpu.command(&of.rt, &[(0, a("lg")), (1, a("probs")), (2, a("ids")), (3, a("weights"))], (1, 1, 1), "routing".into(), 0));
    c.push(of.gpu.command(&of.rt2, &[(0, a("lg")), (1, a("probs")), (2, a("ids2")), (3, a("weights"))], (1, 1, 1), "routing b".into(), 0));
    if u.merge {
        let mvu = |n: &str, x: &str, y: &str, xp: u32, xt: u32| if h16 { of.mv_uh(n, x, y, xp, xt) } else { of.mv_u(n, x, y, xp, xt) };
        c.push(of.gpu.command(&of.un, &[(0, a("ids")), (1, a("ids2")), (2, a("uniforms"))], (1, 1, 1), "merge".into(), 0));
        c.push(mvu(&l("ffn_gate_exps.weight"), "h2", "eg", 0, 2048));
        c.push(mvu(&l("ffn_up_exps.weight"), "h2", "eu", 0, 2048));
        c.push(of.op(swi, &[(0, a("eg")), (1, a("eu"))], a("esw"), None, &u.n8192, (d2(128), 1, 1)));
        c.push(mvu(&l("ffn_down_exps.weight"), "esw", "ed", 512, 4096));
    } else {
        assert!(!h16, "MECHTA_EXPERTS=token requires MECHTA_X16=no");
        for (tok, ids) in [(0u32, "ids"), (1u32, "ids2")] {
            c.push(of.mv_tok(&l("ffn_gate_exps.weight"), "h2", "eg", 0, tok * 2048, tok * 4096, ids));
            c.push(of.mv_tok(&l("ffn_up_exps.weight"), "h2", "eu", 0, tok * 2048, tok * 4096, ids));
        }
        c.push(of.op("swiglu", &[(0, a("eg")), (1, a("eu"))], a("esw"), None, &u.n8192, (128, 1, 1)));
        for (tok, ids) in [(0u32, "ids"), (1u32, "ids2")] {
            c.push(of.mv_tok(&l("ffn_down_exps.weight"), "esw", "ed", 512, tok * 4096, tok * 16384, ids));
        }
    }
    c.push(mv2(&l("ffn_gate_shexp.weight"), "h2", "sg"));
    c.push(mv2(&l("ffn_up_shexp.weight"), "h2", "su"));
    c.push(of.op(swi, &[(0, a("sg")), (1, a("su"))], a("ssw"), None, &u.n1024, (d2(16), 1, 1)));
    c.push(mv2(&l("ffn_down_shexp.weight"), "ssw", "sd"));
    c.push(mv2(&l("ffn_gate_inp_shexp.weight"), "h2", "sgt"));
    c.push(of.op("moe_sum", &[(0, a("ed")), (1, a("weights")), (2, a("sd")), (3, a("sgt")), (4, a("x2"))], a("fo"), Some(a("x")), &u.moe2, (64, 1, 1)));
}

/// M3: the fixed uniforms of the block prefill (NB tokens; the per-token counts are NB, the tokens beyond n give harmless
/// garbage in the buffers; only conv_n, delta_n and routing use n, rewritten at every block in `u_n`).
struct UniformsN { norm: wgpu::Buffer, qn: wgpu::Buffer, kn: wgpu::Buffer, rq: wgpu::Buffer, bg: wgpu::Buffer, l2: wgpu::Buffer, gn: wgpu::Buffer,
              n_x: wgpu::Buffer, n_ag: wgpu::Buffer, n_esw: wgpu::Buffer, n_ssw: wgpu::Buffer, moe: wgpu::Buffer, dq: wgpu::Buffer }
fn uniformi_n(of: &Factory, nb: usize) -> UniformsN {
    let nb = nb as u32;
    UniformsN {
        norm: of.u([2048, nb, fb(EPS), fb(1.0), 2048, 2048, 1, 0]),
        qn: of.u([256, 16 * nb, fb(EPS), fb(1.0), 512, 256, 1, 0]), kn: of.u([256, 2 * nb, fb(EPS), fb(1.0), 256, 256, 1, 0]),
        rq: of.u([256, 16 * nb, 0, 0, 256, 256, 16, 0]),
        bg: of.u([32, nb, 0, 0, 0, 0, 0, 0]),
        l2: of.u([128, 32 * nb, fb(EPS / 128.0), fb(1.0 / 128f32.sqrt()), 128, 128, 0, 0]),
        gn: of.u([128, 32 * nb, fb(EPS), fb(1.0), 128, 128, 3, 0]),
        n_x: of.u([2048 * nb, 0, 0, 0, 0, 0, 0, 0]), n_ag: of.u([4096 * nb, 0, 0, 0, 0, 0, 0, 0]),
        n_esw: of.u([8 * 512 * nb, 0, 0, 0, 0, 0, 0, 0]), n_ssw: of.u([512 * nb, 0, 0, 0, 0, 0, 0, 0]),
        moe: of.u([2048, 8, 0, 0, 0, 0, 0, nb]),
        dq: of.gpu.uniform(&[of.emb_bpr * nb, 0, 0, 0]),
    }
}
/// M3: a block (layer) for NB tokens in the prefill. Same computation as `block_toks`/`block2`, NB-column GEMM, DeltaNet in one
/// dispatch for the block, flash-decoding attention in lots of 8 tokens, experts grouped by expert (each read once).
#[allow(clippy::too_many_arguments)]
fn block_n(of: &Factory, il: usize, c: &mut Vec<Command>, u: &UniformsN, par: usize, conv: &[[wgpu::Buffer; 2]], ssm: &[wgpu::Buffer],
            kc: &[Option<wgpu::Buffer>], vc: &[Option<wgpu::Buffer>], u_n: &[wgpu::Buffer; 3],
            u_pos_n: &[Option<(wgpu::Buffer, wgpu::Buffer, Vec<wgpu::Buffer>)>], tab_n: &wgpu::Buffer, nbu: usize) {
    let a = |n: &str| of.a(n); let w = |n: String| &of.w[&n];
    let l = |s: &str| format!("blk.{il}.{s}");
    let nb = nbu as u32;
    c.push(of.op("rmsnorm", &[(0, a("x")), (1, w(l("attn_norm.weight")))], a("h"), None, &u.norm, (nb, 1, 1)));
    if attention(il) {
        let (kc, vc) = (kc[il].as_ref().unwrap(), vc[il].as_ref().unwrap());
        let (rk, cv, at) = u_pos_n[il].as_ref().unwrap();
        c.push(of.gemm(&l("attn_q.weight"), "h", "qf", nbu));
        c.push(of.gemm(&l("attn_k.weight"), "h", "kb", nbu));
        c.push(of.gemm(&l("attn_v.weight"), "h", "vb", nbu));
        c.push(of.op("rmsnorm", &[(0, a("qf")), (1, w(l("attn_q_norm.weight")))], a("qn"), None, &u.qn, (16 * nb, 1, 1)));
        c.push(of.op("rmsnorm", &[(0, a("kb")), (1, w(l("attn_k_norm.weight")))], a("kn"), None, &u.kn, (2 * nb, 1, 1)));
        c.push(of.op("rope2", &[(0, a("qn")), (1, tab_n)], a("qr"), None, &u.rq, (16 * nb, 1, 1)));
        c.push(of.op(kv_writer("rope2"), &[(0, a("kn")), (1, tab_n), (11, kc)], kc, None, rk, (2 * nb, 1, 1)));
        c.push(of.op(kv_writer("copy"), &[(0, a("vb")), (11, vc)], vc, None, cv, (nb * 512 / 64, 1, 1)));
        if prefill_mx() && nbu >= 64 {
            // attn_fa_mx: Q to f16 once, then as attn_fa (same uniforms, same partials) with grid (kvh x half, chunks, 4 x 16 tokens)
            let (ns, nch) = (fa_splits(), ctx() / FA_CH);
            c.push(of.op("q_f16", &[(0, a("qr"))], a("qh"), None, &of.u([(nbu * 2048) as u32, 0, 0, 0, 0, 0, 0, 0]), ((nbu * 8) as u32, 1, 1)));
            for g in 0..nbu / 64 {
                let ug = &at[NB / 8 + g * ns..NB / 8 + (g + 1) * ns];
                for (sp, ua) in ug.iter().enumerate() {
                    let n = FA_SPLIT.min(nch - sp * FA_SPLIT) as u32;
                    c.push(of.op(mx_entry(), &[(10, vc), (12, kc), (13, a("qh")), (14, vc)], a("att_p"), None, ua, (4, n, 4)));
                }
                c.push(of.op("attn_fd_combine", &[(0, a("att_p"))], a("att"), None, &ug[0], (16, 64, 1)));
            }
        } else if prefill_fa() && nbu >= 64 {
            let (ns, nch) = (fa_splits(), ctx() / FA_CH);
            for g in 0..nbu / 64 {
                let ug = &at[NB / 8 + g * ns..NB / 8 + (g + 1) * ns];
                for (sp, ua) in ug.iter().enumerate() {
                    let n = FA_SPLIT.min(nch - sp * FA_SPLIT) as u32;
                    c.push(of.op("attn_fa", &[(0, a("qr")), (9, kc), (10, vc)], a("att_p"), None, ua, (2, n, 8)));
                }
                c.push(of.op("attn_fd_combine", &[(0, a("att_p"))], a("att"), None, &ug[0], (16, 64, 1)));
            }
        } else { for ua in &at[..nbu / 8] {
            // attn_fd4 (default): K/V read once per 4 tokens, blocks of prefill_block() positions; MECHTA_ATTN_BATCH=0 for attn_fd per token
            if attn_batch() > 0 { c.push(of.op("attn_fd4", &[(0, a("qr")), (9, kc), (10, vc)], a("att_p"), None, ua, (2, (ctx() / prefill_block()) as u32, 1))); }
            else { c.push(of.op("attn_fd", &[(0, a("qr")), (9, kc), (10, vc)], a("att_p"), None, ua, (2, (ctx() / 128) as u32, 8))); }
            c.push(of.op("attn_fd_combine", &[(0, a("att_p"))], a("att"), None, ua, (16, 8, 1)));
        } }
        c.push(of.op("attn_gate", &[(0, a("att")), (1, a("qf"))], a("ag"), None, &u.n_ag, (nb * 4096 / 64, 1, 1)));
        c.push(of.gemm(&l("attn_output.weight"), "ag", "att_out", nbu));
    } else {
        let (cs_in, cs_out) = (&conv[il][par], &conv[il][1 - par]);
        c.push(of.gemm(&l("attn_qkv.weight"), "h", "qkv", nbu));
        c.push(of.gemm(&l("attn_gate.weight"), "h", "z", nbu));
        c.push(of.gemm(&l("ssm_beta.weight"), "h", "braw", nbu));
        c.push(of.gemm(&l("ssm_alpha.weight"), "h", "araw", nbu));
        c.push(of.op("beta_gate2", &[(0, a("braw")), (1, a("araw")), (2, w(l("ssm_dt.bias"))), (3, w(l("ssm_a")))], a("bg"), None, &u.bg, (nb / 2, 1, 1)));
        c.push(of.op("conv_n", &[(0, a("qkv")), (1, w(l("ssm_conv1d.weight"))), (2, cs_in)], a("cv"), Some(cs_out), &u_n[0], (128, 1, 1)));
        c.push(of.op("rmsnorm", &[(0, a("cv"))], a("qk"), None, &u.l2, (32 * nb, 1, 1)));
        c.push(of.op("delta_n", &[(0, a("qk")), (1, a("cv")), (2, a("bg")), (8, a("ao"))], &ssm[il], None, &u_n[1], (32, 1, 1)));
        c.push(of.op("rmsnorm", &[(0, a("ao")), (1, w(l("ssm_norm.weight"))), (2, a("z"))], a("y"), None, &u.gn, (32 * nb, 1, 1)));
        c.push(of.gemm(&l("ssm_out.weight"), "y", "att_out", nbu));
    }
    c.push(of.op("sum", &[(0, a("att_out")), (1, a("x"))], a("x2"), None, &u.n_x, (nb * 2048 / 64, 1, 1)));
    c.push(of.op("rmsnorm", &[(0, a("x2")), (1, w(l("post_attention_norm.weight")))], a("h2"), None, &u.norm, (nb, 1, 1)));
    c.push(of.gemm(&l("ffn_gate_inp.weight"), "h2", "lg", nbu));
    c.push(of.gpu.command(&of.rtn, &[(0, a("lg")), (1, a("probs")), (2, a("ids")), (3, a("weights")), (4, &u_n[2])], (nb, 1, 1), "routing xN".into(), 0));
    c.push(of.gpu.command(&of.rg, &[(2, a("ids")), (4, &u_n[2]), (5, a("grp"))], (1, 1, 1), "gather".into(), 0));
    c.push(of.gemm_e(&l("ffn_gate_exps.weight"), "h2", "eg", 0, 2048, nbu));
    c.push(of.gemm_e(&l("ffn_up_exps.weight"), "h2", "eu", 0, 2048, nbu));
    c.push(of.op("swiglu", &[(0, a("eg")), (1, a("eu"))], a("esw"), None, &u.n_esw, (nb * 8 * 512 / 64, 1, 1)));
    c.push(of.gemm_e(&l("ffn_down_exps.weight"), "esw", "ed", 512, 8 * 512, nbu));
    c.push(of.gemm(&l("ffn_gate_shexp.weight"), "h2", "sg", nbu));
    c.push(of.gemm(&l("ffn_up_shexp.weight"), "h2", "su", nbu));
    c.push(of.op("swiglu", &[(0, a("sg")), (1, a("su"))], a("ssw"), None, &u.n_ssw, (nb * 512 / 64, 1, 1)));
    c.push(of.gemm(&l("ffn_down_shexp.weight"), "ssw", "sd", nbu));
    c.push(of.gemm(&l("ffn_gate_inp_shexp.weight"), "h2", "sgt", nbu));
    c.push(of.op("moe_sum", &[(0, a("ed")), (1, a("weights")), (2, a("sd")), (3, a("sgt")), (4, a("x2"))], a("fo"), Some(a("x")), &u.moe, (nb * 2048 / 64, 1, 1)));
}
/// M3: the program of a block of NB tokens: embedding, 40 layers, no head (the prompt asks for no logits).
#[allow(clippy::too_many_arguments)]
/// Submit boundaries of a prefill program: the layer ends, plus a cut before every `MECHTA_ATTN_PIECES`-th
/// attention dispatch (default 2). One attention dispatch of a 512-token block covers 8 tokens; before, a whole
/// attention layer was one submit of ~370 ms at 18K context, growing with the context, and while a submit runs no
/// other program gets the GPU (mouse and browser stutter). Same order of commands: only the submits are smaller.
fn attn_cuts(c: &[Command], seg_ends: &[usize]) -> Vec<usize> {
    let pieces: usize = std::env::var("MECHTA_ATTN_PIECES").ok().and_then(|v| v.parse().ok()).unwrap_or(2).max(1);
    let mut t = seg_ends.to_vec(); let mut k = 0;
    for (i, x) in c.iter().enumerate() {
        if x.n == "attn_fd4" || x.n == "attn_fd" || x.n == "attn_fa" { if k % pieces == 0 { t.push(i); } k += 1; }
    }
    t.sort_unstable(); t.dedup(); t.retain(|&e| e > 0 && e < c.len()); t
}

fn program_n(of: &Factory, par: usize, conv: &[[wgpu::Buffer; 2]], ssm: &[wgpu::Buffer], kc: &[Option<wgpu::Buffer>], vc: &[Option<wgpu::Buffer>],
               u_n: &[wgpu::Buffer; 3], u_pos_n: &[Option<(wgpu::Buffer, wgpu::Buffer, Vec<wgpu::Buffer>)>], tab_n: &wgpu::Buffer, nb: usize) -> (Vec<Command>, Vec<usize>) {
    let a = |n: &str| of.a(n);
    let u = uniformi_n(of, nb);
    let mut c = vec![]; let mut seg_ends = vec![];
    c.push(of.gpu.command(&of.dq, &[(0, a("emb_raw")), (1, a("x")), (2, &u.dq)], ((of.emb_bpr * nb as u32).div_ceil(64), 1, 1), format!("embedding x{nb}"), 0));
    seg_ends.push(c.len());   // after the embedding
    for il in 0..N_LAYER { block_n(of, il, &mut c, &u, par, conv, ssm, kc, vc, u_n, u_pos_n, tab_n, nb); seg_ends.push(c.len()); }
    (c, seg_ends)
}

/// Argmax of the two tokens of the double step: am[tok], am2[tok].
fn argmax2x(of: &Factory, c: &mut Vec<Command>) {
    let a = |n: &str| of.a(n);
    for tok in 0..2u32 {
        let u1 = of.u([VOCAB as u32, tok * (VOCAB / 256) as u32, 0, 0, 0, 0, 0, tok * VOCAB as u32]);
        let u2 = of.u([(VOCAB / 256) as u32, tok, 0, 0, 0, 0, 0, tok * (VOCAB / 256) as u32]);
        c.push(of.op("argmax1", &[(0, a("logits"))], a("am_v"), Some(a("am_i")), &u1, ((VOCAB / 256) as u32, 1, 1)));
        c.push(of.op("argmax2", &[(0, a("am_v")), (1, a("am_i"))], a("am"), Some(a("am2")), &u2, (1, 1, 1)));
    }
}

/// M4: the double-step program: two tokens (accepted + draft) in one pass over the weights. The buffers hold the two tokens
/// in a row. `seg_ends`: end of each layer (x = [l_out tok0 | l_out tok1]).
#[allow(clippy::too_many_arguments)]
fn program2(of: &Factory, par: usize, conv: &[[wgpu::Buffer; 2]], conv1: &[wgpu::Buffer], ssm: &[wgpu::Buffer], ssm1: &[wgpu::Buffer],
              kc: &[Option<wgpu::Buffer>], vc: &[Option<wgpu::Buffer>], u_pos2: &[Option<[wgpu::Buffer; 3]>], tab2: &wgpu::Buffer) -> (Vec<Command>, Vec<usize>) {
    let a = |n: &str| of.a(n);
    let u = uniformi2(of);
    let mut c = vec![]; let mut seg_ends = vec![];
    let u_dq = of.gpu.uniform(&[2 * of.emb_bpr, 0, 0, 0]);
    c.push(of.gpu.command(&of.dq, &[(0, a("emb_raw")), (1, a("x")), (2, &u_dq)], ((2 * of.emb_bpr).div_ceil(64), 1, 1), "embedding x2".into(), 0));
    for il in 0..N_LAYER {
        block2(of, il, &mut c, &u, par, conv, conv1, ssm, ssm1, kc, vc, u_pos2, tab2, false);
        seg_ends.push(c.len());
    }
    c.push(of.op("rmsnorm", &[(0, a("x")), (1, &of.w["output_norm.weight"])], a("xn"), None, &u.norm2, (2, 1, 1)));   // h_nextn in f32 for the MTP
    if u.x16 {
        c.push(of.op("rmsnorm", &[(0, a("x")), (1, &of.w["output_norm.weight"])], a("xn16"), None, &u.norm2h, (2, 1, 1)));
        c.push(of.mv2h("output.weight", "xn16", "logits"));
    } else {
        c.push(of.mv2("output.weight", "xn", "logits"));
    }
    argmax2x(of, &mut c);
    (c, seg_ends)
}

/// M4: MTP draft for TWO tokens (after an accepted draft the MTP cache must receive the two new tokens).
/// Inputs: "xn" = [h_nextn(pos) | h_nextn(pos+1)] as left by the double step, "emb_raw" = rows of the two tokens
/// (those at positions pos+1 and pos+2), tab2 and u_pos2[N_LAYER] for positions pos+1 and pos+2.
/// Output: am2[0] = draft for pos+3 (the output head runs only on the second token; the first one's is not needed).
#[allow(clippy::too_many_arguments)]
fn program_mtp2(of: &Factory, conv: &[[wgpu::Buffer; 2]], conv1: &[wgpu::Buffer], ssm: &[wgpu::Buffer], ssm1: &[wgpu::Buffer],
                  kc: &[Option<wgpu::Buffer>], vc: &[Option<wgpu::Buffer>], u_pos2: &[Option<[wgpu::Buffer; 3]>], tab2: &wgpu::Buffer) -> Vec<Command> {
    let a = |n: &str| of.a(n); let w = |n: &str| &of.w[n];
    let u = uniformi2(of);
    let mut c = vec![];
    let u_dq = of.gpu.uniform(&[2 * of.emb_bpr, 0, 0, 0]);
    c.push(of.gpu.command(&of.dq, &[(0, a("emb_raw")), (1, a("m_e")), (2, &u_dq)], ((2 * of.emb_bpr).div_ceil(64), 1, 1), "mtp embedding x2".into(), 0));
    let u_en = of.u([2048, 2, fb(EPS), fb(1.0), 2048, 4096, 1, 0]);      // row t → m_cat[t*4096]
    let u_hn = of.u([2048, 2, fb(EPS), fb(1.0), 2048, 4096, 1, 2048]);   // row t → m_cat[t*4096 + 2048]
    c.push(of.op("rmsnorm", &[(0, a("m_e")), (1, w("blk.40.nextn.enorm.weight"))], a("m_cat"), None, &u_en, (2, 1, 1)));
    c.push(of.op("rmsnorm", &[(0, a("xn")), (1, w("blk.40.nextn.hnorm.weight"))], a("m_cat"), None, &u_hn, (2, 1, 1)));
    c.push(of.mv2("blk.40.nextn.eh_proj.weight", "m_cat", "x"));
    // MECHTA_MTP2=full: MoE for the first token too (an earlier version, 16/16 drafts); default: cut (on trial, 15/16)
    let full = std::env::var("MECHTA_MTP2").map(|v| v == "full").unwrap_or(false);
    block2(of, N_LAYER, &mut c, &u, 0, conv, conv1, ssm, ssm1, kc, vc, u_pos2, tab2, !full);
    // only the draft of the second token is needed: norm on both (2 rows, costs nothing), output head with ONE column
    // on xn[2048..] (x_off), argmax in am2[0]. Before: 2-column head (6.3 ms vs 5.0) and two argmax.
    c.push(of.op("rmsnorm", &[(0, a("x")), (1, w("blk.40.nextn.shared_head_norm.weight"))], a("xn"), None, &u.norm2, (2, 1, 1)));
    of.draft_head(&mut c, 2048);
    c
}

/// The dispatch list of one token. `par` = even (0) or odd (1) position: selects the buffer
/// of the conv state to read and the one to write.
#[allow(clippy::too_many_arguments)]
fn program(of: &Factory, par: usize, conv: &[[wgpu::Buffer; 2]], ssm: &[wgpu::Buffer], kc: &[Option<wgpu::Buffer>], vc: &[Option<wgpu::Buffer>],
             u_pos: &[Option<[wgpu::Buffer; 3]>], tab: &wgpu::Buffer) -> (Vec<Command>, Vec<usize>) {
    let a = |n: &str| of.a(n);
    let mut c = vec![];
    let mut seg_ends = vec![];
    // embedding: the row (written at every token in emb_raw, format from emb_format), decoded on the GPU
    let u = of.gpu.uniform(&[of.emb_bpr, 0, 0, 0]);
    c.push(of.gpu.command(&of.dq, &[(0, a("emb_raw")), (1, a("x")), (2, &u)], (of.emb_bpr.div_ceil(64), 1, 1), "embedding".into(), 0));
    let u = make_uniforms(of);
    for il in 0..N_LAYER {
        block_toks(of, il, &mut c, &u, par, conv, ssm, kc, vc, u_pos, tab);
        seg_ends.push(c.len());
    }
    c.push(of.op("rmsnorm", &[(0, a("x")), (1, &of.w["output_norm.weight"])], a("xn"), None, &u.u_norm, (1, 1, 1)));
    c.push(of.mv("output.weight", "xn", "logits", 0));
    // argmax on the GPU: am = maximum, am2 = index (bits in an f32). The full logits stay in "logits" for the verifications.
    assert_eq!(VOCAB % 256, 0);
    let u_am1 = of.u([VOCAB as u32, 0, 0, 0, 0, 0, 0, 0]);
    let u_am2 = of.u([(VOCAB / 256) as u32, 0, 0, 0, 0, 0, 0, 0]);
    c.push(of.op("argmax1", &[(0, a("logits"))], a("am_v"), Some(a("am_i")), &u_am1, ((VOCAB / 256) as u32, 1, 1)));
    c.push(of.op("argmax2", &[(0, a("am_v")), (1, a("am_i"))], a("am"), Some(a("am2")), &u_am2, (1, 1, 1)));
    (c, seg_ends)
}
