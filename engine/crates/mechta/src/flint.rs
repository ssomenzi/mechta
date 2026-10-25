//! Flint (RecursiveSI/flint, GGUF architecture `flint`) on the GPU, one token at a time: F3 of `PLAN-FLINT.md`.
//! Separate from the Qwen engine (`engine.rs`): Flint has attention in every layer (35 sliding with RoPE, 14 full
//! without position), per-head sinks, 32 query / 4 KV heads of 128, a sigmoid router with 6 of 256 experts, a shared
//! expert without gate and two dense layers. Graph as llama.cpp `src/models/flint.cpp`.
//! Reused: the matvec of the quant types (matvec4.wgsl), rmsnorm / sum / swiglu (ops.wgsl). New: flint_*.wgsl.
//! Everything in f32 (activations and KV cache): this is the correctness step, not the fast one.
use crate::engine::{binding_op, mv_piece_geometry, ops_source, q3_source, KvFormat};
use mechta_gguf::{Gguf, TType, Value};
use mechta_gpu::{wgpu, Command, Gpu};
use std::collections::HashMap;

pub const HIDDEN: usize = 2048;
pub const N_HEAD: usize = 32;
pub const N_HEAD_KV: usize = 4;
pub const HEAD_DIM: usize = 128;
pub const N_EXPERT: usize = 256;
pub const N_EXPERT_USED: usize = 6;
pub const EXPERT_FF: usize = 512;
pub const SHARED_FF: usize = 1024;
pub const DENSE_FF: usize = 6144;
pub const VOCAB: usize = 150_144;
/// Positions a token of a sliding layer sees: the current one and the 512 before it (llama.cpp: `p1 - p0 < n_swa`).
pub const WINDOW: usize = 513;
/// Rows of the K/V ring of a sliding layer (row = position % RING_SWA): a prefill block of NB tokens writes all its
/// positions before its attention, so the ring must hold the window of the first token plus the block (513 + 511).
pub const RING_SWA: usize = 1024;
/// Tokens of a prefill block (as the Qwen engine).
pub const NB: usize = 512;
/// Groups of 8 tokens per attention dispatch of the prefill (a dispatch = one submit, to keep the GPU shared).
const GPD: usize = 16;
/// Width of a K or V row in the cache: 4 KV heads x 128.
const KV_ROW: usize = N_HEAD_KV * HEAD_DIM;

/// The hyperparameters read from the GGUF (the fixed sizes above are checked against it).
pub struct Conf { pub n_layer: usize, pub swa: Vec<bool>, pub rope: Vec<bool>, pub n_dense: usize, pub eps: f32, pub rope_base: f32 }

impl Conf {
    pub fn da_gguf(g: &Gguf) -> Self {
        let m = |k: &str| g.meta.get(&format!("flint.{k}")).unwrap_or_else(|| panic!("flint.{k} missing"));
        let u = |k: &str| m(k).as_u64().unwrap_or_else(|| panic!("flint.{k}")) as usize;
        let f = |k: &str| m(k).as_f32().unwrap_or_else(|| panic!("flint.{k}"));
        let bools = |k: &str| match m(k) { Value::Arr(v) => v.iter().map(|x| matches!(x, Value::Bool(true))).collect::<Vec<_>>(), _ => panic!("flint.{k}") };
        assert_eq!(g.meta.get("general.architecture").and_then(Value::as_str), Some("flint"), "not a Flint GGUF");
        for (k, v) in [("embedding_length", HIDDEN), ("attention.head_count", N_HEAD), ("attention.head_count_kv", N_HEAD_KV),
                       ("attention.key_length", HEAD_DIM), ("attention.value_length", HEAD_DIM), ("expert_count", N_EXPERT),
                       ("expert_used_count", N_EXPERT_USED), ("expert_feed_forward_length", EXPERT_FF),
                       ("expert_shared_feed_forward_length", SHARED_FF), ("feed_forward_length", DENSE_FF), ("attention.sliding_window", WINDOW)] {
            assert_eq!(u(k), v, "flint.{k}");
        }
        assert_eq!(u("expert_gating_func"), 2, "router: only sigmoid (2)");
        assert!(matches!(m("expert_weights_norm"), Value::Bool(true)), "expert weights: only normalized");
        assert_eq!(f("expert_weights_scale"), 1.0, "expert weights: only scale 1");
        let n_layer = u("block_count");
        let (swa, rope) = (bools("attention.sliding_window_pattern"), bools("attention.rope_pattern"));
        assert!(swa.len() == n_layer && rope.len() == n_layer);
        Conf { n_layer, swa, rope, n_dense: u("leading_dense_block_count"), eps: f("attention.layer_norm_rms_epsilon"), rope_base: f("rope.freq_base") }
    }
}

/// cos of the 64 pairs, then sin, for `pos`: as ggml_rope_cache_init (theta multiplied in f32, freq_scale 1, no YaRN).
pub fn rope_table(pos: usize, base: f32) -> Vec<f32> {
    let scale = base.powf(-2.0 / HEAD_DIM as f32);
    let mut theta = pos as f32;
    let (mut c, mut s) = (Vec::with_capacity(64), Vec::with_capacity(64));
    for _ in 0..HEAD_DIM / 2 { c.push(theta.cos()); s.push(theta.sin()); theta *= scale; }
    c.extend(s);
    c
}

/// K/V cache format: f32 (default) or, with MECHTA_FLINT_KV=f16, two halves per u32
/// as the f16 cache of Qwen (pack2x16float / unpack2x16float): half the bytes read per token and half the memory.
pub fn kv_f16() -> bool { std::env::var("MECHTA_FLINT_KV").is_ok_and(|v| v == "f16") }
/// Bytes of one K or V element in the cache.
pub fn kv_byte() -> usize { if kv_f16() { 2 } else { 4 } }
const KV_F32: &str = "@group(0) @binding(9) var<storage, read> kcr: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> vcr: array<vec4<f32>>;
@group(0) @binding(11) var<storage, read_write> kcw: array<vec2<f32>>;
@group(0) @binding(12) var<storage, read_write> vcw: array<vec2<f32>>;
fn k4(i: u32) -> vec4<f32> { return kcr[i]; }
fn v4(i: u32) -> vec4<f32> { return vcr[i]; }
fn kv_put(i: u32, k: vec2<f32>, v: vec2<f32>) { kcw[i] = k; vcw[i] = v; }
";
const KV_F16: &str = "@group(0) @binding(9) var<storage, read> kcr: array<vec2<u32>>;
@group(0) @binding(10) var<storage, read> vcr: array<vec2<u32>>;
@group(0) @binding(11) var<storage, read_write> kcw: array<u32>;
@group(0) @binding(12) var<storage, read_write> vcw: array<u32>;
fn k4(i: u32) -> vec4<f32> { let w = kcr[i]; return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn v4(i: u32) -> vec4<f32> { let w = vcr[i]; return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }
fn kv_put(i: u32, k: vec2<f32>, v: vec2<f32>) { kcw[i] = pack2x16float(k); vcw[i] = pack2x16float(v); }
";
/// flint_attn.wgsl with the block of the K/V format (`kv_f16`), and the prefill attention keeping Q, K, V in f32 in
/// shared memory (36 KB per workgroup; with packed f16, as attn_fa of Qwen, the K/V cache drifts to an error of ~1 by
/// layer 46, with f32 to 4e-4). The file is written for packed f16: the three replacements below make it f32.
pub fn attn_source() -> String { attn_source_for(kv_f16()) }
/// As `attn_source`, with the cache format given (the shader check compiles both).
pub fn attn_source_for(f16: bool) -> String {
    let base = include_str!("wgsl/flint_attn.wgsl");
    let sign_bit = "// KV-FORMAT:";
    let i = base.find(sign_bit).expect("KV-FORMAT");
    let s = format!("{}{}{}", &base[..i], if f16 { KV_F16 } else { KV_F32 }, &base[i..]);
    let r = |s: String, a: &str, b: &str| { assert!(s.contains(a), "{a}"); s.replace(a, b) };
    let s = r(s, "var<workgroup> fab: array<vec2<u32>, 1152>;", "var<workgroup> fab: array<vec4<f32>, 1152>;");
    let s = r(s, "fn pk4(v: vec4<f32>) -> vec2<u32> { return vec2<u32>(pack2x16float(v.xy), pack2x16float(v.zw)); }", "fn pk4(v: vec4<f32>) -> vec4<f32> { return v; }");
    r(s, "fn up4(w: vec2<u32>) -> vec4<f32> { return vec4<f32>(unpack2x16float(w.x), unpack2x16float(w.y)); }", "fn up4(w: vec4<f32>) -> vec4<f32> { return w; }")
}


/// K/V of a prefix (see `Flint::read_root`).
pub struct FlintRoot { pub p: usize, pub kv: Vec<(Vec<f32>, Vec<f32>)> }

pub struct Flint {
    pub gpu: Gpu,
    pub conf: Conf,
    g: Gguf,
    /// activations by name
    a: HashMap<&'static str, wgpu::Buffer>,
    /// per layer: (n, slot, lo, ring, ctx) of the attention, rewritten at every token
    u_att: Vec<wgpu::Buffer>,
    kc: Vec<wgpu::Buffer>, vc: Vec<wgpu::Buffer>,
    /// photo of the sliding rings (F5): (layer, K copy, V copy), taken where the next request will start again
    photo: Vec<(usize, wgpu::Buffer, wgpu::Buffer)>,
    /// prefill (F4): activations of a block, program, submit cuts, uniforms rewritten at every block
    ap: HashMap<&'static str, wgpu::Buffer>,
    prog_n: Vec<Command>, cuts_n: Vec<usize>,
    u_pn: wgpu::Buffer, u_swa_n: Vec<wgpu::Buffer>, u_full_n: Vec<wgpu::Buffer>, tab_n: wgpu::Buffer,
    /// longest submit of the last prefill (ms of waiting): the desktop shares the GPU
    pub submit_max: std::cell::Cell<f64>,
    tab: wgpu::Buffer,
    /// the program of one token, and the index where each layer ends
    prog: Vec<Command>,
    pub end: Vec<usize>,
    pub ctx: usize,
    pub byte_gpu: u64,
}

struct Factory<'a> {
    gpu: &'a Gpu, g: &'a Gguf,
    w: HashMap<String, wgpu::Buffer>, a: &'a HashMap<&'static str, wgpu::Buffer>,
    mv: HashMap<&'static str, wgpu::ComputePipeline>, op: HashMap<&'static str, wgpu::ComputePipeline>,
    fl: HashMap<&'static str, wgpu::ComputePipeline>,
    gm: HashMap<&'static str, wgpu::ComputePipeline>,
    empty: wgpu::Buffer, id0: wgpu::Buffer,
}

impl Factory<'_> {
    fn a(&self, n: &str) -> &wgpu::Buffer { self.a.get(n).unwrap_or_else(|| panic!("activation {n}")) }
    fn u(&self, v: [u32; 8]) -> wgpu::Buffer { self.gpu.uniform(&v) }
    /// An op of ops.wgsl, with the bindings that entry point uses.
    fn op(&self, e: &'static str, inputs: &[(u32, &wgpu::Buffer)], o: &wgpu::Buffer, u: &wgpu::Buffer, g: (u32, u32, u32)) -> Command {
        let b: Vec<(u32, &wgpu::Buffer)> = binding_op(e).iter().map(|&i| (i, match i {
            5 => o, 7 => u, _ => inputs.iter().find(|(j, _)| *j == i).map(|(_, x)| *x).unwrap_or(&self.empty) })).collect();
        self.gpu.command(&self.op[e], &b, g, e.to_string(), 0)
    }
    fn fl(&self, e: &'static str, b: &[(u32, &wgpu::Buffer)], g: (u32, u32, u32)) -> Command { self.gpu.command(&self.fl[e], b, g, e.to_string(), 0) }
    /// y = W x. Experts (`_exps`): the 6 matrices chosen by `ids`, x of expert z at z * x_step, y at z * rows.
    fn mv(&self, name: &str, x: &str, y: &str, x_step: u32) -> Command {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}"));
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        let experts = name.contains("_exps");
        let e = match t.ttype { TType::Q8_0 => "mv_q8_0", TType::Q4_K => "mv_q4_k", TType::Q6_K => "mv_q6_k", TType::Q4_1 => "mv_q4_1", TType::F32 | TType::BF16 => "mv_f32", x => panic!("{name}: {x:?}") };
        let (tpr, nx, ny) = mv_piece_geometry(rows, k, 32);
        let u = self.u([rows as u32, k as u32, nx, experts as u32, x_step, tpr, 0, 0]);
        let ids = if experts { self.a("ids") } else { &self.id0 };
        let nz = if experts { N_EXPERT_USED as u32 } else { 1 };
        self.gpu.command(&self.mv[e], &[(0, &self.w[name]), (1, self.a(x)), (2, self.a(y)), (3, ids), (4, &u)], (nx, ny, nz), name.to_string(), t.nbytes)
    }
    /// Prefill: Y = W X with X of nb columns (token t at t * k), dense weights read once per block (gemm2 of gemm.wgsl).
    fn gemm(&self, name: &str, x: &str, y: &str, nb: usize) -> Command {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}"));
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        let e = match t.ttype { TType::Q8_0 => "gemm2_q8_0", TType::Q4_K => "gemm2_q4_k", TType::Q6_K => "gemm2_q6_k", TType::Q4_1 => "gemm2_q4_1", TType::F32 | TType::BF16 => "gemm2_f32", x => panic!("{name}: {x:?}") };
        let u = self.u([rows as u32, k as u32, 0, 0, 0, 0, nb as u32, k as u32]);
        self.gpu.command(&self.gm[e], &[(0, &self.w[name]), (1, self.a(x)), (2, self.a(y)), (3, &self.id0), (4, &u)], ((rows as u32).div_ceil(128), (nb as u32).div_ceil(64), 1), format!("{name} x{nb}"), t.nbytes)
    }
    /// Prefill experts: grid z = 256 experts, columns = the tokens of the block that chose the expert (list `grp` of
    /// gather). x of token t, slot s at t * x_tok + s * x_step; y at (t * 8 + s) * rows (8 slots per token).
    fn gemm_e(&self, name: &str, x: &str, y: &str, x_step: u32, x_tok: u32, nb: usize) -> Command {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}"));
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        let e = match t.ttype { TType::Q4_K => "gemmr5_q4_k", TType::Q6_K => "gemmr5_q6_k", TType::Q8_0 => "gemmr5_q8_0", TType::Q4_1 => "gemmr5_q4_1", x => panic!("{name}: {x:?}") };
        let u = self.u([rows as u32, k as u32, 0, 2, x_step, 0, nb as u32, x_tok]);
        self.gpu.command(&self.gm[e], &[(0, &self.w[name]), (1, self.a(x)), (2, self.a(y)), (3, self.a("grp")), (4, &u)], ((rows as u32).div_ceil(128), (nb as u32).div_ceil(16), N_EXPERT as u32), format!("experts {name} x{nb}"), t.nbytes)
    }
}

impl Flint {
    pub fn new(path: &str, ctx: usize) -> Self {
        let g = Gguf::open(path).expect("GGUF");
        let conf = Conf::da_gguf(&g);
        let gpu = Gpu::with_memory(wgpu::MemoryHints::MemoryUsage);
        let src_mv = q3_source("matvec4");
        let mv = ["mv_q8_0", "mv_q4_k", "mv_q6_k", "mv_q4_1", "mv_f32"].into_iter().map(|e| (e, gpu.pipeline(&src_mv, e))).collect();
        let src_op = ops_source(KvFormat::F32);
        let op = ["rmsnorm", "sum", "swiglu"].into_iter().map(|e| (e, gpu.pipeline(&src_op, e))).collect();
        let mut fl: HashMap<&'static str, wgpu::ComputePipeline> = HashMap::new();
        let src_attn = attn_source();
        if kv_f16() { eprintln!("Flint K/V cache: f16 (MECHTA_FLINT_KV=f16)"); }
        for e in ["rope_neox", "kv_write", "attn_fd", "attn_fd_merge", "rope_neox_n", "kv_write_n", "attn_pf"] { fl.insert(e, gpu.pipeline(&src_attn, e)); }
        for e in ["routing_sig", "routing_sig_n"] { fl.insert(e, gpu.pipeline(include_str!("wgsl/flint_moe.wgsl"), e)); }
        for e in ["moe_sum6", "moe_sum6_n"] { fl.insert(e, gpu.pipeline(include_str!("wgsl/flint_sum.wgsl"), e)); }
        fl.insert("gather", gpu.pipeline(include_str!("wgsl/routing.wgsl"), "gather"));
        let src_gm = q3_source("gemm");
        let gm = ["gemm2_q8_0", "gemm2_q4_k", "gemm2_q6_k", "gemm2_q4_1", "gemm2_f32", "gemmr5_q4_k", "gemmr5_q6_k", "gemmr5_q8_0", "gemmr5_q4_1"].into_iter().map(|e| (e, gpu.pipeline(&src_gm, e))).collect();

        // buffers the GPU writes first, while VRAM is empty (as the Qwen engine)
        let mut a = HashMap::new();
        for (n, k) in [("x", HIDDEN), ("h", HIDDEN), ("q", N_HEAD * HEAD_DIM), ("k", KV_ROW), ("v", KV_ROW), ("qr", N_HEAD * HEAD_DIM),
                       ("kr", KV_ROW), ("qn", N_HEAD * HEAD_DIM), ("kn", KV_ROW), ("att", N_HEAD * HEAD_DIM), ("ao", HIDDEN), ("x2", HIDDEN),
                       ("h2", HIDDEN), ("lg", N_EXPERT), ("probs", N_EXPERT), ("ids", 8), ("weights", 8),
                       ("eg", N_EXPERT_USED * EXPERT_FF), ("eu", N_EXPERT_USED * EXPERT_FF), ("esw", N_EXPERT_USED * EXPERT_FF), ("ed", N_EXPERT_USED * HIDDEN),
                       ("sg", SHARED_FF), ("su", SHARED_FF), ("ssw", SHARED_FF), ("sd", HIDDEN),
                       ("dg", DENSE_FF), ("du", DENSE_FF), ("dsw", DENSE_FF), ("fo", HIDDEN), ("xn", HIDDEN), ("logits", VOCAB),
                       ("sc", N_HEAD * ctx)] {
            a.insert(n, gpu.empty_buffer(n, k as u64 * 4));
        }
        // prefill block: same names, 8 expert slots per token (6 used) so Qwen's expert GEMMs and gather fit
        let mut ap = HashMap::new();
        for (n, k) in [("x", NB * HIDDEN), ("h", NB * HIDDEN), ("q", NB * N_HEAD * HEAD_DIM), ("k", NB * KV_ROW), ("v", NB * KV_ROW),
                       ("qn", NB * N_HEAD * HEAD_DIM), ("kn", NB * KV_ROW), ("qr", NB * N_HEAD * HEAD_DIM), ("kr", NB * KV_ROW),
                       ("att", NB * N_HEAD * HEAD_DIM), ("ao", NB * HIDDEN), ("x2", NB * HIDDEN), ("h2", NB * HIDDEN),
                       ("lg", NB * N_EXPERT), ("probs", NB * N_EXPERT), ("ids", NB * 8), ("weights", NB * 8), ("grp", 512 + NB * 8),
                       ("eg", NB * 8 * EXPERT_FF), ("eu", NB * 8 * EXPERT_FF), ("esw", NB * 8 * EXPERT_FF), ("ed", NB * 8 * HIDDEN),
                       ("sg", NB * SHARED_FF), ("su", NB * SHARED_FF), ("ssw", NB * SHARED_FF), ("sd", NB * HIDDEN),
                       ("dg", NB * DENSE_FF), ("du", NB * DENSE_FF), ("dsw", NB * DENSE_FF), ("fo", NB * HIDDEN)] {
            ap.insert(n, gpu.empty_buffer(n, k as u64 * 4));
        }
        let (mut kc, mut vc) = (vec![], vec![]);
        for il in 0..conf.n_layer {
            let rows = if conf.swa[il] { RING_SWA } else { ctx };
            kc.push(gpu.empty_buffer("kc", (rows * KV_ROW * kv_byte()) as u64));
            vc.push(gpu.empty_buffer("vc", (rows * KV_ROW * kv_byte()) as u64));
        }
        let photo: Vec<(usize, wgpu::Buffer, wgpu::Buffer)> = (0..conf.n_layer).filter(|&il| conf.swa[il])
            .map(|il| (il, gpu.empty_buffer("foto-k", (RING_SWA * KV_ROW * kv_byte()) as u64), gpu.empty_buffer("foto-v", (RING_SWA * KV_ROW * kv_byte()) as u64))).collect();
        let u_att: Vec<wgpu::Buffer> = (0..conf.n_layer).map(|_| gpu.uniform(&[0u32; 8])).collect();
        let tab = gpu.empty_buffer("rope", (HEAD_DIM * 4) as u64);
        let tab_n = gpu.empty_buffer("rope-n", (NB * HEAD_DIM * 4) as u64);
        let u_pn = gpu.uniform(&[0u32; 4]);
        let u_swa_n: Vec<wgpu::Buffer> = (0..NB / 8 / GPD).map(|_| gpu.uniform(&[0u32; 8])).collect();
        let u_full_n: Vec<wgpu::Buffer> = (0..NB / 8 / GPD).map(|_| gpu.uniform(&[0u32; 8])).collect();

        // weights (token_embd stays on the CPU: one row per token)
        let mem0 = crate::tools::gpu_memory();
        let mut w = HashMap::new(); let mut byte_gpu = 0u64;
        for t in g.tensors.clone() {
            if t.name == "token_embd.weight" { continue; }
            let mut v = Vec::new(); g.read(&t, 0, t.nbytes, &mut v).expect("read");
            if t.ttype == TType::BF16 { v = v.chunks_exact(2).flat_map(|c| (u32::from(u16::from_le_bytes([c[0], c[1]])) << 16).to_le_bytes()).collect(); }
            byte_gpu += v.len() as u64;
            w.insert(t.name.clone(), gpu.buffer_byte(&t.name, &v));
            gpu.queue.submit([]);
            gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
            // safety, as the Qwen engine: GPU memory must not grow more than the weights + 3 GB
            let (v1, g1) = crate::tools::gpu_memory();
            let grown = (v1 - mem0.0) + (g1 - mem0.1);
            assert!(grown < byte_gpu as f64 / 1e9 + 3.0, "GPU memory grew {grown:.2} GB for {:.2} GB of weights: stopping", byte_gpu as f64 / 1e9);
        }
        let empty = gpu.empty_buffer("empty", 16);
        let id0 = gpu.empty_buffer("id0", 32);
        let of = Factory { gpu: &gpu, g: &g, w, a: &a, mv, op, fl, gm, empty, id0 };
        let (prog, end) = program(&of, &conf, &kc, &vc, &u_att, &tab);
        let ofn = Factory { a: &ap, ..of };
        let (prog_n, cuts_n) = program_n(&ofn, &conf, &kc, &vc, &u_pn, &u_swa_n, &u_full_n, &tab_n);
        drop(ofn);
        Flint { gpu, conf, g, a, u_att, kc, vc, photo, ap, prog_n, cuts_n, u_pn, u_swa_n, u_full_n, tab_n, submit_max: std::cell::Cell::new(0.0),
                tab, prog, end, ctx, byte_gpu }
    }

    /// Row `tok` of token_embd decoded on the CPU: F32 as it is (the official INT4 GGUF), or Q8_0: d (f16) times the
    /// 32 int8 of each block, as ggml.
    fn embedding(&self, tok: u32) -> Vec<f32> {
        let t = self.g.tensor("token_embd.weight").expect("token_embd");
        if t.ttype == TType::F32 {
            let mut b = Vec::new(); self.g.read(t, tok as u64 * HIDDEN as u64 * 4, HIDDEN as u64 * 4, &mut b).expect("read");
            return b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        }
        assert_eq!(t.ttype, TType::Q8_0, "token_embd: only F32 or Q8_0");
        let row = (HIDDEN / 32 * 34) as u64;
        let mut b = Vec::new(); self.g.read(t, tok as u64 * row, row, &mut b).expect("read");
        let mut x = Vec::with_capacity(HIDDEN);
        for blk in b.chunks_exact(34) {
            let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
            x.extend(blk[2..].iter().map(|&q| d * (q as i8) as f32));
        }
        x
    }

    /// Writes the inputs of position `pos`: embedding of `tok` in x, RoPE table, (n, slot) of every layer.
    pub fn inputs(&self, tok: u32, pos: usize) {
        assert!(pos < self.ctx, "position {pos} beyond ctx {}", self.ctx);
        let b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        self.gpu.write(&self.a["x"], 0, &b(&self.embedding(tok)));
        self.gpu.write(&self.tab, 0, &b(&rope_table(pos, self.conf.rope_base)));
        for il in 0..self.conf.n_layer {
            let (lo, ring) = if self.conf.swa[il] { ((pos + 1).saturating_sub(WINDOW), RING_SWA) } else { (0, self.ctx) };
            let u = [(pos + 1 - lo) as u32, (pos % ring) as u32, lo as u32, ring as u32, self.ctx as u32, 0, 0, 0];
            self.gpu.write(&self.u_att[il], 0, &u.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
        }
    }

    /// K/V caches to zero (a test can check that a path never reads what another one wrote).
    pub fn reset_kv(&self) {
        for b in self.kc.iter().chain(&self.vc) { self.gpu.write(b, 0, &vec![0u8; b.size() as usize]); }
        self.gpu.wait();
    }

    /// Photo of the sliding rings (the full layers keep every position: nothing to copy there).
    pub fn photograph(&self) {
        let c: Vec<(&wgpu::Buffer, &wgpu::Buffer, u64)> = self.photo.iter().flat_map(|(il, k, v)| [(&self.kc[*il], k, k.size()), (&self.vc[*il], v, v.size())]).collect();
        self.gpu.copies(&c);
    }
    /// The rings back from the photo.
    pub fn restore_photo(&self) {
        let c: Vec<(&wgpu::Buffer, &wgpu::Buffer, u64)> = self.photo.iter().flat_map(|(il, k, v)| [(k, &self.kc[*il], k.size()), (v, &self.vc[*il], v.size())]).collect();
        self.gpu.copies(&c);
    }
    /// Root (F5): the caches when they cover exactly the first `p` positions: rows 0..p of the full layers, the whole
    /// ring of the sliding ones. In layer order.
    /// The words are kept as they are in the cache (f32 bit patterns: f32 values, or two f16 halves per word).
    pub fn read_root(&self, p: usize) -> FlintRoot {
        let kv = (0..self.conf.n_layer).map(|il| {
            let words = (if self.conf.swa[il] { RING_SWA } else { p }) * KV_ROW * kv_byte() / 4;
            (self.gpu.read_f32(&self.kc[il], words), self.gpu.read_f32(&self.vc[il], words))
        }).collect();
        FlintRoot { p, kv }
    }
    pub fn write_root(&self, r: &FlintRoot) {
        let b = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        for (il, (k, v)) in r.kv.iter().enumerate() { self.gpu.write(&self.kc[il], 0, &b(k)); self.gpu.write(&self.vc[il], 0, &b(v)); }
        self.gpu.wait();
    }

    /// K and V caches of layer `il`, rows 0..rows.
    pub fn read_kv(&self, il: usize, rows: usize) -> (Vec<f32>, Vec<f32>) {
        if !kv_f16() { return (self.gpu.read_f32(&self.kc[il], rows * KV_ROW), self.gpu.read_f32(&self.vc[il], rows * KV_ROW)); }
        let dec = |w: Vec<f32>| w.iter().flat_map(|x| { let b = x.to_bits(); [half::f16::from_bits(b as u16).to_f32(), half::f16::from_bits((b >> 16) as u16).to_f32()] }).collect::<Vec<f32>>();
        (dec(self.gpu.read_f32(&self.kc[il], rows * KV_ROW / 2)), dec(self.gpu.read_f32(&self.vc[il], rows * KV_ROW / 2)))
    }

    /// Prefill (F4): `toks` at positions pos0.. in blocks of NB, K/V into the caches, no logits. Submits: one per layer
    /// and one per attention dispatch (GPD groups of 8 tokens). Longest wait in `submit_max`.
    pub fn prefill(&self, toks: &[u32], pos0: usize) { self.prefill_cancellable(toks, pos0, &mut || true); }
    /// As `prefill`, asking `alive` between submits (at most every KEEPALIVE_EVERY) whether the client is still there; false =
    /// stop. Returns the tokens of the blocks completed (toks.len() when not stopped): the caches are valid up to there.
    pub fn prefill_cancellable(&self, toks: &[u32], pos0: usize, alive: &mut dyn FnMut() -> bool) -> usize {
        assert!(pos0 + toks.len() <= self.ctx, "prefill beyond ctx {}", self.ctx);
        let mut last = std::time::Instant::now();
        let by = |v: &[u32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let bf = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
        let mut smax = 0f64;
        for (ib, block) in toks.chunks(NB).enumerate() {
            let p0 = pos0 + ib * NB; let n = block.len();
            let mut x = Vec::with_capacity(n * HIDDEN); let mut tab = Vec::with_capacity(n * HEAD_DIM);
            for (t, &tok) in block.iter().enumerate() { x.extend(self.embedding(tok)); tab.extend(rope_table(p0 + t, self.conf.rope_base)); }
            self.gpu.write(&self.ap["x"], 0, &bf(&x));
            self.gpu.write(&self.tab_n, 0, &bf(&tab));
            self.gpu.write(&self.u_pn, 0, &by(&[n as u32, NB as u32, 0, 0]));
            for (d, (us, uf)) in self.u_swa_n.iter().zip(&self.u_full_n).enumerate() {
                let g0 = (d * GPD) as u32;
                self.gpu.write(us, 0, &by(&[n as u32, 0, 0, RING_SWA as u32, 0, p0 as u32, WINDOW as u32, g0]));
                self.gpu.write(uf, 0, &by(&[n as u32, 0, 0, self.ctx as u32, 0, p0 as u32, 0, g0]));
            }
            let mut da = 0;
            for &a in self.cuts_n.iter().chain(std::iter::once(&self.prog_n.len())) {
                if a <= da { continue; }
                // attention dispatches past the last group of a short block do nothing: not sent
                let (_, wait) = self.gpu.run_commands(&self.prog_n[da..a]);
                smax = smax.max(wait);
                da = a;
                if last.elapsed() >= crate::engine::KEEPALIVE_EVERY {
                    last = std::time::Instant::now();
                    if !alive() { self.submit_max.set(smax); return ib * NB; }
                }
            }
        }
        self.submit_max.set(smax);
        toks.len()
    }

    /// One token: logits.
    pub fn token(&self, tok: u32, pos: usize) -> Vec<f32> {
        self.inputs(tok, pos);
        self.gpu.run_commands(&self.prog);
        self.gpu.read_f32(&self.a["logits"], VOCAB)
    }

    /// The residual stream after the last layer (`l_out` of the last layer) of the last token.
    pub fn read_x(&self) -> Vec<f32> { self.gpu.read_f32(&self.a["x"], HIDDEN) }

    /// One token, layer by layer: cb(il, ffn_inp, l_out) after each layer, then the logits.
    pub fn token_layers(&self, tok: u32, pos: usize, cb: &mut dyn FnMut(usize, Vec<f32>, Vec<f32>)) -> Vec<f32> {
        self.inputs(tok, pos);
        let mut da = 0;
        for il in 0..self.conf.n_layer {
            self.gpu.run_commands(&self.prog[da..self.end[il]]);
            da = self.end[il];
            cb(il, self.gpu.read_f32(&self.a["x2"], HIDDEN), self.gpu.read_f32(&self.a["x"], HIDDEN));
        }
        self.gpu.run_commands(&self.prog[da..]);
        self.gpu.read_f32(&self.a["logits"], VOCAB)
    }
}

/// The program of one token: 49 layers, then output_norm and the output head into "logits".
fn program(of: &Factory, conf: &Conf, kc: &[wgpu::Buffer], vc: &[wgpu::Buffer], u_att: &[wgpu::Buffer], tab: &wgpu::Buffer) -> (Vec<Command>, Vec<usize>) {
    let a = |n: &str| of.a(n);
    let (eps, one) = (conf.eps.to_bits(), 1f32.to_bits());
    let u_norm = of.u([HIDDEN as u32, 1, eps, one, HIDDEN as u32, HIDDEN as u32, 1, 0]);
    let u_qn = of.u([HEAD_DIM as u32, N_HEAD as u32, eps, one, HEAD_DIM as u32, HEAD_DIM as u32, 1, 0]);
    let u_kn = of.u([HEAD_DIM as u32, N_HEAD_KV as u32, eps, one, HEAD_DIM as u32, HEAD_DIM as u32, 1, 0]);
    let u_n = |n: usize| of.u([n as u32, 0, 0, 0, 0, 0, 0, 0]);
    let (u_h, u_e, u_s, u_d) = (u_n(HIDDEN), u_n(N_EXPERT_USED * EXPERT_FF), u_n(SHARED_FF), u_n(DENSE_FF));
    let g64 = |n: usize| ((n as u32).div_ceil(64), 1, 1);
    let mut c = vec![];
    let mut end = vec![];
    for il in 0..conf.n_layer {
        let w = |s: &str| &of.w[&format!("blk.{il}.{s}")];
        let l = |s: &str| format!("blk.{il}.{s}");
        // attention
        c.push(of.op("rmsnorm", &[(0, a("x")), (1, w("attn_norm.weight"))], a("h"), &u_norm, (1, 1, 1)));
        c.push(of.mv(&l("attn_q.weight"), "h", "q", 0));
        c.push(of.mv(&l("attn_k.weight"), "h", "k", 0));
        c.push(of.mv(&l("attn_v.weight"), "h", "v", 0));
        c.push(of.op("rmsnorm", &[(0, a("q")), (1, w("attn_q_norm.weight"))], a("qn"), &u_qn, (N_HEAD as u32, 1, 1)));
        c.push(of.op("rmsnorm", &[(0, a("k")), (1, w("attn_k_norm.weight"))], a("kn"), &u_kn, (N_HEAD_KV as u32, 1, 1)));
        let (qa, ka) = if conf.rope[il] {
            c.push(of.fl("rope_neox", &[(0, a("qn")), (1, tab), (5, a("qr"))], (N_HEAD as u32, 1, 1)));
            c.push(of.fl("rope_neox", &[(0, a("kn")), (1, tab), (5, a("kr"))], (N_HEAD_KV as u32, 1, 1)));
            ("qr", "kr")
        } else { ("qn", "kn") };
        c.push(of.fl("kv_write", &[(0, a(ka)), (1, a("v")), (11, &kc[il]), (12, &vc[il]), (7, &u_att[il])], g64(KV_ROW / 2)));
        // flash-decoding: pieces of 256 positions over the rows the layer can have (ring or ctx), the empty ones exit
        let rows = if conf.swa[il] { RING_SWA } else { of.a("sc").size() as usize / 4 / N_HEAD };
        c.push(of.fl("attn_fd", &[(0, a(qa)), (9, &kc[il]), (10, &vc[il]), (4, a("sc")), (7, &u_att[il])], (N_HEAD_KV as u32, rows.div_ceil(256) as u32, 1)));
        c.push(of.fl("attn_fd_merge", &[(3, w("attn_sinks.weight")), (4, a("sc")), (5, a("att")), (7, &u_att[il])], (N_HEAD as u32, 1, 1)));
        c.push(of.mv(&l("attn_output.weight"), "att", "ao", 0));
        c.push(of.op("sum", &[(0, a("ao")), (1, a("x"))], a("x2"), &u_h, g64(HIDDEN)));
        // feed-forward
        c.push(of.op("rmsnorm", &[(0, a("x2")), (1, w("ffn_norm.weight"))], a("h2"), &u_norm, (1, 1, 1)));
        if il < conf.n_dense {
            c.push(of.mv(&l("ffn_gate.weight"), "h2", "dg", 0));
            c.push(of.mv(&l("ffn_up.weight"), "h2", "du", 0));
            c.push(of.op("swiglu", &[(0, a("dg")), (1, a("du"))], a("dsw"), &u_d, g64(DENSE_FF)));
            c.push(of.mv(&l("ffn_down.weight"), "dsw", "fo", 0));
            c.push(of.op("sum", &[(0, a("fo")), (1, a("x2"))], a("x"), &u_h, g64(HIDDEN)));
        } else {
            c.push(of.mv(&l("ffn_gate_inp.weight"), "h2", "lg", 0));
            c.push(of.fl("routing_sig", &[(0, a("lg")), (1, a("probs")), (2, a("ids")), (3, a("weights"))], (1, 1, 1)));
            c.push(of.mv(&l("ffn_gate_exps.weight"), "h2", "eg", 0));
            c.push(of.mv(&l("ffn_up_exps.weight"), "h2", "eu", 0));
            c.push(of.op("swiglu", &[(0, a("eg")), (1, a("eu"))], a("esw"), &u_e, g64(N_EXPERT_USED * EXPERT_FF)));
            c.push(of.mv(&l("ffn_down_exps.weight"), "esw", "ed", EXPERT_FF as u32));
            c.push(of.mv(&l("ffn_gate_shexp.weight"), "h2", "sg", 0));
            c.push(of.mv(&l("ffn_up_shexp.weight"), "h2", "su", 0));
            c.push(of.op("swiglu", &[(0, a("sg")), (1, a("su"))], a("ssw"), &u_s, g64(SHARED_FF)));
            c.push(of.mv(&l("ffn_down_shexp.weight"), "ssw", "sd", 0));
            c.push(of.fl("moe_sum6", &[(0, a("ed")), (1, a("weights")), (2, a("sd")), (3, a("x2")), (4, a("x"))], g64(HIDDEN)));
        }
        end.push(c.len());
    }
    c.push(of.op("rmsnorm", &[(0, a("x")), (1, &of.w["output_norm.weight"])], a("xn"), &u_norm, (1, 1, 1)));
    c.push(of.mv("output.weight", "xn", "logits", 0));
    (c, end)
}

/// The program of a prefill block (NB columns, the block length is in the uniforms): 49 layers, no head. Returns the
/// commands and the submit cuts (end of each layer, after each attention dispatch).
#[allow(clippy::too_many_arguments)]
fn program_n(of: &Factory, conf: &Conf, kc: &[wgpu::Buffer], vc: &[wgpu::Buffer], u_pn: &wgpu::Buffer,
               u_swa_n: &[wgpu::Buffer], u_full_n: &[wgpu::Buffer], tab_n: &wgpu::Buffer) -> (Vec<Command>, Vec<usize>) {
    let a = |n: &str| of.a(n);
    let nb = NB as u32;
    let (eps, one) = (conf.eps.to_bits(), 1f32.to_bits());
    let u_norm = of.u([HIDDEN as u32, 1, eps, one, HIDDEN as u32, HIDDEN as u32, 1, 0]);
    let u_qn = of.u([HEAD_DIM as u32, (NB * N_HEAD) as u32, eps, one, HEAD_DIM as u32, HEAD_DIM as u32, 1, 0]);
    let u_kn = of.u([HEAD_DIM as u32, (NB * N_HEAD_KV) as u32, eps, one, HEAD_DIM as u32, HEAD_DIM as u32, 1, 0]);
    let u_n = |n: usize| of.u([n as u32, 0, 0, 0, 0, 0, 0, 0]);
    let (u_h, u_e, u_s, u_d) = (u_n(NB * HIDDEN), u_n(NB * 8 * EXPERT_FF), u_n(NB * SHARED_FF), u_n(NB * DENSE_FF));
    let g64 = |n: usize| ((n as u32).div_ceil(64), 1, 1);
    let mut c = vec![]; let mut cuts = vec![];
    for il in 0..conf.n_layer {
        let w = |s: &str| &of.w[&format!("blk.{il}.{s}")];
        let l = |s: &str| format!("blk.{il}.{s}");
        let ua = if conf.swa[il] { u_swa_n } else { u_full_n };
        c.push(of.op("rmsnorm", &[(0, a("x")), (1, w("attn_norm.weight"))], a("h"), &u_norm, (nb, 1, 1)));
        c.push(of.gemm(&l("attn_q.weight"), "h", "q", NB));
        c.push(of.gemm(&l("attn_k.weight"), "h", "k", NB));
        c.push(of.gemm(&l("attn_v.weight"), "h", "v", NB));
        c.push(of.op("rmsnorm", &[(0, a("q")), (1, w("attn_q_norm.weight"))], a("qn"), &u_qn, (nb * N_HEAD as u32, 1, 1)));
        c.push(of.op("rmsnorm", &[(0, a("k")), (1, w("attn_k_norm.weight"))], a("kn"), &u_kn, (nb * N_HEAD_KV as u32, 1, 1)));
        let (qa, ka) = if conf.rope[il] {
            c.push(of.fl("rope_neox_n", &[(0, a("qn")), (1, tab_n), (5, a("qr"))], (N_HEAD as u32, nb, 1)));
            c.push(of.fl("rope_neox_n", &[(0, a("kn")), (1, tab_n), (5, a("kr"))], (N_HEAD_KV as u32, nb, 1)));
            ("qr", "kr")
        } else { ("qn", "kn") };
        c.push(of.fl("kv_write_n", &[(0, a(ka)), (1, a("v")), (11, &kc[il]), (12, &vc[il]), (7, &ua[0])], ((KV_ROW / 2 / 64) as u32, nb, 1)));
        for u in ua {
            cuts.push(c.len());
            c.push(of.fl("attn_pf", &[(0, a(qa)), (9, &kc[il]), (10, &vc[il]), (3, w("attn_sinks.weight")), (5, a("att")), (7, u)], (N_HEAD_KV as u32, 1, GPD as u32)));
        }
        cuts.push(c.len());
        c.push(of.gemm(&l("attn_output.weight"), "att", "ao", NB));
        c.push(of.op("sum", &[(0, a("ao")), (1, a("x"))], a("x2"), &u_h, g64(NB * HIDDEN)));
        c.push(of.op("rmsnorm", &[(0, a("x2")), (1, w("ffn_norm.weight"))], a("h2"), &u_norm, (nb, 1, 1)));
        if il < conf.n_dense {
            c.push(of.gemm(&l("ffn_gate.weight"), "h2", "dg", NB));
            c.push(of.gemm(&l("ffn_up.weight"), "h2", "du", NB));
            c.push(of.op("swiglu", &[(0, a("dg")), (1, a("du"))], a("dsw"), &u_d, g64(NB * DENSE_FF)));
            c.push(of.gemm(&l("ffn_down.weight"), "dsw", "fo", NB));
            c.push(of.op("sum", &[(0, a("fo")), (1, a("x2"))], a("x"), &u_h, g64(NB * HIDDEN)));
        } else {
            c.push(of.gemm(&l("ffn_gate_inp.weight"), "h2", "lg", NB));
            c.push(of.fl("routing_sig_n", &[(0, a("lg")), (1, a("probs")), (2, a("ids")), (3, a("weights")), (4, u_pn)], (nb, 1, 1)));
            c.push(of.fl("gather", &[(2, a("ids")), (4, u_pn), (5, a("grp"))], (1, 1, 1)));
            c.push(of.gemm_e(&l("ffn_gate_exps.weight"), "h2", "eg", 0, HIDDEN as u32, NB));
            c.push(of.gemm_e(&l("ffn_up_exps.weight"), "h2", "eu", 0, HIDDEN as u32, NB));
            c.push(of.op("swiglu", &[(0, a("eg")), (1, a("eu"))], a("esw"), &u_e, g64(NB * 8 * EXPERT_FF)));
            c.push(of.gemm_e(&l("ffn_down_exps.weight"), "esw", "ed", EXPERT_FF as u32, (8 * EXPERT_FF) as u32, NB));
            c.push(of.gemm(&l("ffn_gate_shexp.weight"), "h2", "sg", NB));
            c.push(of.gemm(&l("ffn_up_shexp.weight"), "h2", "su", NB));
            c.push(of.op("swiglu", &[(0, a("sg")), (1, a("su"))], a("ssw"), &u_s, g64(NB * SHARED_FF)));
            c.push(of.gemm(&l("ffn_down_shexp.weight"), "ssw", "sd", NB));
            c.push(of.fl("moe_sum6_n", &[(0, a("ed")), (1, a("weights")), (2, a("sd")), (3, a("x2")), (4, a("x")), (5, u_pn)], ((HIDDEN / 64) as u32, nb, 1)));
        }
        cuts.push(c.len());
    }
    (c, cuts)
}

/// `mechta flint-logit-dump <text> <prefill> <steps> <out>`: the tokens of the text
/// (Flint tokenizer) after `<|user|>`; prefill of the first `prefill` in blocks, then `steps` tokens of the text one at
/// a time, writing for each the true next token and the logits (the format of `logit-dump`; compare with
/// `MECHTA_KL_VOCAB=150144 mechta kl`). The generation path is the one a variant changes (cache format, attention).
pub fn logit_dump(text: &str, prefill: usize, steps: usize, out: &str) {
    use std::io::Write;
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let mut ids = vec![150_003u32];
    ids.extend(tk.encode(std::fs::read_to_string(text).expect("text"), false).expect("encode").get_ids());
    assert!(ids.len() > prefill + steps + 1, "text too short: {} tokens", ids.len());
    let ctx = (prefill + steps + 1).next_multiple_of(1024);
    let f = Flint::new(&crate::model_path(), ctx);
    let t = std::time::Instant::now();
    f.prefill(&ids[..prefill], 0);
    println!("kv={} prefill {prefill} tokens in {:.1} s", if kv_f16() { "f16" } else { "f32" }, t.elapsed().as_secs_f64());
    let mut w = std::io::BufWriter::new(std::fs::File::create(out).expect("out"));
    for j in 0..steps {
        let pos = prefill + j;
        let l = f.token(ids[pos], pos);
        w.write_all(&ids[pos + 1].to_le_bytes()).unwrap();
        for x in &l { w.write_all(&x.to_le_bytes()).unwrap(); }
    }
    println!("{steps} steps written to {out}");
}
