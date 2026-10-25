//! Step 3 of M1: the kernels of one layer, one at a time, against the llama.cpp dump.
//! Every operation receives its inputs FROM THE DUMP (not from the previous operation): this way an error
//! stays where it originates and does not add up. For the matrices it also computes, on the CPU in f64:
//!   - "exact": W (decoded with the kernel verified at step 2) times x in f32;
//!   - "ggml":   the same, but with x quantized the way the ggml CPU does it (Q8_0 or Q8_K).
//! Error = max |a - b| / max |b|.
use half::f16;
use mechta_gguf::{Gguf, TType};
use mechta_gpu::{Gpu, wgpu};
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

/// reference dumps of llama.cpp, in the workspace (paths fixed at build time: verification tools, not the server)
const DUMP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../reference/tensors.bin");
const TOKEN: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../reference/token.tsv");
const EPS: f32 = 1e-6;

/// Index of tensors.bin: (name, position) -> records in order of appearance.
pub(crate) struct Dump { f: std::fs::File, idx: HashMap<(String, i32), Vec<(u64, [i64; 4])>> }
impl Dump {
    pub(crate) fn load() -> Self { Self::load_file(DUMP) }
    pub(crate) fn load_file(path: &str) -> Self {
        let mut f = std::fs::File::open(path).expect("tensors.bin");
        let len = f.metadata().unwrap().len();
        let mut idx: HashMap<(String, i32), Vec<(u64, [i64; 4])>> = HashMap::new();
        let mut pos = 0u64;
        while pos < len {
            let mut b4 = [0u8; 4]; f.read_exact(&mut b4).unwrap();
            let n = i32::from_le_bytes(b4) as usize;
            let mut name = vec![0u8; n]; f.read_exact(&mut name).unwrap();
            f.read_exact(&mut b4).unwrap(); let p = i32::from_le_bytes(b4);
            let mut ne = [0i64; 4];
            for v in ne.iter_mut() { let mut b8 = [0u8; 8]; f.read_exact(&mut b8).unwrap(); *v = i64::from_le_bytes(b8); }
            let data = pos + 4 + n as u64 + 4 + 32;
            let sz = (ne.iter().product::<i64>() * 4) as u64;
            idx.entry((String::from_utf8(name).unwrap(), p)).or_default().push((data, ne));
            pos = data + sz;
            f.seek(SeekFrom::Start(pos)).unwrap();
        }
        Dump { f, idx }
    }
    /// occ = which occurrence of the name at that position (0 = the first).
    pub(crate) fn get(&mut self, name: &str, p: i32, occ: usize) -> Vec<f32> {
        let (off, ne) = *self.idx.get(&(name.to_string(), p)).unwrap_or_else(|| panic!("{name}@{p} not in the dump")).get(occ).unwrap_or_else(|| panic!("{name}@{p} occ {occ}"));
        let n = ne.iter().product::<i64>() as usize;
        let mut b = vec![0u8; n * 4];
        self.f.seek(SeekFrom::Start(off)).unwrap(); self.f.read_exact(&mut b).unwrap();
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
}

pub(crate) fn err(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len(), "different lengths");
    // f64::max ignores NaN: without this check an all-NaN output would give error 0 (seen in the delta bench)
    if a.iter().chain(b).any(|v| !v.is_finite()) { return f64::NAN; }
    let mx = b.iter().fold(0f64, |m, v| m.max((*v as f64).abs())).max(1e-30);
    a.iter().zip(b).fold(0f64, |m, (x, y)| m.max((*x as f64 - *y as f64).abs())) / mx
}

/// x quantized as the ggml CPU does before the product with Q8_0 weights (blocks of 32) or K weights (blocks of 256).
fn quantize_x(x: &[f32], kind: TType) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    match kind {
        TType::Q8_0 => for (bi, bl) in x.chunks(32).enumerate() {
            let amax = bl.iter().fold(0f32, |m, v| m.max(v.abs()));
            let d = amax / 127.0; let id = if d != 0.0 { 1.0 / d } else { 0.0 };
            let dh = f16::from_f32(d).to_f32();
            for (j, v) in bl.iter().enumerate() { out[bi * 32 + j] = (v * id).round() * dh; }
        },
        TType::Q4_K | TType::Q6_K => for (bi, bl) in x.chunks(256).enumerate() {
            let (mut amax, mut max) = (0f32, 0f32);
            for v in bl { if v.abs() > amax { amax = v.abs(); max = *v; } }
            if amax == 0.0 { continue; }
            let iscale = -127.0 / max;
            for (j, v) in bl.iter().enumerate() { out[bi * 256 + j] = (iscale * v).round_ties_even().min(127.0) * (1.0 / iscale); }
        },
        _ => out.copy_from_slice(x),
    }
    out
}

/// Bench state: model tensors on the GPU, pipelines, dump.
struct Bench { g: Gguf, gpu: Gpu, mv: HashMap<&'static str, wgpu::ComputePipeline>, op: HashMap<&'static str, wgpu::ComputePipeline>, rt: wgpu::ComputePipeline, dq: HashMap<&'static str, wgpu::ComputePipeline>, d: Dump, empty: wgpu::Buffer, weights: HashMap<String, wgpu::Buffer>, results: Vec<(String, usize, f64, Option<(f64, f64, f64)>)> }

impl Bench {
    fn byte(&self, name: &str) -> (Vec<u8>, mechta_gguf::Tensor) {
        let t = self.g.tensor(name).unwrap_or_else(|| panic!("{name}")).clone();
        let mut v = Vec::new(); self.g.read(&t, 0, t.nbytes, &mut v).unwrap(); (v, t)
    }
    fn f32_model(&self, name: &str) -> Vec<f32> {
        let (v, t) = self.byte(name); assert_eq!(t.ttype, TType::F32, "{name}");
        v.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }
    fn weight(&mut self, name: &str) -> &wgpu::Buffer {
        if !self.weights.contains_key(name) {
            let (v, _) = self.byte(name);
            let b = self.gpu.buffer_byte(name, &v);
            self.weights.insert(name.to_string(), b);
        }
        &self.weights[name]
    }
    /// Decoding on the GPU (kernel of step 2) of `nb` blocks of the tensor starting at block `b0`.
    fn deq(&self, t: &mechta_gguf::Tensor, b0: u64, nb: u64) -> Vec<f32> {
        if t.ttype == TType::F32 {
            let mut v = Vec::new(); self.g.read(t, b0 * 4, nb * 4, &mut v).unwrap();
            return v.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        }
        let (per, bb) = t.ttype.block().unwrap();
        let mut raw = Vec::new(); self.g.read(t, b0 * bb, nb * bb, &mut raw).unwrap();
        let e = match t.ttype { TType::Q8_0 => "q8_0", TType::Q4_K => "q4_k", TType::Q6_K => "q6_k", TType::Q4_1 => "q4_1", x => panic!("{x:?}") };
        let br = self.gpu.buffer_byte("raw", &raw); let bo = self.gpu.empty_buffer("deq", nb * per * 4);
        let u = self.gpu.uniform(&[nb as u32, 0, 0, 0]);
        self.gpu.run(&self.dq[e], &[&br, &bo, &u], ((nb as u32).div_ceil(64), 1, 1));
        self.gpu.read_f32(&bo, (nb * per) as usize)
    }

    fn annotate(&mut self, name: &str, gpu: &[f32], reference: &[f32], cpu: Option<(f64, f64, f64)>) {
        let e = err(gpu, reference);
        self.results.push((name.to_string(), gpu.len(), e, cpu));
    }

    /// y = W x on the GPU. mats = matrices used (one, or the experts `ids`). Comparison with the dump and
    /// with the f64 computation on the CPU (exact, and with x quantized as ggml does).
    fn matvec(&mut self, name_w: &str, x: &[f32], ids: Option<&[u32]>, x_step: usize, reference: &[f32], label: &str) -> Vec<f32> {
        let t = self.g.tensor(name_w).unwrap().clone();
        let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
        let nz = ids.map(|v| v.len()).unwrap_or(1);
        let e = match t.ttype { TType::Q8_0 => "mv_q8_0", TType::Q4_K => "mv_q4_k", TType::Q6_K => "mv_q6_k", TType::Q4_1 => "mv_q4_1", TType::F32 => "mv_f32", x => panic!("{x:?}") };
        let (tpr, nx, ny) = crate::engine::geometry_mv(rows, k);
        let bx = self.gpu.buffer_f32("x", x);
        let bid = match ids { Some(v) => self.gpu.buffer_u32("ids", v), None => self.gpu.buffer_u32("ids", &[0]) };
        let by = self.gpu.empty_buffer("y", (rows * nz * 4) as u64);
        let u = self.gpu.uniform(&[rows as u32, k as u32, nx, ids.is_some() as u32, x_step as u32, tpr, 0, 0]);
        self.weight(name_w);
        let w = &self.weights[name_w];
        self.gpu.run_b(&self.mv[e], &[(0, w), (1, &bx), (2, &by), (3, &bid), (4, &u)], (nx, ny, nz as u32));
        let y = self.gpu.read_f32(&by, rows * nz);
        // CPU in f64
        let (per, _) = t.ttype.block().unwrap();
        let bl_mat = (rows * k) as u64 / per;
        let (mut ex, mut gg) = (Vec::with_capacity(rows * nz), Vec::with_capacity(rows * nz));
        for z in 0..nz {
            let mat = ids.map(|v| v[z] as u64).unwrap_or(0);
            let wd = self.deq(&t, mat * bl_mat, bl_mat);
            let xz = &x[z * x_step..z * x_step + k];
            let xq = quantize_x(xz, t.ttype);
            for r in 0..rows {
                let row = &wd[r * k..(r + 1) * k];
                ex.push(row.iter().zip(xz).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32);
                gg.push(row.iter().zip(&xq).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32);
            }
        }
        let cpu = (err(&y, &ex), err(&ex, reference), err(&gg, reference));
        self.annotate(label, &y, reference, Some(cpu));
        y
    }

    /// Runs an entry point of ops.wgsl. Inputs: (binding 0..4) -> data; outputs o (5) and o2 (6).
    fn op(&self, e: &str, inputs: &[(u32, &[f32])], o: usize, o_init: Option<&[f32]>, o2: usize, p: [u32; 8], groups: (u32, u32, u32)) -> (Vec<f32>, Vec<f32>) {
        let bufs: Vec<(u32, wgpu::Buffer)> = inputs.iter().map(|(i, v)| (*i, self.gpu.buffer_f32("in", v))).collect();
        let bo = match o_init { Some(v) => self.gpu.buffer_f32("o", v), None => self.gpu.empty_buffer("o", (o.max(1) * 4) as u64) };
        let bo2 = self.gpu.empty_buffer("o2", (o2.max(1) * 4) as u64);
        let u = self.gpu.uniform(&p);
        // bindings used by each entry point (the automatic layout contains only those)
        let uses: &[u32] = match e { "rmsnorm" | "attn" => &[0, 1, 2], "conv" | "delta" => &[0, 1, 2, 6], "beta_gate" => &[0, 1, 2, 3], "moe_sum" => &[0, 1, 2, 3, 4, 6], _ => &[0, 1] };
        let mut bind: Vec<(u32, &wgpu::Buffer)> = vec![];
        for &i in uses.iter().filter(|i| **i < 5) { bind.push((i, bufs.iter().find(|(j, _)| *j == i).map(|(_, b)| b).unwrap_or(&self.empty))); }
        bind.push((5, &bo)); if uses.contains(&6) { bind.push((6, &bo2)); }
        if e != "delta" { bind.push((7, &u)); }   // delta does not read p
        if std::env::var("MECHTA_DEBUG").is_ok() { eprintln!("op {e}: binding {:?}", bind.iter().map(|(i, _)| *i).collect::<Vec<_>>()); }
        self.gpu.run_b(&self.op[e], &bind, groups);
        (self.gpu.read_f32(&bo, o), if o2 > 0 { self.gpu.read_f32(&bo2, o2) } else { vec![] })
    }
}

pub(crate) fn token_ids() -> Vec<u32> {
    std::fs::read_to_string(TOKEN).unwrap().lines().skip(1).map(|l| l.split('\t').nth(2).unwrap().parse().unwrap()).collect()
}

fn fb(v: f32) -> u32 { v.to_bits() }

/// RoPE cos|sin table for one position, as ggml_mrope_cache_init does (text: same positions).
fn rope_table(pos: i32) -> Vec<f32> {
    let scale = 1e7f32.powf(-2.0 / 64.0);
    let mut th = pos as f32; let (mut c, mut s) = (vec![0f32; 32], vec![0f32; 32]);
    for i in 0..32 { c[i] = th.cos(); s[i] = th.sin(); th *= scale; }
    [c, s].concat()
}

pub fn layer(positions: &[i32]) {
    let g = Gguf::open(crate::model_path()).expect("GGUF");
    let gpu = Gpu::new();
    let lim = gpu.device.limits();
    println!("GPU {} · binding storage max {} MiB · buffer max {} MiB", gpu.info.name, lim.max_storage_buffer_binding_size >> 20, lim.max_buffer_size >> 20);
    let src_mv = include_str!("wgsl/matvec.wgsl"); let src_op = include_str!("wgsl/ops.wgsl");
    let mv = ["mv_q8_0", "mv_q4_k", "mv_q6_k", "mv_f32"].into_iter().map(|e| (e, gpu.pipeline(src_mv, e))).collect();
    let op = ["rmsnorm", "conv", "beta_gate", "delta", "sum", "swiglu", "moe_sum", "rope", "attn", "attn_gate"].into_iter().map(|e| (e, gpu.pipeline(src_op, e))).collect();
    let dq = ["q8_0", "q4_k", "q6_k"].into_iter().map(|e| (e, gpu.pipeline(include_str!("wgsl/dequant.wgsl"), e))).collect();
    let rt = gpu.pipeline(include_str!("wgsl/routing.wgsl"), "routing");
    let empty = gpu.empty_buffer("empty", 4);
    let mut b = Bench { g, gpu, mv, op, rt, dq, d: Dump::load(), empty, weights: HashMap::new(), results: vec![] };
    let ids_tok = token_ids();
    for &p in positions {
        // ---------- layer 0: DeltaNet ----------
        let tok = ids_tok[p as usize];
        let te = b.g.tensor("token_embd.weight").unwrap().clone();
        let x = b.deq(&te, tok as u64 * 64, 64);
        let w = b.f32_model("blk.0.attn_norm.weight");
        let r = b.d.get("attn_norm-0", p, 0);
        let (y, _) = b.op("rmsnorm", &[(0, &x), (1, &w)], 2048, None, 0, [2048, 1, fb(EPS), fb(1.0), 2048, 2048, 1, 0], (1, 1, 1));
        b.annotate("0 embedding + attn_norm", &y, &r, None);
        let h = r;
        let r = b.d.get("linear_attn_qkv_mixed-0", p, 0); b.matvec("blk.0.attn_qkv.weight", &h, None, 0, &r, "0 attn_qkv (Q8_0)");
        let r = b.d.get("z-0", p, 0); b.matvec("blk.0.attn_gate.weight", &h, None, 0, &r, "0 attn_gate z (Q8_0)");
        let r = b.d.get("beta-0", p, 0); b.matvec("blk.0.ssm_beta.weight", &h, None, 0, &r, "0 ssm_beta (F32)");
        let r = b.d.get("alpha-0", p, 0); b.matvec("blk.0.ssm_alpha.weight", &h, None, 0, &r, "0 ssm_alpha (F32)");
        let (be, al) = (b.d.get("beta-0", p, 0), b.d.get("alpha-0", p, 0));
        let (dt, sa) = (b.f32_model("blk.0.ssm_dt.bias"), b.f32_model("blk.0.ssm_a"));
        let (y, _) = b.op("beta_gate", &[(0, &be), (1, &al), (2, &dt), (3, &sa)], 64, None, 0, [32, 0, 0, 0, 0, 0, 0, 0], (1, 1, 1));
        let r = [b.d.get("beta_sigmoid-0", p, 0), b.d.get("gate-0", p, 0)].concat();
        b.annotate("0 sigmoid(beta) | softplus*a", &y, &r, None);
        let qkv = b.d.get("linear_attn_qkv_mixed-0", p, 0); let cs = b.d.get("conv_states-0", p, 0);
        let cw = b.f32_model("blk.0.ssm_conv1d.weight");
        let (y, st) = b.op("conv", &[(0, &qkv), (1, &cw), (2, &cs)], 8192, None, 24576, [8192, 0, 0, 0, 0, 0, 0, 0], (128, 1, 1));
        let r = b.d.get("conv_output_silu-0", p, 0); b.annotate("0 conv + silu", &y, &r, None);
        if p < 55 { let r = b.d.get("conv_states-0", p + 1, 0); b.annotate("0 new conv state (vs pos+1)", &st, &r, None); }
        let cv = b.d.get("conv_output_silu-0", p, 0);
        let (y, _) = b.op("rmsnorm", &[(0, &cv)], 4096, None, 0, [128, 32, fb(EPS / 128.0), fb(1.0 / 128f32.sqrt()), 128, 128, 0, 0], (32, 1, 1));
        let r = [b.d.get("q_conv_predelta-0", p, 0), b.d.get("k_conv_predelta-0", p, 0)].concat();
        b.annotate("0 l2norm q, k", &y, &r, None);
        let qk = r;
        let bg = [b.d.get("beta_sigmoid-0", p, 0), b.d.get("gate-0", p, 0)].concat();
        let s0 = b.d.get("state_predelta-0", p, 0);
        let (s1, out) = b.op("delta", &[(0, &qk), (1, &cv), (2, &bg)], s0.len(), Some(&s0), 4096, [0; 8], (32, 1, 1));
        let r = b.d.get("attn_output-0", p, 0); b.annotate("0 delta net: output", &out, &r, None);
        let r = b.d.get("new_state-0", p, 0); b.annotate("0 delta net: new state", &s1, &r, None);
        let ao = b.d.get("attn_output-0", p, 0); let z = b.d.get("z-0", p, 0); let sn = b.f32_model("blk.0.ssm_norm.weight");
        let (y, _) = b.op("rmsnorm", &[(0, &ao), (1, &sn), (2, &z)], 4096, None, 0, [128, 32, fb(EPS), fb(1.0), 128, 128, 3, 0], (32, 1, 1));
        let r = b.d.get("final_output-0", p, 0); b.annotate("0 norm gated (ssm_norm * silu z)", &y, &r, None);
        let fo = r; let r = b.d.get("linear_attn_out-0", p, 0); b.matvec("blk.0.ssm_out.weight", &fo, None, 0, &r, "0 ssm_out (Q8_0)");
        let la = r;
        let (y, _) = b.op("sum", &[(0, &la), (1, &x)], 2048, None, 0, [2048, 0, 0, 0, 0, 0, 0, 0], (32, 1, 1));
        let r = b.d.get("attn_residual-0", p, 0); b.annotate("0 attention residual", &y, &r, None);
        moe(&mut b, 0, p);

        // ---------- layer 3: attention ----------
        let xin = b.d.get("l_out-2", p, 0);
        let w = b.f32_model("blk.3.attn_norm.weight");
        let (y, _) = b.op("rmsnorm", &[(0, &xin), (1, &w)], 2048, None, 0, [2048, 1, fb(EPS), fb(1.0), 2048, 2048, 1, 0], (1, 1, 1));
        let r = b.d.get("attn_norm-3", p, 0); b.annotate("3 attn_norm", &y, &r, None);
        let h = r;
        let r = b.d.get("Qcur_full-3", p, 0); b.matvec("blk.3.attn_q.weight", &h, None, 0, &r, "3 attn_q (Q8_0)");
        let r = b.d.get("Kcur-3", p, 0); b.matvec("blk.3.attn_k.weight", &h, None, 0, &r, "3 attn_k (Q8_0)");
        let r = b.d.get("Vcur-3", p, 0); b.matvec("blk.3.attn_v.weight", &h, None, 0, &r, "3 attn_v (Q8_0)");
        let qf = b.d.get("Qcur_full-3", p, 0); let qn = b.f32_model("blk.3.attn_q_norm.weight");
        let (y, _) = b.op("rmsnorm", &[(0, &qf), (1, &qn)], 4096, None, 0, [256, 16, fb(EPS), fb(1.0), 512, 256, 1, 0], (16, 1, 1));
        let r = b.d.get("Qcur_normed-3", p, 0); b.annotate("3 q_norm", &y, &r, None);
        let kr = b.d.get("Kcur-3", p, 0); let kn = b.f32_model("blk.3.attn_k_norm.weight");
        let (y, _) = b.op("rmsnorm", &[(0, &kr), (1, &kn)], 512, None, 0, [256, 2, fb(EPS), fb(1.0), 256, 256, 1, 0], (2, 1, 1));
        let r = b.d.get("Kcur_normed-3", p, 0); b.annotate("3 k_norm", &y, &r, None);
        let tab = rope_table(p);
        let qn = b.d.get("Qcur_normed-3", p, 0);
        let (y, _) = b.op("rope", &[(0, &qn), (1, &tab)], 4096, None, 0, [256, 16, 0, 0, 256, 256, 0, 0], (16, 1, 1));
        let r = b.d.get("Qcur-3", p, 0); b.annotate("3 RoPE q", &y, &r, None);
        let kn = b.d.get("Kcur_normed-3", p, 0);
        let (y, _) = b.op("rope", &[(0, &kn), (1, &tab)], 512, None, 0, [256, 2, 0, 0, 256, 256, 0, 0], (2, 1, 1));
        let r = b.d.get("Kcur-3", p, 1); b.annotate("3 RoPE k", &y, &r, None);
        // K, V cache of positions 0..=p (from the dump): in f32, and rounded to f16 like the llama.cpp cache
        let (mut kc, mut vc) = (Vec::new(), Vec::new());
        for s in 0..=p { kc.extend(b.d.get("Kcur-3", s, 1)); vc.extend(b.d.get("Vcur-3", s, 1)); }
        let q = b.d.get("Qcur-3", p, 0); let r = b.d.get("attn_pregate-3", p, 0);
        let pa = [p as u32 + 1, 0, fb(1.0 / 16.0), 0, 0, 0, 0, 0];
        let (y, _) = b.op("attn", &[(0, &q), (1, &kc), (2, &vc)], 4096, None, 0, pa, (16, 1, 1));
        // exact CPU (f64, f32 cache): checks the kernel independently of llama.cpp
        let mut es = vec![0f32; 4096];
        for hq in 0..16 {
            let kvh = hq / 8; let n = p as usize + 1;
            let sc: Vec<f64> = (0..n).map(|t| (0..256).map(|d| q[hq * 256 + d] as f64 * kc[t * 512 + kvh * 256 + d] as f64).sum::<f64>() / 16.0).collect();
            let mx = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = sc.iter().map(|v| (v - mx).exp()).collect(); let tot: f64 = ex.iter().sum();
            for d in 0..256 { es[hq * 256 + d] = ((0..n).map(|t| ex[t] * vc[t * 512 + kvh * 256 + d] as f64).sum::<f64>() / tot) as f32; }
        }
        b.annotate("3 attention, cache f32", &y, &r, Some((err(&y, &es), err(&es, &r), f64::NAN)));
        let h16 = |v: &Vec<f32>| v.iter().map(|x| f16::from_f32(*x).to_f32()).collect::<Vec<f32>>();
        let (kh, vh) = (h16(&kc), h16(&vc));
        let (y, _) = b.op("attn", &[(0, &q), (1, &kh), (2, &vh)], 4096, None, 0, pa, (16, 1, 1));
        b.annotate("3 attention, cache f16", &y, &r, None);
        let qh = h16(&q);
        let (y, _) = b.op("attn", &[(0, &qh), (1, &kh), (2, &vh)], 4096, None, 0, pa, (16, 1, 1));
        b.annotate("3 attention, cache f16 and q f16", &y, &r, None);
        // CPU: ggml flash attention with f16 cache (q in f16, VKQ accumulator in f16, online softmax)
        let mut emu = vec![0f32; 4096];
        for hq in 0..16 {
            let kvh = hq / 8; let (mut m, mut sm) = (f32::NEG_INFINITY, 0f32); let mut acc = vec![f16::ZERO; 256];
            for t in 0..=p as usize {
                let o = t * 512 + kvh * 256;
                let s = (0..256).map(|d| qh[hq * 256 + d] as f64 * kh[o + d] as f64).sum::<f64>() as f32 / 16.0;
                let (mut ms, mut vs) = (1f32, 1f32);
                if s > m { let mo = m; m = s; ms = (mo - m).exp(); for v in acc.iter_mut() { *v = f16::from_f32(v.to_f32() * ms); } } else { vs = (s - m).exp(); }
                for d in 0..256 { acc[d] = f16::from_f32(vh[o + d].mul_add(vs, acc[d].to_f32())); }   // FMA as in AVX-512
                sm = sm * ms + vs;
            }
            for d in 0..256 { emu[hq * 256 + d] = acc[d].to_f32() / sm; }
        }
        b.annotate("3 attention, CPU like ggml (VKQ in f16)", &emu, &r, None);
        // CPU: ggml attention WITHOUT flash attention: kq = k16 . q16, softmax in f32, weights in f16, sum in f32
        let mut emu2 = vec![0f32; 4096];
        for hq in 0..16 {
            let kvh = hq / 8; let n = p as usize + 1;
            let sc: Vec<f32> = (0..n).map(|t| (0..256).map(|d| qh[hq * 256 + d] as f64 * kh[t * 512 + kvh * 256 + d] as f64).sum::<f64>() as f32 * (1.0 / 16.0)).collect();
            let mx = sc.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let ex: Vec<f32> = sc.iter().map(|v| (v - mx).exp()).collect();
            let tot: f64 = ex.iter().map(|v| *v as f64).sum();
            let pw: Vec<f32> = ex.iter().map(|v| f16::from_f32(v * (1.0 / tot) as f32).to_f32()).collect();
            for d in 0..256 { emu2[hq * 256 + d] = (0..n).map(|t| pw[t] as f64 * vh[t * 512 + kvh * 256 + d] as f64).sum::<f64>() as f32; }
        }
        b.annotate("3 attention, CPU like ggml without FA", &emu2, &r, None);
        eprintln!("pos {p}: attention GPU-f16 {:.2e} · like ggml with FA {:.2e} · like ggml without FA {:.2e}", err(&b.op("attn", &[(0, &q), (1, &kh), (2, &vh)], 4096, None, 0, pa, (16, 1, 1)).0, &r), err(&emu, &r), err(&emu2, &r));
        let ap = b.d.get("attn_pregate-3", p, 0);
        let (y, _) = b.op("attn_gate", &[(0, &ap), (1, &qf)], 4096, None, 0, [4096, 0, 0, 0, 0, 0, 0, 0], (64, 1, 1));
        let r = b.d.get("attn_gated-3", p, 0); b.annotate("3 output gate sigmoid", &y, &r, None);
        let ag = r; let r = b.d.get("attn_output-3", p, 0); b.matvec("blk.3.attn_output.weight", &ag, None, 0, &r, "3 attn_output (Q8_0)");
        let (y, _) = b.op("sum", &[(0, &r), (1, &xin)], 2048, None, 0, [2048, 0, 0, 0, 0, 0, 0, 0], (32, 1, 1));
        let r = b.d.get("attn_residual-3", p, 0); b.annotate("3 attention residual", &y, &r, None);
        moe(&mut b, 3, p);
        eprintln!("position {p} done");
    }
    print(&b.results, positions);
}

fn moe(b: &mut Bench, il: usize, p: i32) {
    let l = |s: &str| format!("{s}-{il}"); let w = |s: &str| format!("blk.{il}.{s}");
    let ar = b.d.get(&l("attn_residual"), p, 0);
    let pn = b.f32_model(&w("post_attention_norm.weight"));
    let (y, _) = b.op("rmsnorm", &[(0, &ar), (1, &pn)], 2048, None, 0, [2048, 1, fb(EPS), fb(1.0), 2048, 2048, 1, 0], (1, 1, 1));
    let r = b.d.get(&l("attn_post_norm"), p, 0); b.annotate(&format!("{il} post_attention_norm"), &y, &r, None);
    let h = r;
    let r = b.d.get(&l("ffn_moe_logits"), p, 0); b.matvec(&w("ffn_gate_inp.weight"), &h, None, 0, &r, &format!("{il} router (F32)"));
    // routing on the GPU, from the logits of the dump
    let lg = r;
    let (bl, bp, bi, bw) = (b.gpu.buffer_f32("lg", &lg), b.gpu.empty_buffer("pr", 1024), b.gpu.empty_buffer("id", 32), b.gpu.empty_buffer("w", 32));
    b.gpu.run_b(&b.rt, &[(0, &bl), (1, &bp), (2, &bi), (3, &bw)], (1, 1, 1));
    let pr = b.gpu.read_f32(&bp, 256); let wn = b.gpu.read_f32(&bw, 8);
    let ids_gpu: Vec<u32> = b.gpu.read_f32(&bi, 8).iter().map(|v| v.to_bits()).collect();
    let r = b.d.get(&l("ffn_moe_probs"), p, 0); b.annotate(&format!("{il} softmax router"), &pr, &r, None);
    // experts chosen by llama.cpp: the probabilities of the dump equal to the chosen weights (same bits)
    let sel = b.d.get(&l("ffn_moe_weights"), p, 0);
    let ids: Vec<u32> = sel.iter().map(|v| r.iter().position(|x| x.to_bits() == v.to_bits()).expect("expert not found") as u32).collect();
    let equal = ids_gpu.iter().filter(|e| ids.contains(e)).count();
    b.results.push((format!("{il} routing: experts different from llama.cpp"), 8, (8 - equal) as f64, None));
    let r = b.d.get(&l("ffn_moe_weights_norm"), p, 0);
    let mut wn_ord = vec![0f32; 8];
    for (k, e) in ids.iter().enumerate() { if let Some(j) = ids_gpu.iter().position(|x| x == e) { wn_ord[k] = wn[j]; } }
    b.annotate(&format!("{il} normalized weights"), &wn_ord, &r, None);
    // experts, with the experts and the inputs of llama.cpp
    let r = b.d.get(&l("ffn_moe_gate"), p, 0); b.matvec(&w("ffn_gate_exps.weight"), &h, Some(&ids), 0, &r, &format!("{il} experts gate"));
    let r = b.d.get(&l("ffn_moe_up"), p, 0); b.matvec(&w("ffn_up_exps.weight"), &h, Some(&ids), 0, &r, &format!("{il} experts up"));
    let (ga, up) = (b.d.get(&l("ffn_moe_gate"), p, 0), b.d.get(&l("ffn_moe_up"), p, 0));
    let (y, _) = b.op("swiglu", &[(0, &ga), (1, &up)], 4096, None, 0, [4096, 0, 0, 0, 0, 0, 0, 0], (64, 1, 1));
    let r = b.d.get(&l("ffn_moe_swiglu"), p, 0); b.annotate(&format!("{il} experts swiglu"), &y, &r, None);
    let sw = r; let r = b.d.get(&l("ffn_moe_down"), p, 0); b.matvec(&w("ffn_down_exps.weight"), &sw, Some(&ids), 512, &r, &format!("{il} experts down"));
    // shared expert
    let r = b.d.get(&l("ffn_gate"), p, 0); b.matvec(&w("ffn_gate_shexp.weight"), &h, None, 0, &r, &format!("{il} shared gate (Q8_0)"));
    let r = b.d.get(&l("ffn_up"), p, 0); b.matvec(&w("ffn_up_shexp.weight"), &h, None, 0, &r, &format!("{il} shared up (Q8_0)"));
    let (ga, up) = (b.d.get(&l("ffn_gate"), p, 0), b.d.get(&l("ffn_up"), p, 0));
    let (y, _) = b.op("swiglu", &[(0, &ga), (1, &up)], 512, None, 0, [512, 0, 0, 0, 0, 0, 0, 0], (8, 1, 1));
    let r = b.d.get(&l("ffn_swiglu"), p, 0); b.annotate(&format!("{il} shared swiglu"), &y, &r, None);
    let sw = r; let r = b.d.get(&l("ffn_shexp"), p, 0); b.matvec(&w("ffn_down_shexp.weight"), &sw, None, 0, &r, &format!("{il} shared down (Q8_0)"));
    let r = b.d.get(&l("shared_expert_gate"), p, 0); b.matvec(&w("ffn_gate_inp_shexp.weight"), &h, None, 0, &r, &format!("{il} shared gate scalar (F32)"));
    let (dn, wn, sh, sg) = (b.d.get(&l("ffn_moe_down"), p, 0), b.d.get(&l("ffn_moe_weights_norm"), p, 0), b.d.get(&l("ffn_shexp"), p, 0), b.d.get(&l("shared_expert_gate"), p, 0));
    let (fo, lo) = b.op("moe_sum", &[(0, &dn), (1, &wn), (2, &sh), (3, &sg), (4, &ar)], 2048, None, 2048, [2048, 8, 0, 0, 0, 0, 0, 0], (32, 1, 1));
    let r = b.d.get(&l("ffn_out"), p, 0); b.annotate(&format!("{il} MoE + shared sum"), &fo, &r, None);
    let r = b.d.get(&l("l_out"), p, 0); b.annotate(&format!("{il} l_out (residual)"), &lo, &r, None);
}

fn print(results: &[(String, usize, f64, Option<(f64, f64, f64)>)], pos: &[i32]) {
    println!("\npositions {pos:?} · error = max|a-b| / max|b|, worst over the positions");
    println!("{:44} {:>7} {:>10} | {:>10} {:>10} {:>10}", "operation", "values", "GPU-dump", "GPU-exact", "exact-dump", "ggml-dump");
    let mut order: Vec<String> = vec![]; let mut agg: HashMap<String, (usize, f64, Option<(f64, f64, f64)>)> = HashMap::new();
    for (n, k, e, c) in results {
        let v = agg.entry(n.clone()).or_insert_with(|| { order.push(n.clone()); (*k, 0.0, None) });
        v.1 = v.1.max(*e);
        if let Some((a, bb, cc)) = c { let o = v.2.unwrap_or((0.0, 0.0, 0.0)); v.2 = Some((o.0.max(*a), o.1.max(*bb), o.2.max(*cc))); }
    }
    for n in &order {
        let (k, e, c) = &agg[n];
        match c {
            Some((a, bb, _)) if n.contains("attention") => println!("{n:44} {k:>7} {e:>10.2e} | {a:>10.2e} {bb:>10.2e} {:>10}", "n/d"),
            Some((a, bb, cc)) => println!("{n:44} {k:>7} {e:>10.2e} | {a:>10.2e} {bb:>10.2e} {cc:>10.2e}"),
            None => println!("{n:44} {k:>7} {e:>10.2e} |"),
        }
    }
}

/// Flint F3: the FFN of layers `layers` at `positions`, every operation with the inputs FROM THE DUMP
/// (`<dir>/tensors.bin`), as `moe` above. Matvec with the engine's kernels (matvec4.wgsl), routing with
/// flint_moe.wgsl. Note: llama.cpp divides the input of ffn_down_exps by e = 2^ceil(log2(amax/1024)) (>= 1, per
/// expert) and multiplies back after: the dumped `ffn_moe_down` is the divided one, so the reference is multiplied by e.
pub fn flint_ffn(dir: &str, layers: &[usize], positions: &[i32]) {
    let g = Gguf::open(crate::model_path()).expect("GGUF");
    let gpu = Gpu::new();
    let src_mv = crate::engine::q3_source("matvec4");
    let mv = ["mv_q8_0", "mv_q4_k", "mv_q6_k", "mv_q4_1", "mv_f32"].into_iter().map(|e| (e, gpu.pipeline(&src_mv, e))).collect();
    let op = ["rmsnorm", "sum", "swiglu"].into_iter().map(|e| (e, gpu.pipeline(include_str!("wgsl/ops.wgsl"), e))).collect();
    let dq = ["q8_0", "q4_k", "q6_k", "q4_1"].into_iter().map(|e| (e, gpu.pipeline(include_str!("wgsl/dequant.wgsl"), e))).collect();
    let rt = gpu.pipeline(include_str!("wgsl/flint_moe.wgsl"), "routing_sig");
    let sum6 = gpu.pipeline(include_str!("wgsl/flint_sum.wgsl"), "moe_sum6");
    let empty = gpu.empty_buffer("empty", 4);
    let mut b = Bench { g, gpu, mv, op, rt, dq, d: Dump::load_file(&format!("{dir}/tensors.bin")), empty, weights: HashMap::new(), results: vec![] };
    for &p in positions {
        for &il in layers {
            let l = |s: &str| format!("{s}-{il}"); let w = |s: &str| format!("blk.{il}.{s}");
            let h = b.d.get(&l("ffn_norm"), p, 0);
            let r = b.d.get(&l("ffn_moe_logits"), p, 0); b.matvec(&w("ffn_gate_inp.weight"), &h, None, 0, &r, &format!("{il} router (F32)"));
            let lg = r;
            let (bl, bp, bi, bw) = (b.gpu.buffer_f32("lg", &lg), b.gpu.empty_buffer("pr", 1024), b.gpu.empty_buffer("id", 32), b.gpu.empty_buffer("w", 32));
            b.gpu.run_b(&b.rt, &[(0, &bl), (1, &bp), (2, &bi), (3, &bw)], (1, 1, 1));
            let pr = b.gpu.read_f32(&bp, 256); let wn = b.gpu.read_f32(&bw, 6);
            let ids_gpu: Vec<u32> = b.gpu.read_f32(&bi, 6).iter().map(|v| v.to_bits()).collect();
            let r = b.d.get(&l("ffn_moe_probs"), p, 0); b.annotate(&format!("{il} sigmoid router"), &pr, &r, None);
            let sel = b.d.get(&l("ffn_moe_weights"), p, 0);
            let ids: Vec<u32> = sel.iter().map(|v| r.iter().position(|x| x.to_bits() == v.to_bits()).expect("expert not found") as u32).collect();
            let equal = ids_gpu.iter().filter(|e| ids.contains(e)).count();
            b.results.push((format!("{il} routing: experts different from llama.cpp"), 6, (6 - equal) as f64, None));
            let r = b.d.get(&l("ffn_moe_weights_norm"), p, 0);
            let mut wn_ord = vec![0f32; 6];
            for (k, e) in ids.iter().enumerate() { if let Some(j) = ids_gpu.iter().position(|x| x == e) { wn_ord[k] = wn[j]; } }
            b.annotate(&format!("{il} normalized weights"), &wn_ord, &r, None);
            let r = b.d.get(&l("ffn_moe_gate"), p, 0); b.matvec(&w("ffn_gate_exps.weight"), &h, Some(&ids), 0, &r, &format!("{il} experts gate"));
            let r = b.d.get(&l("ffn_moe_up"), p, 0); b.matvec(&w("ffn_up_exps.weight"), &h, Some(&ids), 0, &r, &format!("{il} experts up"));
            let (ga, up) = (b.d.get(&l("ffn_moe_gate"), p, 0), b.d.get(&l("ffn_moe_up"), p, 0));
            let (y, _) = b.op("swiglu", &[(0, &ga), (1, &up)], 3072, None, 0, [3072, 0, 0, 0, 0, 0, 0, 0], (48, 1, 1));
            let sw = b.d.get(&l("ffn_moe_swiglu"), p, 0); b.annotate(&format!("{il} experts swiglu"), &y, &sw, None);
            // reference of the down projection, multiplied back by e of each expert
            let mut rd = b.d.get(&l("ffn_moe_down"), p, 0);
            let mut es = vec![];
            for j in 0..6 {
                let amax = sw[j * 512..(j + 1) * 512].iter().fold(0f32, |m, v| m.max(v.abs()));
                let e = ((amax / 1024.0).ln() * std::f32::consts::LOG2_E).ceil().clamp(0.0, 127.0);
                let e = (e * std::f32::consts::LN_2).exp();
                es.push(e);
                for v in &mut rd[j * 2048..(j + 1) * 2048] { *v *= e; }
            }
            b.results.push((format!("{il} amax rescale e of llama.cpp (max)"), 6, es.iter().cloned().fold(0f32, f32::max) as f64, None));
            // the input llama.cpp multiplied: swiglu / e (same for exact; the ggml column quantizes this one)
            let mut sw_div = sw.clone();
            for j in 0..6 { for v in &mut sw_div[j * 512..(j + 1) * 512] { *v /= es[j]; } }
            let yd = b.matvec(&w("ffn_down_exps.weight"), &sw, Some(&ids), 512, &rd, &format!("{il} experts down (x as is)"));
            let _ = yd;
            let mut rd_div = rd.clone();
            for j in 0..6 { for v in &mut rd_div[j * 2048..(j + 1) * 2048] { *v /= es[j]; } }
            b.matvec(&w("ffn_down_exps.weight"), &sw_div, Some(&ids), 512, &rd_div, &format!("{il} experts down (x / e, as ggml)"));
            // shared expert
            let r = b.d.get(&l("ffn_gate"), p, 0); b.matvec(&w("ffn_gate_shexp.weight"), &h, None, 0, &r, &format!("{il} shared gate"));
            let r = b.d.get(&l("ffn_up"), p, 0); b.matvec(&w("ffn_up_shexp.weight"), &h, None, 0, &r, &format!("{il} shared up"));
            let ssw = b.d.get(&l("ffn_swiglu"), p, 0);
            let r = b.d.get(&l("ffn_shexp"), p, 0); b.matvec(&w("ffn_down_shexp.weight"), &ssw, None, 0, &r, &format!("{il} shared down"));
            // sum: llama.cpp's down (multiplied back), weights, shared, residual
            let (wn_r, sh, res) = (b.d.get(&l("ffn_moe_weights_norm"), p, 0), b.d.get(&l("ffn_shexp"), p, 0), b.d.get(&l("ffn_inp"), p, 0));
            let bufs = [b.gpu.buffer_f32("ed", &rd), b.gpu.buffer_f32("w", &wn_r), b.gpu.buffer_f32("sd", &sh), b.gpu.buffer_f32("res", &res)];
            let bo = b.gpu.empty_buffer("o", 2048 * 4);
            b.gpu.run_b(&sum6, &[(0, &bufs[0]), (1, &bufs[1]), (2, &bufs[2]), (3, &bufs[3]), (4, &bo)], (32, 1, 1));
            let lo = b.gpu.read_f32(&bo, 2048);
            let r = b.d.get(&l("l_out"), p, 0); b.annotate(&format!("{il} sum + residual = l_out"), &lo, &r, None);
        }
    }
    print(&b.results, positions);
}

/// Flint F3: the attention of layers `layers` at `positions`, inputs from the dump. Occurrences in the
/// dump: Qcur/Kcur 0 = matmul, 1 = reshaped, 2 = final (after the QK norm and, on sliding layers, RoPE); Vcur 1 = reshaped.
/// Exact = f64 on the CPU. "ggml" = Q, K, V rounded to f16 (llama.cpp keeps K/V in f16), then f64.
pub fn flint_attn(dir: &str, layers: &[usize], positions: &[i32]) {
    let g = Gguf::open(crate::model_path()).expect("GGUF");
    let conf = crate::flint::Conf::da_gguf(&g);
    let gpu = Gpu::new();
    let op = ["rmsnorm"].into_iter().map(|e| (e, gpu.pipeline(include_str!("wgsl/ops.wgsl"), e))).collect();
    // the bench builds K/V as f32 arrays: f32 cache format (MECHTA_FLINT_KV unset)
    assert!(!crate::flint::kv_f16(), "flint-attn: the bench feeds an f32 cache");
    let src = crate::flint::attn_source();
    let rope = gpu.pipeline(&src, "rope_neox");
    let attn = gpu.pipeline(&src, "attn_sink");
    let fd = gpu.pipeline(&src, "attn_fd");
    let fdu = gpu.pipeline(&src, "attn_fd_merge");
    let empty = gpu.empty_buffer("empty", 4);
    let rt = gpu.pipeline(include_str!("wgsl/flint_moe.wgsl"), "routing_sig");
    let mut b = Bench { g, gpu, mv: HashMap::new(), op, rt, dq: HashMap::new(), d: Dump::load_file(&format!("{dir}/tensors.bin")), empty, weights: HashMap::new(), results: vec![] };
    let h16 = |v: &[f32]| v.iter().map(|x| f16::from_f32(*x).to_f32()).collect::<Vec<f32>>();
    for &p in positions {
        for &il in layers {
            let l = |s: &str| format!("{s}-{il}"); let w = |s: &str| format!("blk.{il}.{s}");
            let kind = if conf.swa[il] { "swa" } else { "full" };
            // QK norm (+ RoPE on the layers that have it), Q and K
            for (name, heads, weight) in [("Qcur", 32u32, "attn_q_norm.weight"), ("Kcur", 4u32, "attn_k_norm.weight")] {
                let x = b.d.get(&l(name), p, 1);
                let wn = b.f32_model(&w(weight));
                let (mut y, _) = b.op("rmsnorm", &[(0, &x), (1, &wn)], (heads * 128) as usize, None, 0, [128, heads, fb(conf.eps), fb(1.0), 128, 128, 1, 0], (heads, 1, 1));
                // exact
                let mut ex = vec![0f32; x.len()];
                for h in 0..heads as usize {
                    let r = &x[h * 128..(h + 1) * 128];
                    let s = 1.0 / ((r.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / 128.0 + conf.eps as f64).sqrt());
                    for i in 0..128 { ex[h * 128 + i] = (r[i] as f64 * s * wn[i] as f64) as f32; }
                }
                if conf.rope[il] {
                    let tab = crate::flint::rope_table(p as usize, conf.rope_base);
                    let (bi, bt, bo) = (b.gpu.buffer_f32("in", &y), b.gpu.buffer_f32("tab", &tab), b.gpu.empty_buffer("o", (heads * 128 * 4) as u64));
                    b.gpu.run_b(&rope, &[(0, &bi), (1, &bt), (5, &bo)], (heads, 1, 1));
                    y = b.gpu.read_f32(&bo, (heads * 128) as usize);
                    let mut ex2 = ex.clone();
                    for h in 0..heads as usize {
                        for i in 0..64 {
                            let th = p as f64 * (conf.rope_base as f64).powf(-2.0 * i as f64 / 128.0);
                            let (x0, x1) = (ex[h * 128 + i] as f64, ex[h * 128 + i + 64] as f64);
                            ex2[h * 128 + i] = (x0 * th.cos() - x1 * th.sin()) as f32;
                            ex2[h * 128 + i + 64] = (x0 * th.sin() + x1 * th.cos()) as f32;
                        }
                    }
                    ex = ex2;
                }
                let r = b.d.get(&l(name), p, 2);
                let e = (err(&y, &ex), err(&ex, &r), err(&ex, &r));
                b.annotate(&format!("{il} {kind} {name} norm{}", if conf.rope[il] { " + rope" } else { "" }), &y, &r, Some(e));
            }
            // attention over positions 0..=p from the dump's K and V (all of them are inside the window: p < 513)
            let q = b.d.get(&l("Qcur"), p, 2);
            let (mut kc, mut vc) = (vec![], vec![]);
            for j in 0..=p { kc.extend(b.d.get(&l("Kcur"), j, 2)); vc.extend(b.d.get(&l("Vcur"), j, 1)); }
            let sinks = b.f32_model(&w("attn_sinks.weight"));
            let n = (p + 1) as usize; let ctxs = 64usize;
            let (bq, bk, bv, bs) = (b.gpu.buffer_f32("q", &q), b.gpu.buffer_f32("k", &kc), b.gpu.buffer_f32("v", &vc), b.gpu.buffer_f32("s", &sinks));
            let (bsc, bo) = (b.gpu.empty_buffer("sc", (32 * ctxs * 4) as u64), b.gpu.empty_buffer("o", 4096 * 4));
            let u = b.gpu.uniform(&[n as u32, 0, 0, ctxs as u32, ctxs as u32, 0, 0, 0]);   // n, slot, lo, ring, ctx
            b.gpu.run_b(&attn, &[(0, &bq), (1, &bk), (2, &bv), (3, &bs), (4, &bsc), (5, &bo), (7, &u)], (32, 1, 1));
            let y = b.gpu.read_f32(&bo, 4096);
            let exact = |q: &[f32], kc: &[f32], vc: &[f32]| -> Vec<f32> {
                let mut o = vec![0f32; 4096];
                for h in 0..32 {
                    let gk = h / 8;
                    let sc: Vec<f64> = (0..n).map(|j| (0..128).map(|e| q[h * 128 + e] as f64 * kc[j * 512 + gk * 128 + e] as f64).sum::<f64>() / 128f64.sqrt()).collect();
                    let m = sc.iter().cloned().fold(sinks[h] as f64, f64::max);
                    let ex: Vec<f64> = sc.iter().map(|s| (s - m).exp()).collect();
                    let den = ex.iter().sum::<f64>() + (sinks[h] as f64 - m).exp();
                    for e in 0..128 { o[h * 128 + e] = ((0..n).map(|j| ex[j] * vc[j * 512 + gk * 128 + e] as f64).sum::<f64>() / den) as f32; }
                }
                o
            };
            let ex = exact(&q, &kc, &vc);
            let gg = exact(&h16(&q), &h16(&kc), &h16(&vc));
            let r = b.d.get(&l("kqv_out"), p, 0);
            let e = (err(&y, &ex), err(&ex, &r), err(&gg, &r));
            b.annotate(&format!("{il} {kind} attention with sinks ({n} pos.)"), &y, &r, Some(e));
            // the flash-decoding kernels (attn_fd + attn_fd_merge), same inputs; ring = ctx = 256
            let (bsc2, bo2) = (b.gpu.empty_buffer("sc", (32 * 256 * 4) as u64), b.gpu.empty_buffer("o", 4096 * 4));
            let u2 = b.gpu.uniform(&[n as u32, 0, 0, 256, 256, 0, 0, 0]);
            b.gpu.run_b(&fd, &[(0, &bq), (9, &bk), (10, &bv), (4, &bsc2), (7, &u2)], (4, 1, 1));
            b.gpu.run_b(&fdu, &[(3, &bs), (4, &bsc2), (5, &bo2), (7, &u2)], (32, 1, 1));
            let y2 = b.gpu.read_f32(&bo2, 4096);
            let e2 = (err(&y2, &ex), err(&ex, &r), err(&gg, &r));
            b.annotate(&format!("{il} {kind} attn_fd (flash-decoding)"), &y2, &r, Some(e2));
        }
    }
    print(&b.results, positions);
}

/// The prefill GEMMs against the matvec (verified against exact f64) on the same weights
/// and inputs: dense `gemm2_*` and experts `gemmr5_*` (8 slots per token, 6 used, list built here as `gather` does).
/// NB = 64 columns of random x. Error = max|gemm - matvec| / max|matvec| per tensor.
pub fn flint_gemm(il: usize) {
    let g = Gguf::open(crate::model_path()).expect("GGUF");
    let gpu = Gpu::new();
    let src_mv = crate::engine::q3_source("matvec4"); let src_gm = crate::engine::q3_source("gemm");
    let name_t = |t: TType| match t { TType::Q8_0 => "q8_0", TType::Q4_K => "q4_k", TType::Q6_K => "q6_k", TType::Q4_1 => "q4_1", TType::F32 => "f32", x => panic!("{x:?}") };
    let nb = 64usize;
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut rnd = || { state ^= state << 13; state ^= state >> 7; state ^= state << 17; ((state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0) as f32 };
    println!("layer {il} · {} columns · error = max|gemm - matvec| / max|matvec|", nb);
    for s in ["attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate", "ffn_up", "ffn_down"] {
        let name = format!("blk.{il}.{s}.weight");
        let Some(t) = g.tensor(&name).cloned() else { continue };
        let k = t.dims[0] as usize; let rows = t.dims[1] as usize;
        let experts = s.ends_with("_exps");
        let mut raw = Vec::new(); g.read(&t, 0, t.nbytes, &mut raw).unwrap();
        let bw = gpu.buffer_byte("w", &raw);
        let kind = name_t(t.ttype);
        let pm = gpu.pipeline(&src_mv, &format!("mv_{kind}"));
        // experts: token c uses experts (7c + 13j) % 256, slot j < 6; x of slot j at c * x_tok + j * x_step
        let (x_tok, x_step) = if experts && s == "ffn_down_exps" { (8 * k, k) } else if experts { (k, 0) } else { (k, 0) };
        let x: Vec<f32> = (0..nb * x_tok.max(k)).map(|_| rnd()).collect();
        let bx = gpu.buffer_f32("x", &x);
        let used = if experts { 6 } else { 1 };
        let ids_tok = |c: usize| -> Vec<u32> { (0..6).map(|j| ((7 * c + 13 * j) % 256) as u32).collect() };
        // reference: matvec per column (and per slot)
        let mut reference = vec![0f32; nb * 8 * rows];
        for c in 0..nb {
            let ids = ids_tok(c);
            let bid = gpu.buffer_u32("ids", &if experts { ids.clone() } else { vec![0] });
            let by = gpu.empty_buffer("y", (rows * used * 4) as u64);
            let (tpr, nx, ny) = crate::engine::geometry_mv(rows, k);
            let u = gpu.uniform(&[rows as u32, k as u32, nx, experts as u32, x_step as u32, tpr, (c * x_tok) as u32, 0]);
            gpu.run_b(&pm, &[(0, &bw), (1, &bx), (2, &by), (3, &bid), (4, &u)], (nx, ny, used as u32));
            let y = gpu.read_f32(&by, rows * used);
            for j in 0..used { reference[(c * 8 + j) * rows..(c * 8 + j + 1) * rows].copy_from_slice(&y[j * rows..(j + 1) * rows]); }
        }
        // the GEMM
        let by = gpu.empty_buffer("y", (nb * 8 * rows * 4) as u64);
        let out = if experts {
            let mut grp = vec![0u32; 512 + nb * 8];
            let mut lists: Vec<Vec<u32>> = vec![vec![]; 256];
            for c in 0..nb { for (j, e) in ids_tok(c).iter().enumerate() { lists[*e as usize].push(((c as u32) << 3) | j as u32); } }
            let mut start = 0u32;
            for e in 0..256 { grp[2 * e] = start; grp[2 * e + 1] = lists[e].len() as u32; for (q, v) in lists[e].iter().enumerate() { grp[512 + start as usize + q] = *v; } start += lists[e].len() as u32; }
            let bg = gpu.buffer_u32("grp", &grp);
            let p = gpu.pipeline(&src_gm, &format!("gemmr5_{kind}"));
            let u = gpu.uniform(&[rows as u32, k as u32, 0, 2, x_step as u32, 0, nb as u32, x_tok as u32]);
            gpu.run_b(&p, &[(0, &bw), (1, &bx), (2, &by), (3, &bg), (4, &u)], ((rows as u32).div_ceil(128), (nb as u32).div_ceil(16), 256));
            gpu.read_f32(&by, nb * 8 * rows)
        } else {
            let p = gpu.pipeline(&src_gm, &format!("gemm2_{kind}"));
            let id0 = gpu.buffer_u32("id0", &[0; 8]);
            let u = gpu.uniform(&[rows as u32, k as u32, 0, 0, 0, 0, nb as u32, k as u32]);
            gpu.run_b(&p, &[(0, &bw), (1, &bx), (2, &by), (3, &id0), (4, &u)], ((rows as u32).div_ceil(128), (nb as u32).div_ceil(64), 1));
            let y = gpu.read_f32(&by, nb * rows);
            let mut o = vec![0f32; nb * 8 * rows];
            for c in 0..nb { o[c * 8 * rows..c * 8 * rows + rows].copy_from_slice(&y[c * rows..(c + 1) * rows]); }
            o
        };
        // compare the slots that are used
        let (mut a, mut b) = (vec![], vec![]);
        for c in 0..nb { for j in 0..used { a.extend_from_slice(&out[(c * 8 + j) * rows..(c * 8 + j + 1) * rows]); b.extend_from_slice(&reference[(c * 8 + j) * rows..(c * 8 + j + 1) * rows]); } }
        let e = err(&a, &b);
        println!("  {s:16} {:?} {rows}x{k}: {} {e:.2e}", t.ttype, if e <= 1e-5 { "ok" } else { "DIFFERENT" });
    }
}
