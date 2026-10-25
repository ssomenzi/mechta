//! Bench of the experts GEMM in the prefill (M3): a real `_exps` tensor, n tokens with 8 randomly chosen experts each,
//! per-expert lists as `gather` builds them, random x. `mechta bench-experts <tensor> <n> [reps] [r|16|8|64]`.
//! Check against the CPU in f64 on sample experts (all their sample rows x their tokens).
//! Answers: how much the experts kernel costs on its own, and which variant is better.
use crate::engine::NB;
use crate::verify::err;
use mechta_gguf::{Gguf, TType};
use mechta_gpu::{wgpu, Gpu};

pub fn bench_experts(name: &str, n: usize, rip: usize, variant: &str) {
    assert!(name.contains("_exps"), "needs an expert tensor");
    assert!(n <= NB && n % 8 == 0);
    let g = Gguf::open(crate::model_path()).expect("GGUF");
    let gpu = Gpu::with_memory(wgpu::MemoryHints::MemoryUsage);
    let t = g.tensor(name).unwrap_or_else(|| panic!("{name}")).clone();
    let k = t.dims[0] as usize; let rows = t.dims[1] as usize; let ne = t.dims[2] as usize; assert_eq!(ne, 256);
    let down = k == 512;   // down: x per (token, slot) at tok * 4096 + slot * 512; gate/up: x per token at tok * 2048
    let (x_step, x_tok) = if down { (512u32, 4096u32) } else { (0u32, k as u32) };
    // fake routing: 8 distinct experts per token (linear generator)
    let mut st = 7u64;
    let mut rnd = || { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (st >> 33) as u32 };
    let mut ids = vec![0u32; n * 8];
    for tok in 0..n { let mut s = 0; while s < 8 { let e = rnd() % 256; if !ids[tok * 8..tok * 8 + s].contains(&e) { ids[tok * 8 + s] = e; s += 1; } } }
    // grp as `gather` builds it: [start, count] at 2e, list at 512 + start ((tok << 3) | slot in token order)
    let mut grp = vec![0u32; 512 + n * 8];
    let mut start = 0;
    for e in 0..256u32 {
        let entries: Vec<u32> = (0..n * 8).filter(|&i| ids[i] == e).map(|i| ((i / 8) << 3 | (i % 8)) as u32).collect();
        grp[2 * e as usize] = start; grp[2 * e as usize + 1] = entries.len() as u32;
        for (j, v) in entries.iter().enumerate() { grp[512 + start as usize + j] = *v; }
        start += entries.len() as u32;
    }
    let touched = (0..256).filter(|&e| grp[2 * e + 1] > 0).count();
    let nx = if down { n * 4096 } else { n * k };
    let x: Vec<f32> = (0..nx).map(|_| (rnd() as f32 / (1u64 << 31) as f32) - 0.5).collect();
    let by = gpu.empty_buffer("y", (n * 8 * rows * 4) as u64);
    let bo = gpu.empty_buffer("deq", (k * 4) as u64);
    let mut raw = Vec::new(); g.read(&t, 0, t.nbytes, &mut raw).unwrap();
    let bw = gpu.buffer_byte("w", &raw);
    let bx = gpu.buffer_f32("x", &x);
    let bg = gpu.buffer_u32("grp", &grp);
    gpu.queue.submit([]); gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let suf = match t.ttype { TType::Q4_K => "q4_k", TType::Q6_K => "q6_k", TType::Q8_0 => "q8_0", TType::Q3_K => "q3_k", x => panic!("{x:?}") };   // Q3_K: variant r5 only
    let (pre, bm, bn) = match variant { "r" => ("gemmr", 64, 16), "r2" => ("gemmr2", 64, 16), "r3" => ("gemmr3", 128, 16), "r4" => ("gemmr4", 256, 16), "r5" => ("gemmr5", 128, 16), "16" => ("gemm16", 128, 16), "8" => ("gemm8", 64, 8), "64" => ("gemm", 64, 64), v => panic!("variant {v}") };
    let e = format!("{pre}_{suf}");
    let p = gpu.pipeline(&crate::engine::q3_source("gemm"), &e);
    let mode: u32 = std::env::var("MECHTA_GEMM_PHASE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);   // 3/4: fake weights (measurement only)
    let u = gpu.uniform(&[rows as u32, k as u32, mode, 2, x_step, 0, n as u32, x_tok]);
    let byte = touched as u64 * (t.nbytes / 256);
    let cmd: Vec<_> = (0..rip).map(|_| gpu.command(&p, &[(0, &bw), (1, &bx), (2, &by), (3, &bg), (4, &u)], ((rows as u32).div_ceil(bm), (n as u32).div_ceil(bn), 256), e.clone(), byte)).collect();
    gpu.run_commands(&cmd[..1]);
    let m = gpu.run_commands_measured(&cmd);
    let mut ms = m.gpu_ms.clone(); ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = ms[ms.len() / 2];
    let y = gpu.read_f32(&by, n * 8 * rows);
    // check: 4 sample experts, 32 rows each, all their tokens
    let dq = gpu.pipeline(include_str!("wgsl/dequant.wgsl"), suf);
    let (per, bb) = t.ttype.block().unwrap();
    let (mut gv, mut cv) = (vec![], vec![]);
    let sample: Vec<usize> = (0..256).filter(|&e| grp[2 * e + 1] > 0).step_by(61).take(4).collect();
    for &e in &sample {
        let (i0, q) = (grp[2 * e] as usize, grp[2 * e + 1] as usize);
        for r in (0..rows).step_by((rows / 32).max(1)) {
            let nb = (k as u64) / per; let b0 = ((e * rows + r) as u64) * nb;
            let br = gpu.buffer_byte("r", &raw[(b0 * bb) as usize..((b0 + nb) * bb) as usize]);
            let uu = gpu.uniform(&[nb as u32, 0, 0, 0]);
            gpu.run(&dq, &[&br, &bo, &uu], ((nb as u32).div_ceil(64), 1, 1));
            let wr = gpu.read_f32(&bo, (nb * per) as usize);
            for j in 0..q {
                let v = grp[512 + i0 + j] as usize; let (tok, slot) = (v >> 3, v & 7);
                let x0 = tok * x_tok as usize + slot * x_step as usize;
                cv.push(wr.iter().zip(&x[x0..x0 + k]).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32);
                gv.push(y[(tok * 8 + slot) * rows + r]);
            }
        }
    }
    let gflops = 2.0 * n as f64 * 8.0 * rows as f64 * k as f64 / (mid * 1e-3) / 1e9;
    println!("{name} ({:?}, {rows} x {k}) n = {n} tokens, {touched} experts touched ({:.1} tokens/expert), variant {variant} ({e}) · {rip} dispatch · median {mid:.3} ms (min {:.3}) · touched weights {:.1} GB/s · {gflops:.0} GFLOP/s · error vs f64 on {} values {:.2e}",
        t.ttype, n as f64 * 8.0 / touched as f64, ms[0], byte as f64 / (mid * 1e-3) / 1e9, gv.len(), err(&gv, &cv));
}
