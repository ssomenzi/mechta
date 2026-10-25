//! `mechta verify-q3k <tensor> <reference.f32>` with MECHTA_MODEL = a GGUF whose `<tensor>` is Q3_K experts.
//! `<reference.f32>`: the first 8 experts decoded by ggml itself (engine/tools/deq_ref.c, to_float). The three Q3_K matvec kernels
//! of the engine (matvec4 for the single step and the drafts, matvec_u and matvec_uh for the experts in union of the
//! double step, the last with x in f16) on experts 0..7 with random x, against W x in f64 on the CPU with the ggml values.
use crate::engine::mv_piece_geometry;
use mechta_gguf::{Gguf, TType};
use mechta_gpu::{wgpu, Gpu};

fn random(n: usize, seed: u64) -> Vec<f32> {
    let mut st = seed;
    (0..n).map(|_| { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5 }).collect()
}

/// max |gpu - cpu| / max |cpu| over all outputs
fn compare(name: &str, gpu: &[f32], cpu: &[f64]) -> f64 {
    let scale = cpu.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let err = gpu.iter().zip(cpu).fold(0.0f64, |a, (&g, &c)| a.max((g as f64 - c).abs()));
    let rel = err / scale.max(1e-30);
    println!("{name:28} {} outputs · max |y| {scale:.4} · max error {err:.3e} · relative {rel:.2e} · {}", gpu.len(), if rel < 1e-3 { "OK" } else { "WRONG" });
    rel
}

pub fn verify_q3k(name: &str, reference: &str) {
    let g = Gguf::open(crate::model_path()).expect("GGUF");
    let t = g.tensor(name).unwrap_or_else(|| panic!("{name}")).clone();
    // Also Q8_0 experts (Ornith's MTP layer); the kernel entry follows the type
    let e = match t.ttype { TType::Q3_K => "mv_q3_k", TType::Q8_0 => "mv_q8_0", x => panic!("{name} is {x:?}") };
    let k = t.dims[0] as usize; let rows = t.dims[1] as usize;
    let w: Vec<f32> = std::fs::read(reference).expect("rif").chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(w.len(), 8 * rows * k, "reference: 8 experts of {rows} x {k}");
    let gpu = Gpu::with_memory(wgpu::MemoryHints::MemoryUsage);
    let bytes_e = t.nbytes / t.dims[2];   // bytes of one expert
    let mut raw = Vec::new(); g.read(&t, 0, 8 * bytes_e, &mut raw).unwrap();
    let bw = gpu.buffer_byte("w", &raw);
    let (tpr, nx, ny) = mv_piece_geometry(rows, k, 32);
    // x: down (k = 512) has one x per expert (x_step 512), gate/up share one x (x_step 0)
    let x_step = if k == 512 { 512 } else { 0 };
    let nx_tok = if k == 512 { 8 * 512 } else { k };
    let x0 = random(nx_tok, 1); let x1 = random(nx_tok, 2);
    let cpu = |x: &[f32]| -> Vec<f64> {
        let mut y = vec![0.0f64; 8 * rows];
        for e in 0..8 { for r in 0..rows {
            let row = &w[(e * rows + r) * k..(e * rows + r + 1) * k];
            let xo = e * x_step;
            y[e * rows + r] = row.iter().enumerate().map(|(i, &v)| v as f64 * x[xo + i] as f64).sum();
        } }
        y
    };
    let (y0, y1) = (cpu(&x0), cpu(&x1));
    let mut worst = 0.0f64;

    // 1. matvec4 (single step, drafts): experts ids 0..7, x of token 0
    {
        let p = gpu.pipeline(&crate::engine::q3_source("matvec4"), e);
        let bx = gpu.buffer_f32("x", &x0);
        let by = gpu.empty_buffer("y", (8 * rows * 4) as u64);
        let ids = gpu.buffer_u32("ids", &[0, 1, 2, 3, 4, 5, 6, 7]);
        let u = gpu.uniform(&[rows as u32, k as u32, nx, 1, x_step as u32, tpr, 0, 0]);
        gpu.run_commands(&[gpu.command(&p, &[(0, &bw), (1, &bx), (2, &by), (3, &ids), (4, &u)], (nx, ny, 8), e.into(), 0)]);
        worst = worst.max(compare(&format!("matvec4 {e}"), &gpu.read_f32(&by, 8 * rows), &y0));
    }
    // 2./3. experts in union, two tokens that both use experts 0..7 (slot z for both): x of token 1 at x_tok
    let uniforms: Vec<u32> = std::iter::once(8u32).chain((0..8u32).flat_map(|z| [z, z, z])).collect();
    let bu = gpu.buffer_u32("uniforms", &uniforms);
    let x_tok = nx_tok as u32;
    let y_tok = 8 * rows as u32;
    let u = gpu.uniform(&[rows as u32, k as u32, nx, y_tok, x_step as u32, tpr, 2, x_tok]);
    let both: Vec<f32> = x0.iter().chain(&x1).cloned().collect();
    let want: Vec<f64> = y0.iter().chain(&y1).cloned().collect();
    {
        let p = gpu.pipeline(&crate::engine::q3_source("matvec_u"), e);
        let bx = gpu.buffer_f32("x", &both);
        let by = gpu.empty_buffer("y", (2 * 8 * rows * 4) as u64);
        gpu.run_commands(&[gpu.command(&p, &[(0, &bw), (1, &bx), (2, &by), (3, &bu), (4, &u)], (nx, ny, 16), e.into(), 0)]);
        worst = worst.max(compare(&format!("matvec_u {e} (2 tokens)"), &gpu.read_f32(&by, 2 * 8 * rows), &want));
    }
    {
        // x in f16, 4 values per vec2<u32>; the CPU side uses the same f16-rounded x
        let h: Vec<u8> = both.iter().flat_map(|v| half::f16::from_f32(*v).to_le_bytes()).collect();
        let xr: Vec<f32> = both.iter().map(|v| half::f16::from_f32(*v).to_f32()).collect();
        let (xr0, xr1) = xr.split_at(nx_tok);
        let want_h: Vec<f64> = cpu(xr0).into_iter().chain(cpu(xr1)).collect();
        let p = gpu.pipeline(&crate::engine::q3_source("matvec_uh"), e);
        let bx = gpu.buffer_byte("xh", &h);
        let by = gpu.empty_buffer("y", (2 * 8 * rows * 4) as u64);
        gpu.run_commands(&[gpu.command(&p, &[(0, &bw), (1, &bx), (2, &by), (3, &bu), (4, &u)], (nx, ny, 16), e.into(), 0)]);
        worst = worst.max(compare(&format!("matvec_uh {e} (2 tok, f16)"), &gpu.read_f32(&by, 2 * 8 * rows), &want_h));
    }
    // 4. dequant.wgsl q3_k / q8_0 (used by `bench-experts` to check gemmr5_q3_k) against ggml, value by value
    {
        let p = gpu.pipeline(include_str!("wgsl/dequant.wgsl"), if t.ttype == TType::Q3_K { "q3_k" } else { "q8_0" });
        let nb = (8 * rows * k / t.ttype.block().unwrap().0 as usize) as u32;
        let bo = gpu.empty_buffer("deq", (8 * rows * k * 4) as u64);
        let uu = gpu.uniform(&[nb, 0, 0, 0]);
        gpu.run(&p, &[&bw, &bo, &uu], (nb.div_ceil(64), 1, 1));
        let d = gpu.read_f32(&bo, 8 * rows * k);
        let diff = d.iter().zip(&w).filter(|(a, b)| a.to_bits() != b.to_bits()).count();
        println!("{:28} {} values · different from ggml: {diff} · {}", "dequant.wgsl", d.len(), if diff == 0 { "OK" } else { "WRONG" });
        if diff > 0 { worst = 1.0; }
    }
    println!("RESULT {name}: {}", if worst < 1e-3 { "ALL OK" } else { "WRONG" });
}
