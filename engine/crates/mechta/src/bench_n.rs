//! Bench of the n-column matvec (M4): a real tensor, X of n columns, the weights read once.
//! `mechta bench-n <tensor> <n> [repetitions] [a|b|c|g]` · a = matvec_n.wgsl (array per column), b = matvec_n4.wgsl (vec4, always 4 columns),
//! c = matvec_nh.wgsl (x f16), g = gemm.wgsl (M3: 64 x 64 tile, n up to 512). Answers question 1 of M4:
//! how much does an n-column dispatch cost compared with 1? Check: 256 rows (or all) x n columns against the CPU in f64.
//! Experts (`_exps`): 8 fixed matrices for all columns; x per column: gate/up the same x for the 8 experts
//! (x_step 0, x_tok k), down one x per expert (x_step 512, x_tok 8 * 512).
use crate::engine::mv_piece_geometry;
use crate::verify::err;
use mechta_gguf::{Gguf, TType};
use mechta_gpu::{wgpu, Gpu};

const IDS: [u32; 8] = [3, 17, 42, 77, 128, 160, 201, 250];

pub fn bench_n(name: &str, n: usize, rip: usize, variant: &str) {
    assert!(variant.starts_with('g') || (1..=4).contains(&n), "n = {n}: the kernel handles 1..4 columns (variant g: up to 512)");
    let g = Gguf::open(crate::model_path()).expect("GGUF");
    let gpu = Gpu::with_memory(wgpu::MemoryHints::MemoryUsage);
    let t = g.tensor(name).unwrap_or_else(|| panic!("{name}")).clone();
    let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
    let experts = name.contains("_exps");
    let nz = if experts { 8 } else { 1 };
    let x_step = if experts && k == 512 { 512 } else { 0 };
    let x_tok = if experts { if k == 512 { 8 * 512 } else { k } } else { k };
    // rule: the buffers the GPU writes are created first
    let by = gpu.empty_buffer("y", (n * nz * rows * 4) as u64);
    let bo = gpu.empty_buffer("deq", (k * 4) as u64);
    let e = match t.ttype { TType::Q8_0 => "mv_q8_0", TType::Q4_K => "mv_q4_k", TType::Q6_K => "mv_q6_k", TType::F32 => "mv_f32", x => panic!("{x:?}") };
    let gm = crate::engine::q3_source("gemm");
    let src = match variant { "a" => include_str!("wgsl/matvec_n.wgsl"), "b" => include_str!("wgsl/matvec_n4.wgsl"), "c" => include_str!("wgsl/matvec_nh.wgsl"), "g" | "g2" => gm.as_str(), v => panic!("variant {v}") };
    let e = if variant == "g" || variant == "g2" {
        let pre = if variant == "g2" { "gemm2" } else { "gemm" };
        Box::leak(format!("{pre}_{}", &e[3..]).into_boxed_str()) as &str
    } else { e };
    let p = gpu.pipeline(src, e);
    let mut raw = Vec::new(); g.read(&t, 0, t.nbytes, &mut raw).unwrap();
    let bw = gpu.buffer_byte("w", &raw);
    gpu.queue.submit([]); gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mut st = 1u64;
    let x: Vec<f32> = (0..n * x_tok).map(|_| { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5 }).collect();
    // variant c: x in f16 (round to nearest, ties to even), 2 values per u32
    let bx = if variant == "c" {
        let h: Vec<u32> = x.chunks_exact(2).map(|c| u32::from(f16_bits(c[0])) | (u32::from(f16_bits(c[1])) << 16)).collect();
        gpu.buffer_u32("x16", &h)
    } else { gpu.buffer_f32("x", &x) };
    // the CPU reference uses x rounded to f16 when the kernel reads it that way (the error measures the kernel, not the format)
    let x: Vec<f32> = if variant == "c" { x.iter().map(|v| f16_a_f32(f16_bits(*v))).collect() } else { x };
    let bid = gpu.buffer_u32("ids", if experts { &IDS } else { &[0, 1, 2, 3, 4, 5, 6, 7] });
    let (tpr, nx, ny) = mv_piece_geometry(rows, k, 32);
    // variant g (M3, gemm.wgsl): tile of 64 rows x 64 tokens
    let (nx, ny) = match variant { "g" => ((rows as u32).div_ceil(64), (n as u32).div_ceil(64)), "g2" => ((rows as u32).div_ceil(128), (n as u32).div_ceil(64)), _ => (nx, ny) };
    let u = gpu.uniform(&[rows as u32, k as u32, nx, experts as u32, x_step as u32, tpr, n as u32, x_tok as u32]);
    let byte = if experts { 8 * t.nbytes / 256 } else { t.nbytes };
    let cmd: Vec<_> = (0..rip).map(|_| gpu.command(&p, &[(0, &bw), (1, &bx), (2, &by), (3, &bid), (4, &u)], (nx, ny, nz as u32), e.into(), byte)).collect();
    gpu.run_commands(&cmd[..1]);   // warm-up
    let m = gpu.run_commands_measured(&cmd);
    let mut ms = m.gpu_ms.clone(); ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = ms[ms.len() / 2];
    let y = gpu.read_f32(&by, n * nz * rows);
    // check on sample rows, all columns
    let dq = gpu.pipeline(include_str!("wgsl/dequant.wgsl"), match t.ttype { TType::Q8_0 => "q8_0", TType::Q4_K => "q4_k", TType::Q6_K => "q6_k", _ => "q8_0" });
    let (per, bb) = t.ttype.block().unwrap();
    let step = (rows / 256).max(1);
    let (mut gv, mut cv) = (vec![], vec![]);
    for z in 0..nz {
        let mat = if experts { IDS[z] as usize } else { 0 };
        for r in (0..rows).step_by(step) {
            let nb = (k as u64) / per; let b0 = ((mat * rows + r) as u64) * nb;
            let wr: Vec<f32> = if t.ttype == TType::F32 {
                raw[(b0 * 4) as usize..((b0 + nb) * 4) as usize].chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
            } else {
                let br = gpu.buffer_byte("r", &raw[(b0 * bb) as usize..((b0 + nb) * bb) as usize]);
                let uu = gpu.uniform(&[nb as u32, 0, 0, 0]);
                gpu.run(&dq, &[&br, &bo, &uu], ((nb as u32).div_ceil(64), 1, 1));
                gpu.read_f32(&bo, (nb * per) as usize)
            };
            for c in 0..n {
                let x0 = c * x_tok + z * x_step;
                let xz = &x[x0..x0 + k];
                cv.push(wr.iter().zip(xz).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32);
                gv.push(y[(c * nz + z) * rows + r]);
            }
        }
    }
    let gflops = 2.0 * rows as f64 * k as f64 * n as f64 * nz as f64 / (mid * 1e-3) / 1e9;
    println!("{name} ({:?}, {rows} x {k}{}) n = {n} columns, variant {variant} · {rip} dispatch · median {mid:.4} ms (min {:.4}, max {:.4}) · weights {:.1} GB/s · {gflops:.0} GFLOP/s · error vs f64 on {} values {:.2e}",
        t.ttype, if experts { " x 8 experts" } else { "" }, ms[0], ms[ms.len() - 1], byte as f64 / (mid * 1e-3) / 1e9, gv.len(), err(&gv, &cv));
}

/// f32 → f16 bits (round-to-nearest-even), as `f16::from_f32`; no infinities/NaN in the bench inputs.
fn f16_bits(v: f32) -> u16 {
    let b = v.to_bits(); let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32; let m = b & 0x7F_FFFF;
    if e == 0xFF { return s | 0x7C00 | if m != 0 { 0x200 } else { 0 }; }
    let e16 = e - 127 + 15;
    if e16 >= 0x1F { return s | 0x7C00; }
    if e16 <= 0 {
        if e16 < -10 { return s; }
        let m = m | 0x80_0000; let sh = (1 - e16) as u32;
        let mut h = (m >> (sh + 13)) as u16;
        let resto = m & ((1 << (sh + 13)) - 1); let meta = 1u32 << (sh + 12);
        if resto > meta || (resto == meta && (h & 1) == 1) { h += 1; }
        return s | h;
    }
    let mut h = ((e16 as u32) << 10) as u16 | (m >> 13) as u16;
    let resto = m & 0x1FFF;
    if resto > 0x1000 || (resto == 0x1000 && (h & 1) == 1) { h += 1; }
    s | h
}
fn f16_a_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16; let e = ((h >> 10) & 0x1F) as u32; let m = (h & 0x3FF) as u32;
    if e == 0 {
        if m == 0 { return f32::from_bits(s); }
        // subnormal
        let mut e = 1i32; let mut m = m; while m & 0x400 == 0 { m <<= 1; e -= 1; }
        return f32::from_bits(s | (((e - 15 + 127) as u32) << 23) | ((m & 0x3FF) << 13));
    }
    if e == 0x1F { return f32::from_bits(s | 0x7F80_0000 | (m << 13)); }
    f32::from_bits(s | ((e + 127 - 15) << 23) | (m << 13))
}
