//! Bench of a single matvec (M2): a real tensor, fixed x, the same geometry as the engine.
//! `mechta bench <tensor> <variant> [repetitions]` · variant 1 = matvec.wgsl, 2 = matvec2.wgsl, 3 = matvec3.wgsl (Q4_K as vec4),
//! 4 = matvec4.wgsl (+ Q8_0 as words, unpack4xI8), 5 = matvec4 with hand-written int8 (Q8_0 only), 6 = matvec4 Q4_K in pieces of 64 (Q4_K only),
//! 0 = matvec4 read-only (ceiling; Q8_0 and Q6_K).
//! Time: GPU timestamp of every dispatch (one step per dispatch), median and min-max.
//! Check: 256 rows (or all, if fewer) against the CPU in f64, with W decoded by the kernel of step 2.
//! Experts (`_exps`): 8 fixed matrices, x_step 0 (gate/up) or 512 (down).
use crate::engine::mv_piece_geometry;
use crate::verify::err;
use mechta_gguf::{Gguf, TType};
use mechta_gpu::{wgpu, Gpu};

const IDS: [u32; 8] = [3, 17, 42, 77, 128, 160, 201, 250];

/// Experts per dispatch: 8 as in the engine, or `MECHTA_BENCH_EXPERTS` (up to 256) to see how much the size matters.
fn bench_ids() -> Vec<u32> {
    let n: usize = std::env::var("MECHTA_BENCH_EXPERTS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    if n == 8 { return IDS.to_vec(); }
    assert!((1..=256).contains(&n), "MECHTA_BENCH_EXPERTS = {n}");
    (0..n as u32).map(|i| (i * 37 + 3) % 256).collect()   // 37 and 256 are coprime: all distinct
}

pub fn bench(name: &str, variant: &str, rip: usize) {
    let g = Gguf::open(crate::model_path()).expect("GGUF");
    let gpu = Gpu::with_memory(wgpu::MemoryHints::MemoryUsage);
    let t = g.tensor(name).unwrap_or_else(|| panic!("{name}")).clone();
    let k = t.dims[0] as usize; let rows = *t.dims.get(1).unwrap_or(&1) as usize;
    let experts = name.contains("_exps");
    // rule: the buffers the GPU WRITES are created first (in VRAM), before the weights
    let ids = bench_ids();
    let nz0 = if experts { ids.len() } else { 1 };
    let m0 = crate::profile::gpu_memory();
    let by = gpu.empty_buffer("y", (rows * nz0 * 4) as u64);
    let bo = gpu.empty_buffer("deq", (k * 4) as u64);
    let m1 = crate::profile::gpu_memory();
    let mv4 = crate::engine::q3_source("matvec4");
    let src = match variant { "1" => include_str!("wgsl/matvec.wgsl"), "2" => include_str!("wgsl/matvec2.wgsl"), "3" => include_str!("wgsl/matvec3.wgsl"),
                               "4" | "5" | "6" | "0" => mv4.as_str(), v => panic!("variant {v}") };
    let mut e = match t.ttype { TType::Q8_0 => "mv_q8_0", TType::Q4_K => "mv_q4_k", TType::Q6_K => "mv_q6_k", TType::F32 => "mv_f32", x => panic!("{x:?}") };
    // special variants of matvec4: 5 = Q8_0 hand-written int8, 0 = read-only (Q8_0 or Q6_K) (the check against the CPU fails on purpose)
    if variant == "5" { assert_eq!(t.ttype, TType::Q8_0, "variant 5 only for Q8_0"); e = "mv_q8_0b"; }
    let mut piece = 32;
    if variant == "6" { assert_eq!(t.ttype, TType::Q4_K, "variant 6 only for Q4_K"); e = "mv_q4_k64"; piece = 64; }
    if variant == "0" { e = match t.ttype { TType::Q8_0 => "mv_read", TType::Q6_K => "mv_read6", x => panic!("variant 0 does not exist for {x:?}") }; }
    let p = gpu.pipeline(src, e);
    let mut raw = Vec::new(); g.read(&t, 0, t.nbytes, &mut raw).unwrap();
    let bw = gpu.buffer_byte("w", &raw);
    gpu.queue.submit([]); gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let m2 = crate::profile::gpu_memory();
    eprintln!("written buffers: VRAM +{:.4} GTT +{:.4} GB · weights: VRAM +{:.3} GTT +{:.3} GB", m1.0 - m0.0, m1.1 - m0.1, m2.0 - m1.0, m2.1 - m1.1);
    let nz = nz0;
    let x_step = if experts && k == 512 { 512 } else { 0 };
    let mut st = 1u64;
    let x: Vec<f32> = (0..k * nz).map(|_| { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5 }).collect();
    let bx = gpu.buffer_f32("x", &x);
    let bid = gpu.buffer_u32("ids", if experts { &ids } else { &[0, 1, 2, 3, 4, 5, 6, 7] });
    let (tpr, nx, ny) = mv_piece_geometry(rows, k, piece);
    let u = gpu.uniform(&[rows as u32, k as u32, nx, experts as u32, x_step as u32, tpr, 0, 0]);
    let byte = if experts { nz as u64 * t.nbytes / 256 } else { t.nbytes };
    let cmd: Vec<_> = (0..rip).map(|_| gpu.command(&p, &[(0, &bw), (1, &bx), (2, &by), (3, &bid), (4, &u)], (nx, ny, nz as u32), e.into(), byte)).collect();
    gpu.run_commands(&cmd[..1]);   // warm-up
    let m = gpu.run_commands_measured(&cmd);
    let mut ms = m.gpu_ms.clone(); ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = ms[ms.len() / 2];
    let y = gpu.read_f32(&by, rows * nz);
    // check on sample rows
    let dq = gpu.pipeline(include_str!("wgsl/dequant.wgsl"), match t.ttype { TType::Q8_0 => "q8_0", TType::Q4_K => "q4_k", TType::Q6_K => "q6_k", _ => "q8_0" });
    let (per, bb) = t.ttype.block().unwrap();
    let step = (rows / 256).max(1);
    let (mut gv, mut cv) = (vec![], vec![]);
    for z in 0..nz.min(8) {   // check on the first 8 experts: enough, and it stays fast
        let mat = if experts { ids[z] as usize } else { 0 };
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
            let xz = &x[z * x_step..z * x_step + k];
            cv.push(wr.iter().zip(xz).map(|(a, b)| *a as f64 * *b as f64).sum::<f64>() as f32);
            gv.push(y[z * rows + r]);
        }
    }
    println!("{name} ({:?}, {rows} x {k}{}) variant {variant} · {rip} dispatch · median {mid:.4} ms (min {:.4}, max {:.4}) · {:.1} GB/s · error vs f64 on {} rows {:.2e}",
        t.ttype, if experts { format!(" x {nz} experts") } else { String::new() }, ms[0], ms[ms.len() - 1], byte as f64 / (mid * 1e-3) / 1e9, gv.len(), err(&gv, &cv));
}
