//! VRAM versus GTT test: same GPU work, weights in VRAM (V) or in GTT (G).
//! Work: the Q4_K expert matvec (8 matrices) on two real tensors, repeated for `seconds`,
//! in batches of 1000 dispatches. G: first the VRAM is filled with 1.2 GB of empty buffers.
//! At the end it saves the output y (8 x 512 f32, deterministic) in `y.bin` of the data folder,
//! to compare it with any corrupted bytes in the page cache.
use crate::engine::geometry_mv;
use crate::profile::gpu_memory;
use mechta_gguf::Gguf;
use mechta_gpu::{wgpu, Gpu};

pub fn vram(mode: &str, seconds: f64, dir: &str) {
    let g = Gguf::open(crate::model_path()).expect("GGUF");
    let gpu = Gpu::with_memory(wgpu::MemoryHints::MemoryUsage);
    let (v0, g0) = gpu_memory();
    let mut fill = vec![];
    if mode == "G" { for _ in 0..12 { fill.push(gpu.empty_buffer("fill", 100 << 20)); } }
    let (v1, g1) = gpu_memory();
    let mut weights = vec![];
    for n in ["blk.0.ffn_gate_exps.weight", "blk.1.ffn_gate_exps.weight"] {
        let t = g.tensor(n).unwrap().clone(); let mut v = Vec::new(); g.read(&t, 0, t.nbytes, &mut v).unwrap();
        weights.push(gpu.buffer_byte(n, &v));
        gpu.queue.submit([]); gpu.device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    }
    let (v2, g2) = gpu_memory();
    println!("mode {mode} · filler: VRAM +{:.2} GTT +{:.2} GB · weights (0.30 GB): VRAM +{:.2} GTT +{:.2} GB", v1 - v0, g1 - g0, v2 - v1, g2 - g1);
    let mut st = 7u64;
    let x: Vec<f32> = (0..2048).map(|_| { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5 }).collect();
    let bx = gpu.buffer_f32("x", &x);
    let by = gpu.empty_buffer("y", 8 * 512 * 4);
    let bid = gpu.buffer_u32("ids", &[3, 17, 42, 77, 128, 160, 201, 250]);
    let (tpr, nx, ny) = geometry_mv(512, 2048);
    let u = gpu.uniform(&[512, 2048, nx, 1, 0, tpr, 0, 0]);
    let p = gpu.pipeline(include_str!("wgsl/matvec.wgsl"), "mv_q4_k");
    let cmd: Vec<_> = (0..1000).map(|i| gpu.command(&p, &[(0, &weights[i % 2]), (1, &bx), (2, &by), (3, &bid), (4, &u)], (nx, ny, 8), "q4k".into(), 0)).collect();
    let t0 = std::time::Instant::now(); let mut batches = 0;
    while t0.elapsed().as_secs_f64() < seconds { gpu.run_commands(&cmd); batches += 1; }
    let y = gpu.read_f32(&by, 8 * 512);
    let b: Vec<u8> = y.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(std::path::Path::new(dir).join("y.bin"), &b).unwrap();
    println!("{batches} batches of 1000 dispatches in {:.1} s · y[0..4] = {:?}", t0.elapsed().as_secs_f64(), &y[..4]);
    drop(fill);
}
