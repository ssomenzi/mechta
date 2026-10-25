//! Small test (M2): how much GTT a buffer created the way the engine does it takes, and when it is freed.
//! 8 buffers of 150 MB. Mode A: create_buffer_init (as `buffer_byte`). Mode B: empty buffer + write_buffer.
use crate::profile::gpu_memory;
use mechta_gpu::{Gpu, wgpu};

pub fn memtest(mode: &str, hint: &str) {
    let gpu = Gpu::with_memory(hint_memory(hint));
    println!("memory: {hint}");
    let data = vec![1u8; 150 << 20];
    let (v0, g0) = gpu_memory();
    println!("mode {mode} · VRAM {v0:.2} GB · GTT at start {g0:.2} GB");
    let mut keep = vec![];
    for i in 0..8 {
        let b = if mode == "A" { gpu.buffer_byte("test", &data) } else if mode == "C" { gpu.empty_buffer("test", data.len() as u64) } else {
            let b = gpu.empty_buffer("test", data.len() as u64); gpu.write(&b, 0, &data); b
        };
        keep.push(b);
        let (v, g) = gpu_memory();
        println!("after buffer {}: VRAM +{:.2} GTT +{:.2} GB · buffer sum {:.2} GB (buffers of 0.157)", i + 1, v - v0, g - g0, (v - v0 + g - g0) / (i + 1) as f64);
    }
    gpu.queue.submit([]);
    gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
    for k in 0..3 {
        gpu.queue.submit([]);
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        std::thread::sleep(std::time::Duration::from_millis(500));
        let (v, g) = gpu_memory();
        println!("after empty submit + poll + 0.5 s (round {k}): VRAM +{:.2} GTT +{:.2} GB", v - v0, g - g0);
    }
}

/// "default" = Performance, "usage" = MemoryUsage, "manual" = suballocated blocks from 1 to 64 MiB.
pub fn hint_memory(h: &str) -> wgpu::MemoryHints {
    match h { "usage" => wgpu::MemoryHints::MemoryUsage,
              "manual" => wgpu::MemoryHints::Manual { suballocated_device_memory_block_size: (1 << 20)..(64 << 20) },
              _ => wgpu::MemoryHints::Performance }
}
