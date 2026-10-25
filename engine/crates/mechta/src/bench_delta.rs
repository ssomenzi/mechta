//! Bench of the `delta` kernel (M2): 2 MiB state, fixed random inputs, 50 dispatches measured with the GPU timestamps.
//! `mechta bench-delta <variant> [repetitions]` · variant 1 = current kernel, 2 = transposed state, 3 = transposed + registers.
//! Check: the final state and the output against the CPU in f64 after the same number of steps (warm-up + measurements).
use crate::verify::err;
use mechta_gpu::{wgpu, Gpu};

const H: usize = 32; const D: usize = 128;

pub fn bench_delta(variant: &str, rip: usize) {
    let gpu = Gpu::with_memory(wgpu::MemoryHints::MemoryUsage);
    let mut st = 7u64;
    let mut rnd = move || { st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407); (st >> 40) as f64 / (1u64 << 24) as f64 };
    let s_ini: Vec<f32> = (0..H * D * D).map(|_| (rnd() - 0.5) as f32).collect();            // S[i][j], index i*128 + j (per head)
    // q | k: in the model they are normalized to norm 1 per head (l2 norm). Without that, the state diverges:
    // S^T k += |k|^2 beta (v - S^T k), and with |k|^2 ~ 10 the step overshoots the target (first draft: NaN).
    let mut a: Vec<f32> = (0..4096).map(|_| (rnd() - 0.5) as f32).collect();
    for t in a.chunks_exact_mut(128) { let n = t.iter().map(|v| v * v).sum::<f32>().sqrt(); for v in t.iter_mut() { *v /= n; } }
    let b: Vec<f32> = (0..8192).map(|_| (rnd() - 0.5) as f32).collect();                      // v of 4096
    let c: Vec<f32> = (0..64).map(|i| if i < 32 { rnd() as f32 } else { -(rnd() as f32) }).collect(); // beta in (0,1), log g in (-1,0)
    let transposed = variant != "1";
    // state layout for the kernel: v1 o[h][j][i], v2/v3 o[h][i][j]
    let arrange = |s: &[f32]| -> Vec<f32> {
        if transposed { return s.to_vec(); }
        let mut o = vec![0f32; H * D * D];
        for h in 0..H { for i in 0..D { for j in 0..D { o[h * D * D + j * D + i] = s[h * D * D + i * D + j]; } } }
        o
    };
    // rule: the buffers the GPU writes are created first
    let bo = gpu.buffer_f32("state", &arrange(&s_ini));
    let bo2 = gpu.empty_buffer("o2", (H * D * 4) as u64);
    let ba = gpu.buffer_f32("a", &a); let bb = gpu.buffer_f32("b", &b); let bc = gpu.buffer_f32("c", &c);
    let e = match variant { "1" => "delta1", "2" => "delta2", "3" => "delta3", v => panic!("variant {v}") };
    let p = gpu.pipeline(include_str!("wgsl/delta.wgsl"), e);
    let byte = (H * D * D * 4 * 2) as u64;   // minimum: one read and one write of the state
    let cmd: Vec<_> = (0..rip).map(|_| gpu.command(&p, &[(0, &ba), (1, &bb), (2, &bc), (3, &bo), (4, &bo2)], (H as u32, 1, 1), e.into(), byte)).collect();
    gpu.run_commands(&cmd[..1]);   // warm-up
    let m = gpu.run_commands_measured(&cmd);
    let mut ms = m.gpu_ms.clone(); ms.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mid = ms[ms.len() / 2];
    let s_gpu = gpu.read_f32(&bo, H * D * D); let out_gpu = gpu.read_f32(&bo2, H * D);
    // CPU reference in f64: rip + 1 steps
    let mut s: Vec<f64> = s_ini.iter().map(|&v| v as f64).collect();
    let mut out = vec![0f64; H * D];
    for _ in 0..rip + 1 {
        for h in 0..H {
            let kh = h % 16; let q = &a[kh * D..kh * D + D]; let k = &a[2048 + kh * D..2048 + kh * D + D];
            let eg = (c[32 + h] as f64).exp(); let beta = c[h] as f64;
            let sh = &mut s[h * D * D..(h + 1) * D * D];
            for v in sh.iter_mut() { *v *= eg; }
            for j in 0..D {
                let sk: f64 = (0..D).map(|i| sh[i * D + j] * k[i] as f64).sum();
                let dl = (b[4096 + h * D + j] as f64 - sk) * beta;
                for i in 0..D { sh[i * D + j] += k[i] as f64 * dl; }
                out[h * D + j] = (0..D).map(|i| sh[i * D + j] * q[i] as f64).sum::<f64>() / (D as f64).sqrt();
            }
        }
    }
    let s_reference: Vec<f32> = arrange(&s.iter().map(|&v| v as f32).collect::<Vec<_>>());
    let out_reference: Vec<f32> = out.iter().map(|&v| v as f32).collect();
    println!("delta variant {variant} · {rip} dispatch · median {mid:.4} ms (min {:.4}, max {:.4}) · {:.1} GB/s (2 x 2 MiB) · error vs f64: state {:.2e}, output {:.2e}",
        ms[0], ms[ms.len() - 1], byte as f64 / (mid * 1e-3) / 1e9, err(&s_gpu, &s_reference), err(&out_gpu, &out_reference));
}
