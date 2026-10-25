//! GPU latency probe: every 16 ms it sends a tiny job and measures how long it waits.
//! It imitates another program (browser, compositor) that wants the GPU while mechta works.
//! Usage: `probe-gpu <seconds>`; prints `ms_since_epoch,wait_ms` on stdout.
#![forbid(unsafe_code)]
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const WGSL: &str = "@group(0) @binding(0) var<storage, read_write> b: array<u32>;
@compute @workgroup_size(1) fn probe() { b[0] = b[0] + 1u; }";

fn main() {
    let seconds: u64 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let g = mechta_gpu::Gpu::new();
    let p = g.pipeline(WGSL, "probe");
    let b = g.empty_buffer("probe", 4);
    let end = Instant::now() + Duration::from_secs(seconds);
    while Instant::now() < end {
        let t0 = Instant::now();
        g.run(&p, &[&b], (1, 1, 1));   // submit + wait (sleeping poll)
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let now_ms = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis();
        println!("{now_ms},{ms:.2}");
        if let Some(resto) = Duration::from_millis(16).checked_sub(t0.elapsed()) { std::thread::sleep(resto); }
    }
}
