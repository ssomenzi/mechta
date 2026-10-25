//! Check for the page cache corruption: reads the GGUF as `Engine::new` does
//! (tensor by tensor, same `read` calls), but does not open the GPU. Then waits as long as a `generate` run.
use mechta_gguf::Gguf;

pub fn read_only(wait_s: u64) {
    let g = Gguf::open(crate::model_path()).expect("GGUF");
    let t0 = std::time::Instant::now(); let mut tot = 0u64; let mut sum = 0u64;
    for t in g.tensors.clone() {
        if t.name.starts_with("blk.40.") || t.name == "token_embd.weight" { continue; }
        let mut v = Vec::new(); g.read(&t, 0, t.nbytes, &mut v).expect("read");
        tot += v.len() as u64; sum = sum.wrapping_add(v.iter().step_by(4096).map(|b| *b as u64).sum::<u64>());
    }
    println!("read {:.2} GB in {:.1} s, no GPU (checksum {sum}); waiting {wait_s} s", tot as f64 / 1e9, t0.elapsed().as_secs_f64());
    std::thread::sleep(std::time::Duration::from_secs(wait_s));
}
