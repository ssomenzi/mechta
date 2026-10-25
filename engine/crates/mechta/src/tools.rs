//! Small tools of the engine: GPU memory in use, and `logit-dump` (the logits of a run, compared with `mechta kl`
//! by tools/numeric-check.sh).
use crate::engine::Engine;
use std::time::Instant;

/// GPU memory in use (amdgpu, sysfs): (VRAM, GTT) in GB.
pub fn gpu_memory() -> (f64, f64) {
    let read = |f: &str| -> f64 {
        for c in 0..4 { if let Ok(s) = std::fs::read_to_string(format!("/sys/class/drm/card{c}/device/{f}")) { return s.trim().parse::<f64>().unwrap_or(0.0) / 1e9; } }
        0.0
    };
    (read("mem_info_vram_used"), read("mem_info_gtt_used"))
}

/// `logit-dump <text> <prefill> <steps> <out.bin> [single|double]`: like `quality-kv`, for
/// the engine as configured by the environment (MECHTA_KV, ...). Prefill
/// of <prefill> tokens of real text, then <steps> steps fed with the true tokens; per step the true next token and the
/// logits, for `mechta kl`. `single` (default): `step`, one token per step. `double`: `step2` on pairs of true tokens,
/// the double step of MTP generation (both logits rows written), so a change of the generation kernels is covered too.
pub fn logit_dump(text: &str, prefill: usize, steps: usize, out: &str, double: bool) {
    use std::io::Write;
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(text).expect("text"), false).expect("encode").get_ids().to_vec();
    assert!(ids.len() > prefill + steps + 2, "text too short: {} tokens", ids.len());
    m.reset();
    let t = Instant::now(); m.prefill(&ids[..prefill], 0);
    println!("kv={} prefill {prefill} tokens in {:.1} s · mode {}", crate::engine::kv_format().name(), t.elapsed().as_secs_f64(),
        if double { "double" } else { "single" });
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).expect("out"));
    let mut put = |target: u32, l: &[f32]| { f.write_all(&target.to_le_bytes()).unwrap(); for x in l { f.write_all(&x.to_le_bytes()).unwrap(); } };
    let mut j = 0;
    while j < steps {
        let pos = prefill + j;
        if double {
            m.step2(ids[pos], ids[pos + 1], pos);
            let l = m.read("logits", 2 * crate::engine::VOCAB);
            put(ids[pos + 1], &l[..crate::engine::VOCAB]);
            put(ids[pos + 2], &l[crate::engine::VOCAB..]);
            j += 2;
        } else {
            m.step(ids[pos], pos);
            put(ids[pos + 1], &m.read("logits", crate::engine::VOCAB));
            j += 1;
        }
    }
    println!("{j} steps written to {out}");
}
