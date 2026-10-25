//! Step 4 of M1: the whole model, one token at a time, against the llama.cpp dump.
//! Prompt: the 24 tokens of the dump. Then 32 tokens chosen by mechta (the most probable) and fed back as input.
//! For each position: error of the logits and of x at the end of each layer (here the errors add up from one layer
//! to the next, as in real use), mechta's token against llama.cpp's.
use crate::engine::{Engine, VOCAB};
use crate::verify::{err, token_ids, Dump};

fn argmax2(v: &[f32]) -> (usize, f32, f32) {
    let (mut i1, mut v1, mut v2) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, x) in v.iter().enumerate() { if *x > v1 { v2 = v1; v1 = *x; i1 = i; } else if *x > v2 { v2 = *x; } }
    (i1, v1, v1 - v2)
}

pub fn generate() {
    let t0 = std::time::Instant::now();
    let (v0, g0) = crate::profile::gpu_memory();
    let m = Engine::new(&crate::model_path());
    let (v1, g1) = crate::profile::gpu_memory();
    println!("VRAM {v0:.2} -> {v1:.2} GB · GTT {g0:.2} -> {g1:.2} GB");
    println!("GPU {} · weights on the GPU {:.2} GB · loaded in {:.1} s", m.gpu.info.name, m.byte_gpu as f64 / 1e9, t0.elapsed().as_secs_f64());
    let mut d = Dump::load();
    let reference = token_ids();               // 56 llama.cpp tokens: 24 of prompt + 32 generated
    let n_prompt = 24;
    m.reset();
    let mut input = reference[0];
    let mut my_tokens: Vec<u32> = vec![];
    let mut first_different: Option<usize> = None;
    let mut argmax_gpu_different = 0;
    println!("{:>3} {:>7} | {:>7} {:>7} {:>6} | {:>9} {:>9} {:>9} | {:>6}", "pos", "input", "mechta", "llama", "same", "logit err", "l_out max", "l_out 39", "ms");
    // M3: with MECHTA_PREFILL=blocks the first n_prompt - 1 tokens go through in one block; l_out-39 of each position of the block
    // is compared with the dump, then it goes on one token at a time from position n_prompt - 1 (whose logit depends on the whole block).
    let mut start = 0;
    if Engine::prefill_in_blocks() {
        let t = std::time::Instant::now();
        let np = n_prompt - 1;
        // layer by layer: maximum error over the positions against the dump (embedding = "inp_embd"? no: it starts from layer 0)
        let mut cb = |il: usize, x: Vec<f32>| {
            if il == 0 { return; }
            let e: Vec<f64> = (0..np).map(|pos| err(&x[pos * 2048..(pos + 1) * 2048], &d.get(&format!("l_out-{}", il - 1), pos as i32, 0))).collect();
            let emax = e.iter().cloned().fold(0f64, f64::max);
            println!("  prefill layer {:>2}: max error {emax:.2e} · first positions {}", il - 1, e.iter().take(6).map(|v| format!("{v:.1e}")).collect::<Vec<_>>().join(" "));
        };
        m.prefill_verify(&reference[..np], 0, &mut cb);
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let x = m.prefill_outputs(n_prompt - 1);
        let errs: Vec<f64> = (0..n_prompt - 1).map(|pos| err(&x[pos * 2048..(pos + 1) * 2048], &d.get("l_out-39", pos as i32, 0))).collect();
        let emax = errs.iter().cloned().fold(0f64, f64::max);
        println!("prefill in blocks of {} tokens: {} tokens in {ms:.0} ms · max error of l_out-39 over the {} positions {emax:.2e} (per position: {})",
            crate::engine::NB, n_prompt - 1, n_prompt - 1, errs.iter().map(|e| format!("{e:.1e}")).collect::<Vec<_>>().join(" "));
        start = n_prompt - 1; input = reference[start];
    }
    for pos in start..reference.len() {
        let t = std::time::Instant::now();
        let mut e_l = vec![0f64; 40];
        let mut cb = |il: usize, x: Vec<f32>| { e_l[il] = err(&x, &d.get(&format!("l_out-{il}"), pos as i32, 0)); };
        let logits = m.token(input, pos, Some(&mut cb));
        let ms = t.elapsed().as_secs_f64() * 1e3;
        assert_eq!(logits.len(), VOCAB);
        let r = d.get("result_output", pos as i32, 0);
        let (mine, _, margin) = argmax2(&logits);
        let (ag, _) = m.argmax_gpu();
        if ag as usize != mine { argmax_gpu_different += 1; }
        let (theirs, _, _) = argmax2(&r);
        let same = mine == theirs;
        if !same && first_different.is_none() { first_different = Some(pos); }
        let emax = e_l.iter().cloned().fold(0f64, f64::max);
        println!("{pos:>3} {input:>7} | {mine:>7} {theirs:>7} {:>6} | {:>9.2e} {emax:>9.2e} {:>9.2e} | {ms:>6.0}  (margin {margin:.3})",
            if same { "si" } else { "NO" }, err(&logits, &r), e_l[39]);
        // input of the next step: within the prompt the prompt token, then the one chosen by mechta
        if pos + 1 < reference.len() {
            input = if pos + 1 < n_prompt { reference[pos + 1] } else { my_tokens.push(mine as u32); mine as u32 };
        } else if pos + 1 >= n_prompt { my_tokens.push(mine as u32); }
    }
    let generated_llama = &reference[n_prompt..];
    let equal = my_tokens.iter().zip(generated_llama).take_while(|(a, b)| a == b).count();
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    println!("\ngenerated tokens equal to llama.cpp, from the start: {equal} of {} · first logit with a different argmax: {:?}", generated_llama.len().min(my_tokens.len()), first_different);
    println!("GPU argmax different from the CPU one: {argmax_gpu_different} positions of {}", reference.len());
    println!("mechta text: {:?}", tk.decode(&my_tokens, false).unwrap());
    println!("total time {:.1} s", t0.elapsed().as_secs_f64());
}
