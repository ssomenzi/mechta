//! M4: generation with MTP (one draft per step), against plain generation, on the same 24 prompt tokens
//! of the M1 dump. Greedy: the 32 generated tokens must be IDENTICAL to those of `generate` (MTP does not change the output,
//! it only changes how many times the weights are read). Prints tokens/s of both, acceptance, and the comparison.
//! Loop: double step (accepted token at pos, draft at pos+1) → if the draft is right 2 tokens are kept and the MTP
//! receives the two new tokens (draft2); otherwise the model's token is kept, the states after the
//! first token are restored (restore) and the MTP receives that token (draft).
use crate::engine::Engine;
use crate::verify::token_ids;
use std::time::Instant;

pub fn generate_mtp(prompt: Option<&str>, n_gen_arg: Option<usize>) {
    let t0 = Instant::now();
    let m = Engine::new(&crate::model_path());
    // without arguments: the 24 prompt tokens and the 32 generated ones of the M1 dump (comparison with llama.cpp);
    // with a file: the tokenized prompt (Qwen chat already in the file), n generated tokens, comparison of MTP against plain
    let (reference, n_prompt, n_gen, with_reference) = match prompt {
        None => (token_ids(), 24usize, 32usize, true),
        Some(f) => {
            let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
            let ids: Vec<u32> = tk.encode(std::fs::read_to_string(f).expect("prompt"), false).expect("encode").get_ids().to_vec();
            let n = ids.len(); (ids, n, n_gen_arg.unwrap_or(200), false)
        }
    };
    assert!(n_prompt + n_gen < crate::engine::ctx(), "prompt + generated beyond the context");
    println!("loaded in {:.1} s · GPU {}", t0.elapsed().as_secs_f64(), m.gpu.info.name);
    let _ = &reference;
    // --- plain generation (speed reference) ---
    m.reset();
    let mut y = 0;
    for i in 0..n_prompt { y = m.step(reference[i], i); }
    let t = Instant::now();
    let mut simple = vec![y];
    for k in 1..n_gen { y = m.step(y, n_prompt + k - 1); simple.push(y); }
    let ms_s = t.elapsed().as_secs_f64() * 1e3 / (n_gen - 1) as f64;
    // --- generation with MTP ---
    m.reset();
    for i in 0..n_prompt {
        y = m.step(reference[i], i);
        if i + 1 < n_prompt { m.draft(reference[i + 1], i + 1); }       // the MTP cache receives every prompt token
    }
    let t = Instant::now();
    let mut emitted = vec![y];
    let mut d = m.draft(y, n_prompt);                            // draft for position n_prompt + 1
    let (mut pos, mut cur) = (n_prompt, y);
    let (mut steps, mut accepted) = (0, 0);
    // MECHTA_REJECT=k: every k-th step the draft is corrupted on purpose (d + 1): exercises the rejection branch
    // (restore + single draft). The output must stay identical.
    let reject: usize = std::env::var("MECHTA_REJECT").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
    while emitted.len() < n_gen {
        if reject > 0 && (steps + 1) % reject == 0 { d = (d + 1) % crate::engine::VOCAB as u32; }
        let (a0, a1) = m.step2(cur, d, pos);
        steps += 1;
        if a0 == d {
            emitted.push(d); emitted.push(a1); accepted += 1;
            d = m.draft2(d, a1, pos);
            pos += 2; cur = a1;
        } else {
            emitted.push(a0);
            m.restore();
            d = m.draft(a0, pos + 1);
            pos += 1; cur = a0;
        }
    }
    let ms_m = t.elapsed().as_secs_f64() * 1e3 / (emitted.len() - 1) as f64;
    emitted.truncate(n_gen);
    let equal_sem = emitted.iter().zip(&simple).take_while(|(a, b)| a == b).count();
    let llama = |v: &[u32]| if with_reference { format!(" · equal to llama.cpp from the start: {} of {n_gen}", v.iter().zip(&reference[n_prompt..]).take_while(|(a, b)| a == b).count()) } else { String::new() };
    println!("prompt {n_prompt} tokens, generated {n_gen}{}", if m.head_ids.is_empty() { "" } else { " · reduced head for the drafts" });
    println!("simple:   {ms_s:.1} ms/token = {:.1} token/s{}", 1e3 / ms_s, llama(&simple));
    println!("MTP:      {ms_m:.1} ms/token = {:.1} token/s · double steps {steps}, drafts accepted {accepted} ({:.0}%){} · equal to simple: {equal_sem} of {n_gen}",
        1e3 / ms_m, 100.0 * accepted as f64 / steps as f64, llama(&emitted));
    if prompt.is_some() {
        let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
        println!("text: {:?}", tk.decode(&emitted[..emitted.len().min(60)], false).unwrap());
    }
    println!("ratio MTP / simple: x{:.2}{}", ms_s / ms_m, if reject > 0 { format!(" · drafts spoiled on purpose every {reject} steps") } else { String::new() });
}
