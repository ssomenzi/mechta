//! Phase test: in which phase does a GPU run corrupt the page cache?
//! Loads, then waits; computes 56 tokens, then waits; frees the GPU and exits. Between one phase and the next
//! an external script compares the cached GGUF with the disk. Signals via files in the data folder:
//! writes `loaded`, waits for `go1`; writes `computed`, waits for `go2`; writes `released` and exits.
use crate::engine::Engine;
use crate::verify::token_ids;
use std::path::Path;

fn wait(dir: &Path, name: &str) {
    while !dir.join(name).exists() { std::thread::sleep(std::time::Duration::from_millis(100)); }
}

pub fn phases(dir: &str) {
    let dir = Path::new(dir);
    let m = Engine::new(&crate::model_path());
    m.reset();
    std::fs::write(dir.join("loaded"), "").unwrap();
    wait(dir, "go1");
    let reference = token_ids();
    let mut chosen = vec![];
    for (pos, t) in reference.iter().enumerate() {
        let l = m.token(*t, pos, None);
        chosen.push(l.iter().enumerate().fold((0, f32::MIN), |b, (i, v)| if *v > b.1 { (i, *v) } else { b }).0);
    }
    let equal = chosen[23..55].iter().zip(&reference[24..56]).filter(|(a, b)| **a as u32 == **b).count();
    std::fs::write(dir.join("computed"), format!("tokens equal to llama.cpp: {equal} of 32\n")).unwrap();
    wait(dir, "go2");
    drop(m);
    std::fs::write(dir.join("released"), "").unwrap();
}

/// `session-phases <dir> <text> <chars> <max_tokens>`: the same phases, with the work of a real server request:
/// load a Session (as `serve`), then one chat request: system + user with the first <chars>
/// characters of <text> and a request for a long HTML file; prefill (MECHTA_PREFILL_ATTN), then up to <max_tokens>
/// sampled tokens with MTP (temperature 0.6, top_p 0.95, top_k 20, seed 1). Writes the answer in `<dir>/answer.txt`.
/// Use MECHTA_ROOT=no so the saved roots are not touched.
pub fn session_phases(dir: &str, text: &str, chars: usize, max_token: usize) {
    use mechta_api::{Engine as _, Params};
    let dir = Path::new(dir);
    let mut s = crate::session::Session::new();
    std::fs::write(dir.join("loaded"), "").unwrap();
    wait(dir, "go1");
    let code: String = std::fs::read_to_string(text).expect("text").chars().take(chars).collect();
    let prompt = format!("<|im_start|>system\nYou are a coding assistant.<|im_end|>\n<|im_start|>user\nHere is some code.\n\n{code}\n\n\
Write, from scratch, a complete single-file HTML spreadsheet with a formula engine, keyboard navigation, CSV import and export, \
undo and redo, and self-tests. Write the whole file, with all the CSS and JavaScript.<|im_end|>\n<|im_start|>assistant\n<think>\n");
    let p = Params { max_token, temperature: 0.6, top_p: 0.95, top_k: 20, seed: 1, mtp: true, reasoning_budget: 0 };
    let mut out = String::new();
    let t = std::time::Instant::now();
    let outcome = s.generate(&prompt, &p, &mut |x: &str| { out.push_str(x); true });
    let cjk = out.chars().filter(|c| ('\u{4e00}'..='\u{9fff}').contains(c)).count();
    std::fs::write(dir.join("answer.txt"), &out).unwrap();
    std::fs::write(dir.join("computed"), format!("{} chars in {:.0} s, Chinese chars {cjk}, outcome {:?}\n",
        out.len(), t.elapsed().as_secs_f64(), outcome.map(|e| e.end))).unwrap();
    wait(dir, "go2");
    drop(s);
    std::fs::write(dir.join("released"), "").unwrap();
}
