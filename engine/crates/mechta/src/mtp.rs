//! M4: check of the MTP layer against the llama.cpp dump (`reference-mtp/`, `mechta-dump-mtp`).
//! For each position i+1 (1..55): h = h_nextn(i) from the dump, true token t_{i+1}; our draft goes through
//! eh_proj → attention (its own MTP cache, built position by position as llama.cpp does) → MoE → norm → logits.
//! Relative errors (maximum / maximum of the reference) of each segment, and argmax against the argmax of the dump.
use crate::engine::{Engine, VOCAB};
use crate::verify::{err, token_ids, Dump};

/// reference dump of llama.cpp, in the workspace (path fixed at build time: a verification tool, not the server)
const DUMP_MTP: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../reference-mtp/tensors.bin");

pub fn mtp() {
    let t0 = std::time::Instant::now();
    let m = Engine::new(&crate::model_path());
    println!("loaded in {:.1} s · weights on the GPU {:.2} GB · MTP segments {}", t0.elapsed().as_secs_f64(), m.byte_gpu as f64 / 1e9, m.end_mtp.len());
    let mut d = Dump::load_file(DUMP_MTP);
    let toks = token_ids();
    let names = ["mtp:mtp_eh_proj-40", "mtp:mtp_attn_residual-40", "mtp:mtp_post_ffn-40", "mtp:h_nextn", "mtp:result_output"];
    let buf = ["x", "x2", "x", "xn", "logits"];
    let mut worst = [0f64; 5]; let mut equal = 0; let mut n = 0;
    println!("{:>3} {:>9} {:>9} {:>9} {:>9} {:>9} | {:>7} {:>7} {}", "pos", "eh_proj", "attn", "moe", "norm", "logit", "mine", "dump", "same");
    for pos in 1..toks.len() {
        let h = d.get("h_nextn", pos as i32 - 1, 0);
        let mut e = [0f64; 5];
        let mut f = |k: usize| {
            let nv = if k == 4 { VOCAB } else { 2048 };
            let mine = m.read(buf[k], nv);
            let reference = d.get(names[k], pos as i32, 0);
            e[k] = err(&mine, &reference);
        };
        let mine = m.draft_verify(&h, toks[pos], pos, &mut f);
        let reference = d.get("mtp:result_output", pos as i32, 0);
        let theirs = reference.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) }).0 as u32;
        n += 1; if mine == theirs { equal += 1; }
        for k in 0..5 { worst[k] = worst[k].max(e[k]); }
        println!("{pos:>3} {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} {:>9.2e} | {mine:>7} {theirs:>7} {}", e[0], e[1], e[2], e[3], e[4], if mine == theirs { "si" } else { "NO" });
    }
    println!("\nmax error: eh_proj {:.2e} · attention {:.2e} · moe {:.2e} · norm {:.2e} · logit {:.2e}", worst[0], worst[1], worst[2], worst[3], worst[4]);
    println!("draft argmax equal to llama.cpp: {equal} of {n} · {:.1} s", t0.elapsed().as_secs_f64());
}

/// M4: the double draft (`draft2`) against the dump. Pairs (i, i+1) for odd i: h(i-1), h(i) from the dump, true tokens
/// t_i, t_{i+1}; the draft for i+2 against the argmax of `mtp:result_output` at position i+1. The MTP cache is filled
/// in pairs, in sequence. With MECHTA_MTP2=full the version without the MoE cut of the first token is tested.
pub fn mtp2() {
    let t0 = std::time::Instant::now();
    let m = Engine::new(&crate::model_path());
    let mut d = Dump::load_file(DUMP_MTP);
    let toks = token_ids();
    let (mut n, mut equal) = (0, 0);
    let mut i = 1;
    while i + 2 < toks.len() {
        let h0 = d.get("h_nextn", i as i32 - 1, 0); let h1 = d.get("h_nextn", i as i32, 0);
        let mine = m.draft2_verify(&h0, &h1, toks[i], toks[i + 1], i - 1);
        let reference = d.get("mtp:result_output", i as i32 + 1, 0);
        let theirs = reference.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |(bi, bv), (j, &v)| if v > bv { (j, v) } else { (bi, bv) }).0 as u32;
        n += 1; if mine == theirs { equal += 1; } else { println!("  pos {}: my draft {mine}, llama.cpp {theirs}", i + 2); }
        i += 2;
    }
    println!("double draft: argmax equal to llama.cpp in {equal} of {n} pairs ({}) · {:.1} s",
        if std::env::var("MECHTA_MTP2").map(|v| v == "full").unwrap_or(false) { "full MoE" } else { "MoE second token only" }, t0.elapsed().as_secs_f64());
}
