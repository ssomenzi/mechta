//! M4: check of the double step (2 tokens in one pass over the weights) against the M1 dump, position by position.
//! `mechta step2 <pos>...`: for each pos, the states are brought up to pos-1 with single steps, then one double step
//! on the true tokens (pos, pos+1): l_out of every layer for the two tokens against `l_out-il` of the dump at the two positions,
//! and the argmaxes against the true tokens of pos+1 and pos+2.
use crate::engine::{Engine, N_LAYER};
use crate::verify::{err, token_ids, Dump};

pub fn step2(positions: &[usize]) {
    let t0 = std::time::Instant::now();
    let m = Engine::new(&crate::model_path());
    println!("loaded in {:.1} s · layers in the double step {}", t0.elapsed().as_secs_f64(), m.end2.len());
    let mut d = Dump::load();
    let reference = token_ids();
    for &p in positions {
        assert!(p + 2 < reference.len(), "pos {p}: the tokens of pos+1 and pos+2 are needed");
        m.reset();
        for pos in 0..p { m.step(reference[pos], pos); }
        let (mut worst0, mut worst1, mut first) = (0f64, 0f64, None);
        let mut rows = vec![];
        let mut f = |il: usize| {
            let x = m.read("x", 4096);
            let e0 = err(&x[..2048], &d.get(&format!("l_out-{il}"), p as i32, 0));
            let e1 = err(&x[2048..], &d.get(&format!("l_out-{il}"), p as i32 + 1, 0));
            worst0 = worst0.max(e0); worst1 = worst1.max(e1);
            if first.is_none() && (e0 > 0.3 || e1 > 0.3) { first = Some(il); }
            if il < 4 || il % 10 == 9 || il == N_LAYER - 1 { rows.push(format!("    layer {il:>2}: tok0 {e0:.2e} tok1 {e1:.2e}")); }
        };
        let t = std::time::Instant::now();
        let (a0, a1) = m.step2_verify(reference[p], reference[p + 1], p, &mut f);
        println!("pos {p},{}: argmax {a0} / {a1} vs {} / {} ({}, {}) · max error tok0 {worst0:.2e} tok1 {worst1:.2e}{} · {:.0} ms",
            p + 1, reference[p + 1], reference[p + 2], if a0 == reference[p + 1] { "si" } else { "NO" }, if a1 == reference[p + 2] { "si" } else { "NO" },
            first.map(|l| format!(" · first layer above 0.3: {l}")).unwrap_or_default(), t.elapsed().as_secs_f64() * 1e3);
        for r in rows { println!("{r}"); }
    }
    println!("{:.1} s", t0.elapsed().as_secs_f64());
}
