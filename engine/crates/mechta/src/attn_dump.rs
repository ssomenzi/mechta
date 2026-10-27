//! `mechta attn-dump <text> <checkpoints> <steps> <out_dir>` (measurement, attention sparsity): prefill of the text up
//! to each checkpoint (comma list, increasing), then <steps> single steps fed with the true next tokens, recording the
//! rotated Q of the 10 attention layers at every step; at the end of each checkpoint's steps, the KV cache of the 10
//! layers. Files per checkpoint P: `q-P.bin` (steps x 10 x 16 x 256 f32), `kv-P.bin` (per layer: K words then V words,
//! as `read_root` returns them: raw f16 pairs with the f16 cache).
use crate::engine::Engine;

pub fn attn_dump(text: &str, checkpoints: &str, steps: usize, out: &str) {
    use std::io::Write;
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(text).expect("text"), false).expect("encode").get_ids().to_vec();
    std::fs::create_dir_all(out).expect("out dir");
    m.reset();
    let mut done = 0;
    for p in checkpoints.split(',').map(|s| s.trim().parse::<usize>().expect("checkpoint")) {
        assert!(p >= done && p + steps < ids.len(), "checkpoint {p}: text of {} tokens", ids.len());
        let t = std::time::Instant::now();
        if p > done { m.prefill(&ids[done..p], done); }
        println!("prefill {done}..{p} in {:.0} s", t.elapsed().as_secs_f64());
        let mut q = std::io::BufWriter::new(std::fs::File::create(format!("{out}/q-{p}.bin")).expect("q"));
        for j in 0..steps {
            for layer in m.step_q(ids[p + j], p + j) { for x in layer { q.write_all(&x.to_le_bytes()).unwrap(); } }
        }
        drop(q);
        done = p + steps;
        let r = m.read_root(done);
        let mut kv = std::io::BufWriter::new(std::fs::File::create(format!("{out}/kv-{p}.bin")).expect("kv"));
        for (k, v) in &r.kv { for x in k.iter().chain(v.iter()) { kv.write_all(&x.to_le_bytes()).unwrap(); } }
        println!("checkpoint {p}: {steps} steps, Q and KV of {done} positions ({} layers, kv={}) -> {out}", r.kv.len(), crate::engine::kv_format().name());
    }
}

/// The masked decode attention for `sparse-check`: attn_fd of ops.wgsl with one change, a score is kept only if the
/// mask allows its block for that head (`msk`, binding 12: 16 heads x CTX/128 blocks, 0 = drop). Same code otherwise.
fn masked_source() -> String {
    let ops = crate::engine::ops_source(crate::engine::kv_format());
    let a = ops.find("@compute @workgroup_size(256)\nfn attn_fd(").expect("attn_fd entry");
    let b = ops[a..].find("// attn_fd2:").expect("end of attn_fd") + a;
    let keep = "select(-3.0e38, (d8[h] + fs[h * FD_B + pos]) * scale, valid)";
    let k = ops[a..b].matches(keep).count();
    assert_eq!(k, 1, "the score line of attn_fd changed");
    let body = ops[a..b].replace("fn attn_fd(", "fn attn_fd_mask(").replace("attn_fd_one(", "attn_fdm_one(")
        .replace(keep, "select(-3.0e38, (d8[h] + fs[h * FD_B + pos]) * scale, valid && msk[(8u * kvh + h) * nb + block] != 0u)");
    format!("{ops}\n@group(0) @binding(12) var<storage, read> msk: array<u32>;\n{body}")
}

/// `mechta sparse-check <text> <checkpoints> <steps> <out.tsv>` (measurement, sparse attention quality): prefill to each
/// checkpoint, then <steps> single steps with the true tokens. At every step, from the same state (photo of the
/// recurrent states): the variants with a block-sparse decode attention, then the exact step (which moves on). The
/// blocks (128 positions) are chosen per query head as Quest does: upper bound sum_d max(q_d min_d, q_d max_d) from the
/// per-block min / max of K, top B, the newest block always kept. Writes KL(exact || variant) and top-1 agreement.
/// Variant `ones` keeps every block through the masked kernel: it must give the exact logits (plumbing check).
pub fn sparse_check(text: &str, checkpoints: &str, steps: usize, out: &str) {
    use std::io::Write;
    const BB: usize = 128;
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(text).expect("text"), false).expect("encode").get_ids().to_vec();
    let pipe = m.gpu.pipeline(&masked_source(), "attn_fd_mask");
    let nbk = crate::engine::ctx() / BB;
    let mask = m.gpu.empty_buffer("mask", (16 * nbk * 4) as u64);
    // (name, sparse attention layers by index 0..10, blocks kept per head; 0 = every block)
    let dense_ends = [false, false, true, true, true, true, true, true, true, false];   // layers 3, 7, 39 stay dense
    let variants: Vec<(&str, [bool; 10], usize)> = vec![("ones", [true; 10], 0), ("mid32", dense_ends, 32), ("mid64", dense_ends, 64),
                                                          ("mid128", dense_ends, 128), ("all128", [true; 10], 128)];
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).expect("out"));
    writeln!(f, "checkpoint\tstep\tpos\tvariant\tkl\ttop1_same").unwrap();
    m.reset();
    let mut done = 0;
    for p in checkpoints.split(',').map(|s| s.trim().parse::<usize>().expect("checkpoint")) {
        assert!(p >= done && p + steps < ids.len());
        let t = std::time::Instant::now();
        if p > done { m.prefill(&ids[done..p], done); }
        println!("prefill {done}..{p} in {:.0} s", t.elapsed().as_secs_f64());
        // per layer, KV head, block, dimension: min and max of K over the positions written so far
        let r = m.read_root(p);
        let mut mn = vec![vec![f32::MAX; 2 * nbk * 256]; 10]; let mut mx = vec![vec![f32::MIN; 2 * nbk * 256]; 10];
        let add = |li: usize, pos: usize, row: &[f32], mn: &mut Vec<Vec<f32>>, mx: &mut Vec<Vec<f32>>| {
            for kv in 0..2 { for d in 0..256 { let i = (kv * nbk + pos / BB) * 256 + d; let x = row[kv * 256 + d];
                if x < mn[li][i] { mn[li][i] = x; } if x > mx[li][i] { mx[li][i] = x; } } }
        };
        for (li, (k, _)) in r.kv.iter().enumerate() {
            let row: Vec<f32> = k.iter().flat_map(|w| { let b = w.to_bits(); [half::f16::from_bits(b as u16).to_f32(), half::f16::from_bits((b >> 16) as u16).to_f32()] }).collect();
            for pos in 0..p { add(li, pos, &row[pos * 512..(pos + 1) * 512], &mut mn, &mut mx); }
        }
        drop(r);
        for j in 0..steps {
            let pos = p + j; let tok = ids[pos];
            let nbv = (pos + 1).div_ceil(BB);
            m.photograph();
            let mut res = vec![];
            for (_, layers, budget) in &variants {
                let (mn, mx) = (&mn, &mx);
                let mut sel = |li: usize, q: &[f32]| -> Option<Vec<u32>> {
                    if !layers[li] { return None; }
                    let mut msk = vec![0u32; 16 * nbk];
                    for h in 0..16 {
                        let kv = h / 8; let qh = &q[h * 256..(h + 1) * 256];
                        if *budget == 0 || nbv <= *budget { for b in 0..nbv { msk[h * nbk + b] = 1; } continue; }
                        let mut ub: Vec<(f32, usize)> = (0..nbv - 1).map(|b| { let i0 = (kv * nbk + b) * 256;
                            ((0..256).map(|d| (qh[d] * mn[li][i0 + d]).max(qh[d] * mx[li][i0 + d])).sum::<f32>(), b) }).collect();
                        ub.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                        for &(_, b) in ub.iter().take(budget - 1) { msk[h * nbk + b] = 1; }
                        msk[h * nbk + nbv - 1] = 1;
                    }
                    Some(msk)
                };
                res.push(m.step_masked(tok, pos, &pipe, &mask, &mut sel));
                m.restore_photo();
            }
            let exact = m.step_masked(tok, pos, &pipe, &mask, &mut |_, _| None);
            for li in 0..10 { let row = m.read_k_row(li, pos); add(li, pos, &row, &mut mn, &mut mx); }
            let soft = |l: &[f32]| -> Vec<f64> { let mx = l.iter().cloned().fold(f32::MIN, f32::max) as f64;
                let e: Vec<f64> = l.iter().map(|x| (*x as f64 - mx).exp()).collect(); let z: f64 = e.iter().sum(); e.iter().map(|x| x / z).collect() };
            let argmax = |l: &[f32]| l.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
            let pe = soft(&exact);
            for ((name, _, _), lv) in variants.iter().zip(&res) {
                let pv = soft(lv);
                let kl: f64 = pe.iter().zip(&pv).filter(|(a, _)| **a > 0.0).map(|(a, b)| a * (a.ln() - b.max(1e-300).ln())).sum();
                writeln!(f, "{p}\t{j}\t{pos}\t{name}\t{kl:.3e}\t{}", (argmax(&exact) == argmax(lv)) as u8).unwrap();
            }
            f.flush().unwrap();
        }
        done = p + steps;
        println!("checkpoint {p}: {steps} steps x {} variants -> {out}", variants.len());
    }
}

/// Per-block min / max of K for the Quest-style choice (blocks of `BB` positions): [layer][(kv * nbk + block) * 256 + d].
struct Summary { mn: Vec<Vec<f32>>, mx: Vec<Vec<f32>>, nbk: usize }
const BB: usize = 128;
impl Summary {
    fn new(nbk: usize) -> Self { Summary { mn: vec![vec![f32::MAX; 2 * nbk * 256]; 10], mx: vec![vec![f32::MIN; 2 * nbk * 256]; 10], nbk } }
    fn add(&mut self, li: usize, pos: usize, row: &[f32]) {
        for kv in 0..2 { for d in 0..256 { let i = (kv * self.nbk + pos / BB) * 256 + d; let x = row[kv * 256 + d];
            if x < self.mn[li][i] { self.mn[li][i] = x; } if x > self.mx[li][i] { self.mx[li][i] = x; } } }
    }
    fn from_root(m: &Engine, p: usize, nbk: usize) -> Self {
        let mut s = Summary::new(nbk);
        let r = m.read_root(p);
        for (li, (k, _)) in r.kv.iter().enumerate() {
            let row: Vec<f32> = k.iter().flat_map(|w| { let b = w.to_bits(); [half::f16::from_bits(b as u16).to_f32(), half::f16::from_bits((b >> 16) as u16).to_f32()] }).collect();
            for pos in 0..p { s.add(li, pos, &row[pos * 512..(pos + 1) * 512]); }
        }
        s
    }
    /// The mask of one attention layer: per query head the `budget` blocks with the highest upper bound
    /// sum_d max(q_d min_d, q_d max_d), the newest block always kept; every block if they are not more than `budget`.
    fn mask(&self, li: usize, q: &[f32], visible: usize, budget: usize) -> Vec<u32> {
        let nbv = visible.div_ceil(BB);
        let mut msk = vec![0u32; 16 * self.nbk];
        for h in 0..16 {
            let kv = h / 8; let qh = &q[h * 256..(h + 1) * 256];
            if budget == 0 || nbv <= budget { for b in 0..nbv { msk[h * self.nbk + b] = 1; } continue; }
            let mut ub: Vec<(f32, usize)> = (0..nbv - 1).map(|b| { let i0 = (kv * self.nbk + b) * 256;
                ((0..256).map(|d| (qh[d] * self.mn[li][i0 + d]).max(qh[d] * self.mx[li][i0 + d])).sum::<f32>(), b) }).collect();
            ub.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            for &(_, b) in ub.iter().take(budget - 1) { msk[h * self.nbk + b] = 1; }
            msk[h * self.nbk + nbv - 1] = 1;
        }
        msk
    }
}

/// `mechta sparse-recall <text> <tokens> <out.tsv>` (measurement, sparse attention on retrieval): ~<tokens> of the text
/// with 8 "needles" ("the access code of module NAME is NUMBER") at depths 6%..94%, prefilled once (dense, as the
/// server); then 12 questions (8 on one code, 4 on two or three codes), each answered greedily (thinking off, at most
/// 40 tokens) from the same state with the exact attention and with block-sparse decode attention (as sparse-check).
/// Writes per question and variant: the answer and whether it contains every expected code.
pub fn sparse_recall(text: &str, tokens: usize, out: &str) {
    use std::io::Write;
    const IM_END: u32 = 248046; const ENDOFTEXT: u32 = 248044;
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let enc = |s: &str| -> Vec<u32> { tk.encode(s, false).expect("encode").get_ids().to_vec() };
    let names = ["ORCHID", "FALCON", "GRANITE", "LANTERN", "MERIDIAN", "PEBBLE", "SAFFRON", "TUNDRA"];
    let codes = ["418273", "905126", "362891", "777014", "150638", "829457", "246905", "613382"];
    // the text cut to ~tokens (chars per token measured on the whole text), needles at fixed depths, on their own line
    let full = std::fs::read_to_string(text).expect("text");
    let cpt = full.len() as f64 / enc(&full).len() as f64;
    let mut body: String = full.chars().take((tokens as f64 * cpt) as usize).collect();
    for (i, (n, c)) in names.iter().zip(codes).enumerate().rev() {
        let at = body.len() * (1 + 2 * i) / 16;
        let at = body[at..].find('\n').map(|k| at + k + 1).unwrap_or(at);
        body.insert_str(at, &format!("// NOTE: the access code of module {n} is {c}.\n"));
    }
    let prefix = enc(&format!("<|im_start|>user\nHere is a long source file.\n\n{body}\n"));
    let mut qs: Vec<(String, Vec<&str>)> = names.iter().zip(codes).map(|(n, c)| (format!("What is the access code of module {n}? Answer with the number only."), vec![c])).collect();
    for (a, b, c) in [(0, 3, None), (1, 6, None), (2, 5, Some(7)), (4, 0, Some(6))] {
        let mut ns = vec![names[a], names[b]]; let mut cs = vec![codes[a], codes[b]];
        if let Some(c) = c { ns.push(names[c]); cs.push(codes[c]); }
        qs.push((format!("What are the access codes of modules {}? Answer as NAME=CODE, one per line.", ns.join(" and ")), cs));
    }
    let pipe = m.gpu.pipeline(&masked_source(), "attn_fd_mask");
    let nbk = crate::engine::ctx() / BB;
    let mask = m.gpu.empty_buffer("mask", (16 * nbk * 4) as u64);
    let dense_ends = [false, false, true, true, true, true, true, true, true, false];
    let variants: Vec<(&str, [bool; 10], usize)> = vec![("dense", [false; 10], 0), ("mid128", dense_ends, 128), ("all128", [true; 10], 128), ("mid64", dense_ends, 64)];
    let t = std::time::Instant::now();
    m.reset(); m.prefill(&prefix, 0);
    println!("prefill of {} tokens in {:.0} s", prefix.len(), t.elapsed().as_secs_f64());
    let base = Summary::from_root(&m, prefix.len(), nbk);
    m.photograph();
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).expect("out"));
    writeln!(f, "question\tvariant\tcorrect\texpected\tanswer").unwrap();
    for (qi, (q, expect)) in qs.iter().enumerate() {
        let suffix = enc(&format!("\nQuestion: {q}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        for (name, layers, budget) in &variants {
            m.restore_photo();
            let mut s = Summary { mn: base.mn.clone(), mx: base.mx.clone(), nbk };
            let mut pos = prefix.len();
            m.prefill(&suffix[..suffix.len() - 1], pos);
            for p in pos..pos + suffix.len() - 1 { for li in 0..10 { let row = m.read_k_row(li, p); s.add(li, p, &row); } }
            pos += suffix.len() - 1;
            let mut tok = suffix[suffix.len() - 1];
            let mut answer = vec![];
            for _ in 0..40 {
                let logits = {
                    let s = &s;
                    let mut sel = |li: usize, qv: &[f32]| -> Option<Vec<u32>> { if !layers[li] { None } else { Some(s.mask(li, qv, pos + 1, *budget)) } };
                    m.step_masked(tok, pos, &pipe, &mask, &mut sel)
                };
                for li in 0..10 { let row = m.read_k_row(li, pos); s.add(li, pos, &row); }
                pos += 1;
                tok = logits.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32;
                if tok == IM_END || tok == ENDOFTEXT { break; }
                answer.push(tok);
            }
            let text = tk.decode(&answer, false).unwrap_or_default();
            let ok = expect.iter().all(|c| text.contains(c));
            writeln!(f, "{qi}\t{name}\t{}\t{}\t{:?}", ok as u8, expect.join(","), text).unwrap();
            f.flush().unwrap();
            println!("q{qi} {name}: {} {:?}", if ok { "OK " } else { "NO " }, text);
        }
    }
}
