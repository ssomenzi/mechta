//! M2, first measurement: where the time of one token goes. No change to the computation.
//! Positions 0..27 without measurements (true time per token), then 28..55 with the GPU timestamps for every
//! dispatch (one computation step per dispatch: it adds a bit of cost, visible in the total).
use crate::engine::{Engine, Profile};
use crate::verify::token_ids;
use std::time::Instant;

/// GPU memory in use (amdgpu, sysfs): (VRAM, GTT) in GB.
pub fn gpu_memory() -> (f64, f64) {
    let read = |f: &str| -> f64 {
        for c in 0..4 { if let Ok(s) = std::fs::read_to_string(format!("/sys/class/drm/card{c}/device/{f}")) { return s.trim().parse::<f64>().unwrap_or(0.0) / 1e9; } }
        0.0
    };
    (read("mem_info_vram_used"), read("mem_info_gtt_used"))
}

/// M4: time of the double step (2 tokens in one pass over the weights) against the single step, same tokens of the dump.
/// Single steps at positions 0..27 (warm-up + measurement), then 14 double steps on the pairs (28,29) .. (54,55).
pub fn profile2() {
    let m = Engine::new(&crate::model_path());
    let reference = token_ids();
    m.reset();
    *m.span.borrow_mut() = Some(vec![]);
    let mut singoli = vec![];
    for pos in 0..28 { let t = Instant::now(); std::hint::black_box(m.step(reference[pos], pos)); singoli.push(t.elapsed().as_secs_f64() * 1e3); }
    let s1 = m.span.borrow_mut().take().unwrap();
    *m.span.borrow_mut() = Some(vec![]);
    let mut doubles = vec![]; let mut correct = 0;
    let mut pos = 28;
    while pos + 1 < reference.len() {
        let t = Instant::now();
        let (a0, a1) = m.step2(reference[pos], reference[pos + 1], pos);
        doubles.push(t.elapsed().as_secs_f64() * 1e3);
        if a0 == reference[pos + 1] { correct += 1; }
        if pos + 2 < reference.len() && a1 == reference[pos + 2] { correct += 1; }
        pos += 2;
    }
    let s2 = m.span.borrow_mut().take().unwrap();
    let mid = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let mut g1: Vec<f64> = s1[4..].iter().map(|x| x.2).collect(); let mut g2: Vec<f64> = s2.iter().map(|x| x.2).collect();
    let (ts, td) = (mid(&mut singoli[4..].to_vec()), mid(&mut doubles.clone()));
    println!("GPU {} · weights on the GPU {:.2} GB", m.gpu.info.name, m.byte_gpu as f64 / 1e9);
    println!("single step (pos 4..27): median {ts:.1} ms · GPU first→last {:.1} ms", mid(&mut g1));
    println!("double step ({} pairs, pos 28..55): median {td:.1} ms · GPU first→last {:.1} ms · per token {:.1} ms", doubles.len(), mid(&mut g2), td / 2.0);
    println!("double / single ratio: {:.2} · correct argmax in double steps: {correct} of {}", td / ts, 2 * doubles.len() - 1);
    // where the extra time goes: kernel by kernel, double steps measured (pos 28..55 again, states rebuilt)
    m.reset();
    for pos in 0..28 { m.step(reference[pos], pos); }
    let mut agg: std::collections::HashMap<String, (f64, u64, u32)> = std::collections::HashMap::new();
    let mut n = 0; let mut pos = 28;
    while pos + 1 < reference.len() {
        for (name, ms, byte) in m.step2_measured(reference[pos], reference[pos + 1], pos) {
            let e = agg.entry(name).or_default(); e.0 += ms; e.1 += byte; e.2 += 1;
        }
        n += 1; pos += 2;
    }
    let tot: f64 = agg.values().map(|v| v.0).sum::<f64>() / n as f64;
    println!("\nDOUBLE STEP KERNELS (mean per double step over {n} steps, total {tot:.1} ms)");
    println!("{:34} {:>9} {:>6} {:>8} {:>10} {:>8}", "operation", "ms/step", "%", "dispatch", "MB read", "GB/s");
    let mut v: Vec<_> = agg.iter().collect();
    v.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    for (k, (ms, byte, c)) in v.iter().take(22) {
        let (ms, mb) = (ms / n as f64, *byte as f64 / n as f64 / 1e6);
        println!("{k:34} {ms:>9.2} {:>5.1}% {:>8} {mb:>10.1} {:>8}", 100.0 * ms / tot, c / n, if mb > 0.0 { format!("{:.1}", mb / ms) } else { "-".into() });
    }
}

/// `profile [prompt.txt]`: without arguments the 56 tokens of the M1 dump; with a file, the prompt is processed first
/// (single steps) and the measurements are taken AFTER, at that context, on greedily generated tokens (M4: the cost of the context).
/// M3: per-kernel profile of the block prefill on a prompt: loads, runs the prefill with per-dispatch measurements, prints
/// the table (average per block and per token).
pub fn profile_prefill(prompt: &str) {
    let t0 = Instant::now();
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(prompt).expect("prompt"), false).expect("encode").get_ids().to_vec();
    let nb = crate::engine::NB; let blocks = ids.len().div_ceil(nb);
    println!("loaded in {:.1} s · prompt of {} tokens = {blocks} blocks of {nb}", t0.elapsed().as_secs_f64(), ids.len());
    m.reset();
    let t = Instant::now(); m.prefill(&ids, 0); let s_simple = t.elapsed().as_secs_f64();
    m.reset();
    *m.prof.borrow_mut() = Some(Profile::default());
    let t = Instant::now(); m.prefill(&ids, 0); let s_meas = t.elapsed().as_secs_f64();
    let p = m.prof.borrow_mut().take().unwrap();
    let n = blocks as f64; let per_tok = ids.len() as f64;
    let gpu_tot: f64 = p.gpu.values().map(|v| v.0).sum::<f64>();
    let cpu_tot: f64 = p.cpu.values().sum::<f64>();
    println!("prefill without measurements: {s_simple:.1} s = {:.1} ms/token · with measurements {s_meas:.1} s · GPU kernel total {:.1} ms/token · CPU parts {:.1} ms/token",
        s_simple * 1e3 / per_tok, gpu_tot / per_tok, cpu_tot / per_tok);
    println!("\nPREFILL GPU KERNELS (mean per block, sorted by time)");
    println!("{:34} {:>10} {:>9} {:>6} {:>8} {:>10} {:>8}", "operation", "ms/block", "ms/token", "%", "dispatch", "MB read", "GB/s");
    let mut v: Vec<_> = p.gpu.iter().collect();
    v.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    for (k, (ms, byte, c)) in v {
        let (msb, mb) = (ms / n, *byte as f64 / n / 1e6);
        let gbs = if mb > 0.0 { format!("{:.1}", mb / msb) } else { "-".into() };
        println!("{k:34} {msb:>10.2} {:>9.3} {:>5.1}% {:>8.0} {mb:>10.1} {gbs:>8}", ms / per_tok, 100.0 * ms / gpu_tot, *c as f64 / n);
    }
}

pub fn profile(prompt: Option<&str>) {
    let (v0, g0) = gpu_memory();
    let t0 = Instant::now();
    let m = Engine::new(&crate::model_path());
    let (v1, g1) = gpu_memory();
    println!("loaded in {:.1} s · VRAM {v0:.2} -> {v1:.2} GB · GTT {g0:.2} -> {g1:.2} GB", t0.elapsed().as_secs_f64());
    // token sequence: the M1 dump, or the prompt + 84 greedily generated tokens (three phases of 28)
    let (reference, base) = match prompt {
        None => (token_ids(), 0usize),
        Some(f) => {
            let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
            let ids: Vec<u32> = tk.encode(std::fs::read_to_string(f).expect("prompt"), false).expect("encode").get_ids().to_vec();
            let n = ids.len(); (ids, n)
        }
    };
    // MECHTA_FAKE_POS=N: no prefill, KV cache and states at zero, measurements at context N (meaningless output, valid times)
    let fake: Option<usize> = std::env::var("MECHTA_FAKE_POS").ok().and_then(|v| v.parse().ok());
    let (mut reference, base) = match fake { Some(n) => (vec![0u32; n + 85], n), None => (reference, base) };
    assert!(base + 84 < crate::engine::ctx(), "prompt too long for the context");
    m.reset();
    if fake.is_some() { println!("fake position: measurements at context {base}..{} without prefill", base + 84); }
    else if base > 0 {
        let t = Instant::now();
        let blocks = Engine::prefill_in_blocks();
        // M3: in blocks (MECHTA_PREFILL=blocks) the first base - 1 tokens, then the last with `step` to get the first logit;
        // otherwise one token at a time. (Earlier versions sent the last prompt token went through TWICE: DeltaNet state
        // advanced twice with the same token; the generated tokens of those measurements are not comparable with these.)
        let y = if blocks { m.prefill(&reference[..base - 1], 0); m.step(reference[base - 1], base - 1) }
                else { let mut y = 0; for pos in 0..base { y = m.step(reference[pos], pos); } y };
        let ms_tok = t.elapsed().as_secs_f64() * 1e3 / base as f64;
        reference.push(y); for _ in 1..84 { reference.push(0); }   // placeholder: the generated tokens are discovered step by step
        println!("prompt of {base} tokens processed in {:.1} s ({ms_tok:.1} ms/token, {}) · measurements are at context {base}..{}",
            t.elapsed().as_secs_f64(), if blocks { format!("in blocks of {}", crate::engine::NB) } else { "one token at a time".into() }, base + 84);
    }
    let mut simple = vec![];
    for pos in 0..28 { let t = Instant::now(); let y = m.step(reference[base + pos], base + pos); if base > 0 { reference[base + pos + 1] = y; } simple.push(t.elapsed().as_secs_f64() * 1e3); }
    if base > 0 { println!("tokens generated after the prompt (28): {:?}", &reference[base..base + 29]); }
    // tokens with only the two timestamps (start and end of the step): the true GPU time of the token, gaps included
    // MECHTA_STEP=old: token + read of the logits (1 MB, second submit) and argmax on the CPU, the old path.
    // Default: `step` (argmax on the GPU, a single submit with the 4-byte read).
    let old = std::env::var("MECHTA_STEP").map(|v| v == "old").unwrap_or(false);
    *m.span.borrow_mut() = Some(vec![]);
    let mut with_span = vec![];
    for pos in base + 28..base + 56 {
        let t = Instant::now();
        if old { let l = m.token(reference[pos], pos, None); std::hint::black_box(l.iter().cloned().fold((0usize, f32::NEG_INFINITY), |(bi, bv), v| if v > bv { (0, v) } else { (bi, bv) })); }
        else { let y = m.step(reference[pos], pos); if base > 0 { reference[pos + 1] = y; } }
        with_span.push(t.elapsed().as_secs_f64() * 1e3);
    }
    let spans = m.span.borrow_mut().take().unwrap();
    m.reset();
    // start over for the tokens measured per dispatch (prompt as above: in blocks or one by one)
    if fake.is_none() && base > 0 && Engine::prefill_in_blocks() { m.prefill(&reference[..base - 1], 0); for pos in base - 1..base + 28 { m.step(reference[pos], pos); } }
    else { for pos in (if fake.is_some() { base } else { 0 })..base + 28 { m.step(reference[pos], pos); } }
    *m.prof.borrow_mut() = Some(Profile::default());
    let mut measured = vec![];
    for pos in base + 28..base + 56 { let t = Instant::now(); m.token(reference[pos], pos, None); measured.push(t.elapsed().as_secs_f64() * 1e3); }
    let p = m.prof.borrow_mut().take().unwrap();
    let n = measured.len() as f64;
    let mid = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let (ts, tm) = (mid(&mut simple[4..].to_vec()), mid(&mut measured.clone()));
    let gpu_tot: f64 = p.gpu.values().map(|v| v.0).sum::<f64>() / n;
    let cpu_tot: f64 = p.cpu.values().sum::<f64>() / n;
    println!("GPU {} · weights on the GPU {:.2} GB · kernel {}", m.gpu.info.name, m.byte_gpu as f64 / 1e9, if crate::engine::kernel_m1() { "M1" } else { "M2" });
    println!("tokens without measurements (pos 4..27): median {ts:.1} ms = {:.2} token/s", 1e3 / ts);
    let mut sp: Vec<f64> = spans.iter().map(|x| x.2).collect(); let mut sa: Vec<f64> = spans.iter().map(|x| x.1).collect(); let mut spp: Vec<f64> = spans.iter().map(|x| x.0).collect();
    let (tsp, msp, msa, mpp) = (mid(&mut with_span.clone()), mid(&mut sp), mid(&mut sa), mid(&mut spp));
    println!("tokens with 2 timestamps (pos 28..55, {}{}): median {tsp:.1} ms", if old { "token + 1 MB logits + CPU argmax" } else { "step: GPU argmax, 4 bytes" }, if m.gpu.poll_active { ", active poll" } else { ", sleeping poll" });
    println!("  GPU from first to last dispatch {msp:.1} ms · GPU from first to last dispatch {msp:.1} ms · submit wait {msa:.1} ms · preparation {mpp:.2} ms");
    println!("  → outside the GPU: {:.1} ms per token · gaps between dispatches (whole GPU - sum of the kernels measured below): {:.1} ms", tsp - msp, msp - gpu_tot);
    println!("tokens with measurements (pos 28..55): median {tm:.1} ms · GPU kernel total {gpu_tot:.1} ms · CPU parts {cpu_tot:.1} ms · remainder {:.1} ms",
        measured.iter().sum::<f64>() / n - gpu_tot - cpu_tot);
    println!("\nGPU KERNELS (mean per token, sorted by time)");
    println!("{:34} {:>9} {:>6} {:>8} {:>10} {:>8}", "operation", "ms/token", "%", "dispatch", "MB read", "GB/s");
    let mut v: Vec<_> = p.gpu.iter().collect();
    v.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
    for (k, (ms, byte, c)) in v {
        let (ms, mb) = (ms / n, *byte as f64 / n / 1e6);
        let gbs = if mb > 0.0 { format!("{:.1}", mb / ms) } else { "-".into() };
        println!("{k:34} {ms:>9.2} {:>5.1}% {:>8.0} {mb:>10.1} {gbs:>8}", 100.0 * ms / gpu_tot, *c as f64 / n);
    }
    println!("\nCPU PARTS (mean per token)");
    let mut c: Vec<_> = p.cpu.iter().collect();
    c.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());
    for (k, ms) in c { println!("{k:46} {:>8.2} ms", ms / n); }
}

/// `prefill-span <prompt.txt> <output.csv>` (the desktop lags during the prefill): time of EVERY submit of the
/// block prefill (one per segment: embedding, then one layer per segment). No change to the computation: the
/// `span` mode only adds two timestamps per submit. CSV: blocco, segmento, attesa_ms (CPU, submit → end), gpu_ms.
pub fn prefill_span(prompt: &str, output: &str) {
    let t0 = Instant::now();
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(prompt).expect("prompt"), false).expect("encode").get_ids().to_vec();
    println!("loaded in {:.1} s · prompt of {} tokens", t0.elapsed().as_secs_f64(), ids.len());
    m.reset();
    *m.span.borrow_mut() = Some(vec![]);
    let t = Instant::now(); m.prefill(&ids, 0);
    println!("prefill in {:.1} s", t.elapsed().as_secs_f64());
    let s = m.span.borrow_mut().take().unwrap();
    let per_block = crate::engine::N_LAYER + 2;   // embedding + 40 layers + rest
    let mut f = String::from("block,segment,wait_ms,gpu_ms\n");
    for (i, (_, wait, gpu)) in s.iter().enumerate() { f += &format!("{},{},{wait:.2},{gpu:.2}\n", i / per_block, i % per_block); }
    std::fs::write(output, f).expect("write");
    println!("{} submits ({} per block) in {output}", s.len(), per_block);
}

/// `kv-check`: compiles the compact-KV variants of ops.wgsl (f16, q8) and creates the pipelines that use
/// them. No model, almost no GPU memory: it only shows WGSL errors before a long run.
pub fn kv_check() {
    use crate::engine::{KvFormat, ops_source};
    let g = mechta_gpu::Gpu::new();
    for f in [KvFormat::F16, KvFormat::Q8] {
        let src = ops_source(f);
        for e in ["kv_rope", "kv_rope2", "kv_copy", "attn_fd", "attn_fd4", "attn_fa", "attn_fd_combine"] { std::hint::black_box(g.pipeline(&src, e)); }
        println!("{}: ok", f.name());
    }
}

/// `bench-kv <ctx,ctx,...>`: generation time at the given context lengths, with the KV format of MECHTA_KV.
/// No prefill: cache and states are zero (output meaningless, times valid, as MECHTA_FAKE_POS). For each context:
/// median of 30 single steps (`step`) and of 20 MTP cycles (`step2` + `draft2`, the cost of one accepted cycle).
pub fn bench_kv(ctxs: &str) {
    let m = Engine::new(&crate::model_path());
    let kv = crate::engine::kv_format().name();
    let mid = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); (v[v.len() / 2], v[0], v[v.len() - 1]) };
    println!("kv={kv}");
    for c in ctxs.split(',').map(|s| s.trim().parse::<usize>().expect("ctx")) {
        m.reset();
        for pos in c..c + 5 { m.step(0, pos); }   // warm-up
        let mut s = vec![];
        for pos in c + 5..c + 35 { let t = Instant::now(); m.step(0, pos); s.push(t.elapsed().as_secs_f64() * 1e3); }
        let mut d = vec![]; let mut pos = c + 35;
        m.draft(0, pos);
        for _ in 0..20 { let t = Instant::now(); m.step2(0, 0, pos); m.draft2(0, 0, pos); d.push(t.elapsed().as_secs_f64() * 1e3); pos += 2; }
        let ((s50, s0, s1), (d50, d0, d1)) = (mid(s), mid(d));
        println!("kv={kv} ctx={c} single_ms={s50:.1} ({s0:.1}-{s1:.1}) single_tps={:.1} mtp_cycle_ms={d50:.1} ({d0:.1}-{d1:.1}) mtp_tps_all_accepted={:.1}",
            1000.0 / s50, 2000.0 / d50);
    }
}

/// `quality-kv <text> <prefill> <steps> <out.bin>`: prefill of the first <prefill> tokens of the text, then
/// <steps> single steps fed with the TRUE next tokens of the text (teacher forcing). Writes, per step, the true next token
/// (u32) and the full logits (VOCAB f32). Run once per MECHTA_KV and compare the files.
pub fn quality_kv(text: &str, prefill: usize, steps: usize, out: &str) {
    use std::io::Write;
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(text).expect("text"), false).expect("encode").get_ids().to_vec();
    assert!(ids.len() > prefill + steps + 1, "text too short: {} tokens", ids.len());
    m.reset();
    let t = Instant::now(); m.prefill(&ids[..prefill], 0);
    println!("kv={} prefill {prefill} tokens in {:.1} s", crate::engine::kv_format().name(), t.elapsed().as_secs_f64());
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).expect("out"));
    for j in 0..steps {
        let pos = prefill + j;
        m.step(ids[pos], pos);
        let l = m.read("logits", crate::engine::VOCAB);
        f.write_all(&ids[pos + 1].to_le_bytes()).unwrap();
        for x in l { f.write_all(&x.to_le_bytes()).unwrap(); }
    }
    println!("{steps} steps written to {out}");
}

/// `logit-dump <text> <prefill> <steps> <out.bin> [single|double]`: like `quality-kv`, for
/// any engine variant chosen with the environment (MECHTA_KV, MECHTA_PREFILL_ATTN, MECHTA_PREFILL_BLOCK, ...). Prefill
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

/// `block-cost <text> <pos,...> [rounds]` (calibration): wall time of ONE prefill block
/// of each size of BLOCKS (512/128/32/8) at the given context positions, the numbers of the `COST` table of
/// `Engine::prefill` (first measured at ~14K, before attn_fa and the new expert kernels). Fake positions with
/// an empty cache (like `profile-prefill-at`): the attention reads zeros, the times are valid. One warm-up, then
/// `rounds` rounds (default 5), the sizes in a different order each round; median and min-max per size.
pub fn block_cost(text: &str, pos_list: &str, rounds: usize) {
    use crate::engine::BLOCKS;
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(text).expect("text"), false).expect("encode").get_ids().to_vec();
    assert!(ids.len() >= BLOCKS[0], "text too short");
    for pos in pos_list.split(',').map(|s| s.trim().parse::<usize>().expect("pos")) {
        m.reset();
        for ib in 0..BLOCKS.len() { m.single_block(&ids[..BLOCKS[ib]], pos, ib); }   // warm-up
        let mut t: Vec<Vec<f64>> = vec![vec![]; BLOCKS.len()];
        for r in 0..rounds {
            for k in 0..BLOCKS.len() {
                let ib = (k + r) % BLOCKS.len();
                let t0 = Instant::now(); m.single_block(&ids[..BLOCKS[ib]], pos, ib);
                t[ib].push(t0.elapsed().as_secs_f64() * 1e3);
            }
        }
        let mut cells = vec![];
        for (ib, v) in t.iter_mut().enumerate() {
            v.sort_by(|a, b| a.total_cmp(b));
            let mid = v[v.len() / 2];
            cells.push(format!("{} -> {:.0} ms ({:.0}-{:.0}; {:.2} ms/token)", BLOCKS[ib], mid, v[0], v[v.len() - 1], mid / BLOCKS[ib] as f64));
        }
        println!("pos={pos} rounds={rounds} · {}", cells.join(" · "));
    }
}

/// `kv-dump <text> <prefill> <steps> <out.bin>` (compact KV bug): prefill, then <steps> single steps fed
/// with the true tokens, then writes the raw K and V words of the first attention layer for all positions written
/// (f32 values, or raw f16 / q8 words + scales). Compare a f32 run with a f16 run position by position.
pub fn kv_dump(text: &str, prefill: usize, steps: usize, out: &str) {
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(text).expect("text"), false).expect("encode").get_ids().to_vec();
    m.reset(); m.prefill(&ids[..prefill], 0);
    for j in 0..steps { m.step(ids[prefill + j], prefill + j); }
    let r = m.read_root(prefill + steps);
    let (k, v) = &r.kv[0];
    let b: Vec<u8> = k.iter().chain(v.iter()).flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(out, b).expect("out");
    println!("kv={} positions {} · K {} words · V {} words -> {out}", crate::engine::kv_format().name(), prefill + steps, k.len(), v.len());
}

/// `bench-attn <ctx,...>` (speed path 1): GPU time per kernel of the double step at the given contexts
/// (fake positions, cache of zeros: times valid). Median of 8 profiled double steps; prints the attention kernels and
/// the total. MECHTA_FD_PHASE=1|2 skips phase 2|1 of attn_fd2 (output wrong, for measuring only).
pub fn bench_attn(ctxs: &str) {
    let m = Engine::new(&crate::model_path());
    for c in ctxs.split(',').map(|s| s.trim().parse::<usize>().expect("ctx")) {
        m.reset();
        let mut per: std::collections::HashMap<String, Vec<f64>> = std::collections::HashMap::new();
        let mut tot = vec![];
        for k in 0..10 {
            let r = m.step2_measured(0, 0, c + 2 * k);
            if k < 2 { continue; }
            let mut agg: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
            for (n, ms, _) in &r { *agg.entry(n.clone()).or_default() += ms; }
            tot.push(agg.values().sum::<f64>());
            for (n, ms) in agg { per.entry(n).or_default().push(ms); }
        }
        let mid = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
        let mut attn: Vec<(String, f64)> = per.iter_mut().filter(|(n, _)| n.starts_with("attn_fd")).map(|(n, v)| (n.clone(), mid(v))).collect();
        attn.sort_by(|a, b| a.0.cmp(&b.0));
        println!("ctx={c} kv={} total {:.1} ms · {}", crate::engine::kv_format().name(), mid(&mut tot),
            attn.iter().map(|(n, ms)| format!("{n} {ms:.2} ms")).collect::<Vec<_>>().join(" · "));
    }
}

/// `check-attn2 <text> <prefill> <out.bin>`: prefill of real text, then 4 double steps fed with the true
/// tokens; writes the logits of both tokens of each step. Run with two MECHTA_ATTN2 variants and compare the files.
pub fn check_attn2(text: &str, prefill: usize, out: &str) {
    use std::io::Write;
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(text).expect("text"), false).expect("encode").get_ids().to_vec();
    m.reset(); m.prefill(&ids[..prefill], 0);
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).expect("out"));
    for k in 0..4 {
        let pos = prefill + 2 * k;
        m.step2(ids[pos], ids[pos + 1], pos);
        for x in m.read("logits", 2 * crate::engine::VOCAB) { f.write_all(&x.to_le_bytes()).unwrap(); }
    }
    println!("4 double steps after {prefill} tokens -> {out}");
}

/// `bench-step <ctx>` (speed path 3): every kernel of the double step and of the double MTP draft at the
/// given context, median of 8 profiled steps, with bytes read and GB/s (for the kernels whose weights are counted).
pub fn bench_step(ctx: usize) {
    let m = Engine::new(&crate::model_path());
    m.reset();
    let mut per: std::collections::HashMap<String, (Vec<f64>, u64, usize)> = std::collections::HashMap::new();
    for k in 0..10 {
        let r = m.step2_measured(0, 0, ctx + 2 * k);
        if k < 2 { continue; }
        let mut agg: std::collections::HashMap<String, (f64, u64, usize)> = std::collections::HashMap::new();
        for (n, ms, b) in &r { let e = agg.entry(n.clone()).or_default(); e.0 += ms; e.1 += b; e.2 += 1; }
        for (n, (ms, b, c)) in agg { let e = per.entry(n).or_default(); e.0.push(ms); e.1 = b; e.2 = c; }
    }
    let mut v: Vec<(String, f64, u64, usize)> = per.into_iter().map(|(n, (mut t, b, c))| { t.sort_by(|a, b| a.partial_cmp(b).unwrap()); (n, t[t.len() / 2], b, c) }).collect();
    v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let tot: f64 = v.iter().map(|x| x.1).sum(); let tb: u64 = v.iter().map(|x| x.2).sum();
    println!("double step at ctx {ctx}: {tot:.1} ms, weights counted {:.2} GB ({:.0} GB/s overall)", tb as f64 / 1e9, tb as f64 / 1e6 / tot);
    println!("{:46} {:>8} {:>6} {:>9} {:>7}", "kernel", "ms", "n", "MB", "GB/s");
    for (n, ms, b, c) in v.iter().take(30) {
        let gbs = if *b > 0 { format!("{:.0}", *b as f64 / 1e6 / ms) } else { "-".into() };
        println!("{:46} {:>8.2} {:>6} {:>9.1} {:>7}", n, ms, c, *b as f64 / 1e6, gbs);
    }
}

/// `bench-cycle <ctx>`: unprofiled double steps as in the server, with the `span` record of each:
/// CPU time to encode the commands, wait from submit to end, GPU time first-to-last dispatch, and wall time.
pub fn bench_cycle(ctx: usize) {
    let m = Engine::new(&crate::model_path());
    m.reset();
    *m.span.borrow_mut() = Some(vec![]);
    let mut wall = vec![];
    for k in 0..24 { let t = Instant::now(); m.step2(0, 0, ctx + 2 * k); wall.push(t.elapsed().as_secs_f64() * 1e3); }
    let s = m.span.borrow_mut().take().unwrap();
    let mid = |mut v: Vec<f64>| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[v.len() / 2] };
    let col = |i: usize| mid(s[4..].iter().map(|x| match i { 0 => x.0, 1 => x.1, _ => x.2 }).collect());
    println!("ctx={ctx} double step: wall {:.1} ms · encode (CPU) {:.1} ms · submit→end {:.1} ms · GPU first→last dispatch {:.1} ms",
        mid(wall[4..].to_vec()), col(0), col(1), col(2));
}

/// `profile-prefill-at <text> <n> <pos,pos,...>` (prefill at long context): prefill of the first <n> tokens
/// of the text starting at each position (cache and states zero: times valid, output meaningless, as bench-kv).
/// For each position: wall time without profiling, then GPU time per kernel with profiling; attention kernels
/// (names with "attn") against the rest, and the 8 most expensive kernels.
pub fn profile_prefill_at(text: &str, n: usize, pos_list: &str) {
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(text).expect("text"), false).expect("encode").get_ids().to_vec();
    assert!(ids.len() >= n, "text too short");
    for pos in pos_list.split(',').map(|s| s.trim().parse::<usize>().expect("pos")) {
        m.reset();
        let t = Instant::now(); m.prefill(&ids[..n], pos); let wall = t.elapsed().as_secs_f64() * 1e3;
        m.reset();
        *m.prof.borrow_mut() = Some(Profile::default());
        m.prefill(&ids[..n], pos);
        let p = m.prof.borrow_mut().take().unwrap();
        let tot: f64 = p.gpu.values().map(|v| v.0).sum();
        let attn: f64 = p.gpu.iter().filter(|(k, _)| k.contains("attn")).map(|(_, v)| v.0).sum();
        println!("pos={pos} n={n} wall {:.2} ms/token ({:.0} t/s) · GPU {:.2} ms/token · attention {:.2} ms/token ({:.0}%) · rest {:.2} ms/token",
            wall / n as f64, n as f64 * 1e3 / wall, tot / n as f64, attn / n as f64, 100.0 * attn / tot, (tot - attn) / n as f64);
        let mut v: Vec<_> = p.gpu.iter().collect();
        v.sort_by(|a, b| b.1.0.partial_cmp(&a.1.0).unwrap());
        for (k, (ms, _, c)) in v.iter().take(8) { println!("    {k:40} {:>8.3} ms/token {:>5.1}% · {c} dispatch", ms / n as f64, 100.0 * ms / tot); }
        // GPU time of each submit: the longest one is how long the desktop waits for the GPU
        m.reset();
        *m.span.borrow_mut() = Some(vec![]);
        m.prefill(&ids[..n], pos);
        let sp = m.span.borrow_mut().take().unwrap();
        let mut g: Vec<f64> = sp.iter().map(|x| x.2).collect();
        g.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("    submits: {} · GPU ms per submit: median {:.1} · p95 {:.1} · max {:.1}", g.len(), g[g.len() / 2], g[g.len() * 95 / 100], g[g.len() - 1]);
    }
}

/// `probe-experts <text> <n> <out>`: deterministic probe of the prefill for the expert
/// GEMM. Prefill of the first <n> tokens of the text (multiple of 512) in blocks of 512, one submit per layer; after each
/// layer it writes: block, layer, max |h2| (input of gate/up), max |esw| (input of down), how many values of "x" are not
/// finite, and 8 rows of "x" (tokens 0, 64, .., 448 of the block). Run once per MECHTA_GEMM_EXPERTS and compare the two files
/// with `compare-probe`. Same input in both runs: the differences come only from the kernels.
pub fn probe_experts(text: &str, n: usize, out: &str) {
    use std::io::Write;
    let m = Engine::new(&crate::model_path());
    let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
    let ids: Vec<u32> = tk.encode(std::fs::read_to_string(text).expect("text"), false).expect("encode").get_ids().to_vec();
    assert!(ids.len() >= n && n % 512 == 0, "n must be a multiple of 512 and at most the text");
    m.reset();
    let mut f = std::io::BufWriter::new(std::fs::File::create(out).expect("out"));
    let maxabs = |v: &[f32]| v.iter().fold(0f32, |a, x| a.max(x.abs()));
    let t = Instant::now();
    m.prefill_probe(&ids[..n], &mut |b, il| {
        if il == 0 { return; }
        let (h2, esw, x) = (m.read_p("h2", 512 * 2048), m.read_p("esw", 512 * 8 * 512), m.read_p("x", 512 * 2048));
        let nonfin = x.iter().filter(|v| !v.is_finite()).count() as u32;
        f.write_all(&(b as u32).to_le_bytes()).unwrap(); f.write_all(&(il as u32 - 1).to_le_bytes()).unwrap();
        f.write_all(&maxabs(&h2).to_le_bytes()).unwrap(); f.write_all(&maxabs(&esw).to_le_bytes()).unwrap();
        f.write_all(&nonfin.to_le_bytes()).unwrap();
        for r in 0..8 { for v in &x[r * 64 * 2048..r * 64 * 2048 + 2048] { f.write_all(&v.to_le_bytes()).unwrap(); } }
    });
    println!("probe: {n} tokens, {} blocks, in {:.1} s, written to {out}", n / 512, t.elapsed().as_secs_f64());
}

/// `compare-probe <ref> <var>`: the two files of `probe-experts`. Prints the largest |h2| and |esw| seen
/// (f16 holds up to 65504), the non-finite values, and per block the largest relative difference of "x" over the layers
/// (max |var - ref| / max |ref| on the 8 rows), with the first layer of the block where it passes 1e-2.
pub fn compare_probe(a: &str, b: &str) {
    let (ra, rb) = (std::fs::read(a).expect("ref"), std::fs::read(b).expect("var"));
    const REC: usize = 20 + 8 * 2048 * 4;
    assert!(ra.len() == rb.len() && ra.len() % REC == 0, "files of different size");
    let u = |r: &[u8], o: usize| u32::from_le_bytes(r[o..o + 4].try_into().unwrap());
    let fl = |r: &[u8], o: usize| f32::from_le_bytes(r[o..o + 4].try_into().unwrap());
    let (mut mh, mut me, mut nf_a, mut nf_b) = (0f32, 0f32, 0u32, 0u32);
    let mut block = u32::MAX; let mut worst = (0f32, 0u32); let mut first: Option<u32> = None;
    let print = |blk: u32, p: (f32, u32), pr: Option<u32>| println!("block {blk:3}: max rel diff {:.2e} (layer {:2}) · first layer > 1e-2: {}", p.0, p.1, pr.map_or("-".into(), |l| l.to_string()));
    for k in 0..ra.len() / REC {
        let (x, y) = (&ra[k * REC..(k + 1) * REC], &rb[k * REC..(k + 1) * REC]);
        let (blk, il) = (u(x, 0), u(x, 4));
        mh = mh.max(fl(x, 8)).max(fl(y, 8)); me = me.max(fl(x, 12)).max(fl(y, 12));
        nf_a += u(x, 16); nf_b += u(y, 16);
        if blk != block { if block != u32::MAX { print(block, worst, first); } block = blk; worst = (0.0, 0); first = None; }
        let (mut d, mut r) = (0f32, 0f32);
        for i in 0..8 * 2048 { let (p, q) = (fl(x, 20 + 4 * i), fl(y, 20 + 4 * i)); d = d.max((p - q).abs()); r = r.max(p.abs()); }
        let rel = if r > 0.0 { d / r } else { d };
        if rel > worst.0 || rel.is_nan() { worst = (rel, il); }
        if first.is_none() && !(rel <= 1e-2) { first = Some(il); }
    }
    print(block, worst, first);
    println!("largest |h2| {mh:.1} · largest |esw| {me:.1} (f16 max 65504) · non-finite values of x: ref {nf_a}, var {nf_b}");
}
