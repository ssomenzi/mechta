//! The engine behind the trait `mechta_api::Engine`, for the HTTP server.
//!
//! One request = an already formatted prompt. Steps:
//! 1. prefix reuse: if the request starts with the tokens covered by the photo of the previous request, the
//!    DeltaNet + conv states of the photo are restored and it restarts from there (the KV cache is overwritten in order);
//!    otherwise everything is reset.
//! 2. new photo at the LAST `<|im_start|>` of the prompt (state after the tokens before it). Rationale: the template of
//!    Qwen3.6 removes the reasoning of past turns, so the next turn certainly shares the prompt up to
//!    the last `<|im_start|>` (the start of the assistant's answer), and almost never beyond.
//! 3. block prefill up to the tail; the tail (last `tail_mtp` tokens, default 32) goes one token at a time with
//!    `step` + `draft`, to fill the cache of the MTP layer only there (the drafts are attempts: the output stays
//!    exact even with a short MTP cache; `mtp_start` tells the MTP attention where to look from).
//! 4. greedy generation with the MTP loop (double step, draft accepted or rejected), stop on <|im_end|> / <|endoftext|>.
use crate::engine::{Engine as M, Root, ctx};
use mechta_api::{Outcome, Finish, Engine, Params};
use mechta_gpu::trace;
use std::time::Instant;

const IM_START: u32 = 248045;
const IM_END: u32 = 248046;
const ENDOFTEXT: u32 = 248044;
const THINK_END: u32 = 248069;
/// Text that closes the reasoning when the thinking budget is reached (official Qwen method, quickstart of
/// qwen.readthedocs.io: the sentence, then `</think>`; the model goes on with the answer).
const CLOSE_THINKING: &str = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n";

pub struct Session {
    m: M,
    tk: tokenizers::Tokenizer,
    /// tokens covered by the photo: the photographed state is the one after these tokens (empty = no valid photo)
    photo_tok: Vec<u32>,
    tail_mtp: usize,
    /// ROOT (prefix across sessions): state after the system message (up to the second <|im_start|>),
    /// in RAM and on disk (`root_file`). A new session with the same system (opencode: prompt + tools, ~17K tokens)
    /// restarts from there. MECHTA_ROOT=no turns it off.
    root: Option<(Vec<u32>, Root)>,
    use_root: bool,
    /// the model id of /v1/models (see `model_id`)
    name: String,
}

/// The id the server gives the model: the file name in lower case, cut before "-mtp" or "-ud-" if there is
/// one: Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf -> "tiel-coder-35b-a3b", Qwen3.6-35B-A3B-UD-Q4_K_S.gguf -> "qwen3.6-35b-a3b".
/// `MECHTA_MODEL_ID` overrides it (no default model any more).
pub fn model_id(path: &str) -> String {
    if let Ok(id) = std::env::var("MECHTA_MODEL_ID") && !id.is_empty() { return id; }
    let stem = std::path::Path::new(path).file_stem().map(|s| s.to_string_lossy().to_lowercase()).unwrap_or_default();
    let trim = ["-mtp", "-ud-"].iter().filter_map(|t| stem.find(t).filter(|&i| i > 0)).min();
    match trim { Some(i) => stem[..i].to_string(), None => stem }
}

/// Folder of the saved roots: `MECHTA_CACHE`.
fn root_dir() -> String { crate::cache_dir() }
/// Older versions kept one root file per KV format; converted to a per-prompt file at start-up.
fn legacy_root_file() -> String {
    match crate::engine::kv_format() {
        crate::engine::KvFormat::F32 => format!("{}/radice.bin", root_dir()),
        f => format!("{}/radice-{}.bin", root_dir(), f.name()),
    }
}
/// FNV-1a 64 of the root tokens: the key of a saved root (like ds4, which keys its KV cache by the SHA1 of the prefix).
fn root_key(tok: &[u32]) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for t in tok { for b in t.to_le_bytes() { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); } }
    format!("{h:016x}")
}
/// One root file per system prompt, per KV format and per model file (a root is the state of one model:
/// Qwen's roots must never be loaded into another model). Every model carries its hash, there is no
/// default model any more (the roots saved before for Qwen are not found and get rebuilt once).
fn root_file(tok: &[u32]) -> String {
    let model = root_key(&crate::model_path().bytes().map(u32::from).collect::<Vec<_>>());
    format!("{}/root-{}-{model}-{}.bin", root_dir(), crate::engine::kv_format().name(), root_key(tok))
}
/// Keep at most MECHTA_ROOTS (default 6) root files of the current format: the least recently used go.
fn prune_roots() {
    let keep: usize = std::env::var("MECHTA_ROOTS").ok().and_then(|v| v.parse().ok()).unwrap_or(6).max(1);
    let prefix = format!("root-{}-", crate::engine::kv_format().name());
    let Ok(rd) = std::fs::read_dir(root_dir()) else { return };
    let mut v: Vec<(std::time::SystemTime, std::path::PathBuf)> = rd.flatten()
        .filter(|e| e.file_name().to_string_lossy().starts_with(&prefix) && e.file_name().to_string_lossy().ends_with(".bin"))
        .filter_map(|e| Some((e.metadata().ok()?.modified().ok()?, e.path()))).collect();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    for (_, p) in v.into_iter().skip(keep) { if std::fs::remove_file(&p).is_ok() { eprintln!("root removed (more than {keep}): {}", p.display()); } }
}
/// Marks a root file as just used (the pruning keeps the most recently used).
fn touch(path: &str) {
    if let Ok(f) = std::fs::File::options().append(true).open(path) { let _ = f.set_modified(std::time::SystemTime::now()); }
}
/// the root is taken only if the system message is at least this long (below that it does not pay off)
const ROOT_MIN: usize = 2048;

fn write_root_file(tok: &[u32], r: &Root) -> std::io::Result<()> {
    use std::io::Write;
    let file = root_file(tok);
    std::fs::create_dir_all(std::path::Path::new(&file).parent().unwrap())?;
    let tmp = format!("{file}.tmp");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    f.write_all(b"MECHTA-ROOT-1\n")?;
    let n = |f: &mut std::io::BufWriter<std::fs::File>, x: u64| f.write_all(&x.to_le_bytes());
    n(&mut f, r.p as u64)?; n(&mut f, r.ssm.len() as u64)?; n(&mut f, r.kv.len() as u64)?;
    for t in tok { f.write_all(&t.to_le_bytes())?; }
    let v = |f: &mut std::io::BufWriter<std::fs::File>, x: &Vec<f32>| -> std::io::Result<()> {
        f.write_all(&(x.len() as u64).to_le_bytes())?; for y in x { f.write_all(&y.to_le_bytes())?; } Ok(()) };
    for x in &r.ssm { v(&mut f, x)?; }
    for x in &r.conv { v(&mut f, x)?; }
    for (k, w) in &r.kv { v(&mut f, k)?; v(&mut f, w)?; }
    f.flush()?; drop(f);
    std::fs::rename(&tmp, &file)?;
    prune_roots();
    Ok(())
}

fn read_root_file(path: &str) -> Option<(Vec<u32>, Root)> {
    let b = std::fs::read(path).ok()?;
    let header = b"MECHTA-ROOT-1\n";
    if !b.starts_with(header) { return None; }
    let mut o = header.len();
    let u64_ = |o: &mut usize| -> Option<u64> { let x = u64::from_le_bytes(b.get(*o..*o + 8)?.try_into().ok()?); *o += 8; Some(x) };
    let p = u64_(&mut o)? as usize; let ns = u64_(&mut o)? as usize; let nk = u64_(&mut o)? as usize;
    let mut tok = Vec::with_capacity(p);
    for _ in 0..p { tok.push(u32::from_le_bytes(b.get(o..o + 4)?.try_into().ok()?)); o += 4; }
    let vector = |o: &mut usize| -> Option<Vec<f32>> {
        let n = u64::from_le_bytes(b.get(*o..*o + 8)?.try_into().ok()?) as usize; *o += 8;
        let v = b.get(*o..*o + 4 * n)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(); *o += 4 * n; Some(v) };
    let mut ssm = vec![]; for _ in 0..ns { ssm.push(vector(&mut o)?); }
    let mut conv = vec![]; for _ in 0..ns { conv.push(vector(&mut o)?); }
    let mut kv = vec![]; for _ in 0..nk { let k = vector(&mut o)?; let w = vector(&mut o)?; kv.push((k, w)); }
    Some((tok, Root { p, ssm, conv, kv }))
}

impl Default for Session { fn default() -> Self { Self::new() } }

impl Session {
    /// Loads the model (reserving 2 GB of VRAM during loading, as in the tests) and the tokenizer.
    pub fn new() -> Self {
        let m = M::new_with_reserve(&crate::model_path(), Some(2));
        let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("tokenizer");
        let tail_mtp = std::env::var("MECHTA_MTP_TAIL").ok().and_then(|v| v.parse().ok()).unwrap_or(8usize).max(1);
        let use_root = std::env::var("MECHTA_ROOT").map(|v| v != "no").unwrap_or(true);
        // the legacy single root becomes a per-prompt file; then the root of a request is loaded when it arrives
        let root = if use_root { read_root_file(&legacy_root_file()) } else { None };
        if let Some((t, _)) = &root {
            let to = root_file(t);
            match std::fs::rename(legacy_root_file(), &to) { Ok(()) => eprintln!("root converted: {} -> {to} ({} tokens)", legacy_root_file(), t.len()), Err(e) => eprintln!("root not converted: {e}") }
        }
        Session { m, tk, photo_tok: vec![], tail_mtp, root, use_root, name: model_id(&crate::model_path()) }
    }
    pub fn tokenize(&self, s: &str) -> Vec<u32> { self.tk.encode(s, false).expect("encode").get_ids().to_vec() }
    /// Like `tokenize`, but a tokenizer error goes back to the caller (this is the path of the
    /// HTTP requests, where a panic would make the server exit).
    pub fn tokenize_ok(&self, s: &str) -> Result<Vec<u32>, String> { Ok(self.tk.encode(s, false).map_err(|e| e.to_string())?.get_ids().to_vec()) }
    /// Forgets the photo: the next request redoes the prefill from zero (for the checks).
    pub fn forget(&mut self) { self.photo_tok.clear(); }
}

/// Sampling: temperature, top_k, top_p over the logits of one position. temperature <= 0 → argmax.
/// With MTP, sampling uses the model's TRUE distribution at every position: the draft counts only if the sampled token
/// matches it (so the output has the same distribution as without MTP; MTP only saves steps).
pub struct Sampler { t: f32, k: usize, p: f32, state: u64 }
impl Sampler {
    pub fn new(par: &Params) -> Self {
        let seed = if par.seed != 0 { par.seed } else {
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(1) };
        Sampler { t: par.temperature, k: if par.top_k == 0 { 1000 } else { par.top_k.min(1000) }, p: par.top_p, state: seed | 1 }
    }
    pub fn greedy(&self) -> bool { self.t <= 0.0 }
    fn random(&mut self) -> f64 {   // xorshift64*, [0, 1)
        self.state ^= self.state >> 12; self.state ^= self.state << 25; self.state ^= self.state >> 27;
        (self.state.wrapping_mul(0x2545F4914F6CDD1D) >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn sample(&mut self, logit: &[f32]) -> u32 {
        let ts = trace::start();
        let tok = self.sample_(logit);
        trace::span(trace::Kind::Sample, ts, |e| e.label = "session");
        tok
    }
    fn sample_(&mut self, logit: &[f32]) -> u32 {
        // the k highest logits (partial selection), then softmax with temperature, top_p cut, draw
        let mut idx: Vec<u32> = (0..logit.len() as u32).collect();
        let k = self.k.min(idx.len());
        idx.select_nth_unstable_by(k - 1, |&a, &b| logit[b as usize].total_cmp(&logit[a as usize]));
        let mut top: Vec<(u32, f64)> = idx[..k].iter().map(|&i| (i, logit[i as usize] as f64)).collect();
        top.sort_by(|a, b| b.1.total_cmp(&a.1));
        let m = top[0].1; let t = self.t as f64;
        let mut tot = 0.0;
        for e in top.iter_mut() { e.1 = ((e.1 - m) / t).exp(); tot += e.1; }
        let mut cum = 0.0; let mut n = 0;
        for e in top.iter() { cum += e.1 / tot; n += 1; if cum >= self.p as f64 { break; } }
        let top = &top[..n]; let tot: f64 = top.iter().map(|e| e.1).sum();
        let mut r = self.random() * tot;
        for e in top { r -= e.1; if r <= 0.0 { return e.0; } }
        top[n - 1].0
    }
}

/// Passes a generated token to the caller. Returns Some(finish) if the generation must end.
/// Thinking budget: `thinking` is true while the output is inside the reasoning; when `thought` reaches
/// `budget` (0 = no limit) `emit` sets `cut` and stops the generation, and the caller closes the reasoning.
/// Context n-gram draft (measurement): the longest suffix of `h` (4, 3 or 2 tokens) found earlier in `h`, most recent
/// match first; returns (suffix length, up to `k` tokens that followed it). (0, []) if none.
fn ngram_next(h: &[u32], k: usize) -> (usize, Vec<u32>) {
    for n in [4usize, 3, 2] {
        if h.len() <= n { continue; }
        let suf = &h[h.len() - n..];
        if let Some(i) = (0..h.len() - n).rev().find(|&i| &h[i..i + n] == suf) {
            return (n, h[i + n..(i + n + k).min(h.len())].to_vec());
        }
    }
    (0, vec![])
}

struct Output<'a> { tk: &'a tokenizers::Tokenizer, emitted: Vec<u32>, k0: usize, max: usize, piece: &'a mut dyn FnMut(&str) -> bool,
                    thinking: bool, thought: usize, budget: usize, cut: bool }
impl Output<'_> {
    fn emit(&mut self, tok: u32) -> Option<Finish> {
        let ts = trace::start();
        let f = self.emit_(tok);
        trace::span(trace::Kind::Emit, ts, |e| e.label = "session");
        f
    }
    fn emit_(&mut self, tok: u32) -> Option<Finish> {
        if tok == IM_END || tok == ENDOFTEXT { return Some(Finish::Stop); }
        self.emitted.push(tok);
        // incremental decoding: the tokens not yet printed are decoded; if the text ends in the middle of a character
        // (U+FFFD) it waits for the next token
        let s = self.tk.decode(&self.emitted[self.k0..], false).unwrap_or_default();
        if !s.ends_with('\u{FFFD}') {
            self.k0 = self.emitted.len();
            if !s.is_empty() && !(self.piece)(&s) { return Some(Finish::Cancelled); }
        }
        if self.emitted.len() >= self.max { return Some(Finish::Length); }
        if self.thinking {
            if tok == THINK_END { self.thinking = false; }
            else { self.thought += 1; if self.budget > 0 && self.thought >= self.budget { self.cut = true; return Some(Finish::Stop); } }
        }
        None
    }
}

impl Engine for Session {
    fn model_name(&self) -> &str { &self.name }
    fn max_context(&self) -> usize { ctx() }
    fn count_tokens(&self, text: &str) -> Result<usize, String> { Ok(self.tokenize_ok(text)?.len()) }

    fn generate(&mut self, prompt: &str, p: &Params, piece: &mut dyn FnMut(&str) -> bool) -> Result<Outcome, String> {
        let t0 = Instant::now();
        let trace_from = trace::mark();
        let tp = trace::start();
        let toks = self.tokenize_ok(prompt)?;
        let n = toks.len();
        if n == 0 { return Err("empty prompt".into()); }
        if n + 4 >= ctx() { return Err(format!("prompt of {n} tokens: beyond the maximum context of {}", ctx())); }
        let m = &self.m;
        // 1. prefix reuse
        let k = self.photo_tok.len();
        let reuse = k > 0 && k < n && toks[..k] == self.photo_tok[..];
        // root: valid if the prompt starts with its tokens; used only if it covers more than the photo of the previous turn
        // the root of this prompt (end of the system message), if long enough: from memory, else from its file
        if self.use_root && toks.first() == Some(&IM_START)
            && let Some(q) = toks.iter().skip(1).position(|&t| t == IM_START).map(|q| q + 1).filter(|&q| q >= ROOT_MIN && q < n)
        {
            let in_memory = matches!(&self.root, Some((rt, _)) if rt[..] == toks[..q]);
            let path = root_file(&toks[..q]);
            if !in_memory && std::path::Path::new(&path).exists() {
                let tl = Instant::now(); let tr = trace::start();
                let r = read_root_file(&path);
                trace::span(trace::Kind::RootLoad, tr, |e| { e.label = "session"; e.pos = q as u32; });
                if let Some(r) = r {
                    eprintln!("root loaded from {path}: {} tokens in {:.0} ms", r.0.len(), tl.elapsed().as_secs_f64() * 1e3);
                    self.root = Some(r);
                }
            }
            if std::path::Path::new(&path).exists() { touch(&path); }
        }
        let kr = match &self.root { Some((rt, _)) if rt.len() < n && toks[..rt.len()] == rt[..] => rt.len(), _ => 0 };
        let done = if reuse && k >= kr { m.restore_photo(); k }
            else if kr > 0 { m.reset(); self.photo_tok.clear(); m.write_root(&self.root.as_ref().unwrap().1); kr }
            else { m.reset(); self.photo_tok.clear(); 0 };
        // point of the new root: end of the system message (second <|im_start|>), if long and if different from the current one
        let r = if self.use_root && toks.first() == Some(&IM_START) {
            toks.iter().skip(1).position(|&t| t == IM_START).map(|q| q + 1)
                .filter(|&q| q >= ROOT_MIN && q > done && kr != q)
        } else { None };
        // 2. point of the new photo, 3. start of the single-step tail
        let b = toks.iter().rposition(|&t| t == IM_START).filter(|&b| b > done);
        let t = n.saturating_sub(self.tail_mtp).max(done);
        // if the end of the system falls inside the single-step tail (short question), the tail starts there: this way the root
        // is taken at a block boundary (otherwise the root is never taken)
        let t = match r { Some(q) if q >= t && q < n => q, _ => t };
        m.mtp_start.set((t + 1) as u32);
        trace::span(trace::Kind::PrefixReuse, tp, |e| { e.label = "session"; e.pos = done as u32; e.ntok = n as u32; });
        let ms_rip = t0.elapsed().as_secs_f64() * 1e3;
        let (mut ms_bl, mut n_bl, mut ms_tail, mut n_tail) = (0.0, 0usize, 0.0, 0usize);
        let mut y = 0; let mut i = done;
        let r = r.filter(|&q| q <= t);
        while i < n {
            if r == Some(i) {
                let tr = Instant::now();
                let new_root = m.read_root(i);
                let tok_r = toks[..i].to_vec();
                eprintln!("new root: {i} tokens, read in {:.0} ms", tr.elapsed().as_secs_f64() * 1e3);
                // to disk in a thread (~800 MB): the request does not wait. The copy for the thread costs little extra RAM.
                let (t2, r2) = (tok_r.clone(), Root { p: new_root.p, ssm: new_root.ssm.clone(), conv: new_root.conv.clone(), kv: new_root.kv.clone() });
                std::thread::spawn(move || {
                    trace::set_thread(trace::ROOT_SAVER); let ts = trace::start();
                    let w = write_root_file(&t2, &r2);
                    trace::span(trace::Kind::RootSave, ts, |e| { e.label = "session"; e.pos = t2.len() as u32; });
                    match w { Ok(()) => eprintln!("root saved in {}", root_file(&t2)), Err(e) => eprintln!("root NOT saved: {e}") }
                });
                self.root = Some((tok_r, new_root));
            }
            if b == Some(i) { m.photograph(); self.photo_tok = toks[..i].to_vec(); }
            if i < t {
                let mut end = t;
                if let Some(b) = b { if b > i && b < end { end = b; } }
                if let Some(r) = r { if r > i && r < end { end = r; } }
                let tb = Instant::now();
                // an empty piece is a probe: it fails when the client has gone (the server drops the channel)
                if !m.prefill_cancellable(&toks[i..end], i, &mut || piece("")) {
                    eprintln!("prefill stopped at token {i} of {n}: the client has gone");
                    return Ok(Outcome { token_prompt: n, reused_tokens: done, generated_tokens: 0, end: Finish::Cancelled,
                                      ms_prefill: t0.elapsed().as_secs_f64() * 1e3, ms_generation: 0.0 });
                }
                ms_bl += tb.elapsed().as_secs_f64() * 1e3; n_bl += end - i;
                i = end;
            } else {
                let tc = Instant::now();
                y = m.step(toks[i], i);
                if i + 1 < n { m.draft(toks[i + 1], i + 1); }
                ms_tail += tc.elapsed().as_secs_f64() * 1e3; n_tail += 1;
                i += 1;
            }
        }
        let ms_prefill = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("prefill: tokenize+restore {ms_rip:.0} ms · blocks {n_bl} tokens in {ms_bl:.0} ms · tail {n_tail} tokens in {ms_tail:.0} ms · total {ms_prefill:.0} ms");
        // 4. generation
        let mut camp = Sampler::new(p);
        if !camp.greedy() { y = camp.sample(&m.read("logits", crate::engine::VOCAB)); }
        let t1 = Instant::now();
        let thinking = prompt.ends_with("<think>\n");
        let close = if thinking && p.reasoning_budget > 0 { self.tokenize(CLOSE_THINKING) } else { vec![] };
        let mut u = Output { tk: &self.tk, emitted: vec![], k0: 0, max: p.max_token.max(1), piece,
                             thinking, thought: 0, budget: if thinking { p.reasoning_budget } else { 0 }, cut: false };
        // MTP acceptance. MECHTA_MTP_ACCEPT=exact (default): the token is sampled from the model and the
        // draft counts only if it is the same token (same distribution as without MTP). =greedy: like the default of
        // ds4 (antirez), a draft equal to the model's top token is accepted as is; otherwise the token is sampled.
        let greedy_accept = std::env::var("MECHTA_MTP_ACCEPT").map(|v| v == "greedy").unwrap_or(false);
        // MECHTA_DRAFT_PROBE=<file> (measurement): at the end of every MTP cycle,
        // 2 more MTP drafts in a chain and the next 3 tokens of the context n-gram, one line per cycle; the generated
        // tokens are written at the end. The generated text does not change.
        let mut probe = std::env::var("MECHTA_DRAFT_PROBE").ok().filter(|_| p.mtp)
            .map(|f| std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).append(true).open(f).expect("MECHTA_DRAFT_PROBE file")));
        let mut row1: bool;   // where draft/draft2 left the MTP state (probe only)
        let (mut cycles, mut accepted, mut steps4) = (0usize, 0usize, 0usize);
        // wall time of the MTP cycles by phase (cycles profiled per kernel are left out), and with
        // MECHTA_GEN_PROFILE=1 the GPU time of every kernel of one cycle in 8
        let gen_profile = std::env::var("MECHTA_GEN_PROFILE").is_ok_and(|v| v == "1");
        let mut prof_sum: std::collections::HashMap<String, (f64, u32)> = std::collections::HashMap::new();
        let (mut w_double, mut w_sample, mut w_draft, mut w_restore, mut w_cycles) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
        let mut profiled = 0usize;
        *m.prof.borrow_mut() = None;
        // `start`: position of the next token to emit (y). `fed`: positions the model has processed (the state covers
        // the positions before it); a token emitted at a position >= fed is not processed yet.
        let mut start = n;
        let mut fed;
        let mut budget_cuts = 0usize;
        let end = 'all: loop {
        fed = start;
        let f = 'g: {
            if let Some(f) = u.emit(y) { break 'g f; }
            if !p.mtp {
                // without MTP: one step per token (same sampling)
                let (mut pos, mut cur) = (start, y);
                loop {
                    if pos + 1 >= ctx() { break 'g Finish::Length; }
                    let mut a = m.step(cur, pos);
                    fed = pos + 1;
                    if !camp.greedy() { a = camp.sample(&m.read("logits", crate::engine::VOCAB)); }
                    if let Some(f) = u.emit(a) { break 'g f; }
                    pos += 1; cur = a;
                }
            }
            if m.has_step4() {
                // the 4-token step (quad.rs, default; MECHTA_STEP4=0 for the double step): the sure token + 3 MTP drafts (1 from the rows, 2 in a
                // chain) checked in one pass. Exact acceptance: row i is sampled (argmax when greedy) and the draft
                // counts only if it is that token; the first refused draft ends the step.
                let mut ds = [m.draft(y, start), 0, 0];
                let ch = m.draft_chain(ds[0], start + 1, 0, 2); ds[1] = ch[0]; ds[2] = ch[1];
                let (mut pos, mut cur) = (start, y);
                // one submit per cycle (cycle4: drafts and step together) when the reduced draft head is there
                let fused = m.has_cycle4();
                if pos + crate::engine::STEP4_ROWS + 2 >= ctx() { break 'g Finish::Length; }
                let tc = Instant::now();
                let mut a = m.step4([cur, ds[0], ds[1], ds[2]], pos);
                let mut t_step = tc.elapsed().as_secs_f64() * 1e3;
                let f = 'q: loop {
                    let ts = Instant::now();
                    if !camp.greedy() {
                        let l = m.read("logits", crate::engine::STEP4_ROWS * crate::engine::VOCAB);
                        for i in 0..crate::engine::STEP4_ROWS {
                            a[i] = camp.sample(&l[i * crate::engine::VOCAB..(i + 1) * crate::engine::VOCAB]);
                            if i == 3 || a[i] != ds[i] { break; }
                        }
                    }
                    let t_sample = ts.elapsed().as_secs_f64() * 1e3;
                    let mut na = 0; while na < 3 && a[na] == ds[na] { na += 1; }
                    m.accept4(na + 1);
                    cycles += 3; accepted += na; steps4 += 1;
                    fed = pos + 1 + na;
                    for &t in &ds[..na] { if let Some(f) = u.emit(t) { break 'q f; } }
                    if let Some(f) = u.emit(a[na]) { break 'q f; }
                    w_double += t_step; w_sample += t_sample; w_cycles += 1;
                    let toks: Vec<u32> = ds[..na].iter().copied().chain([a[na]]).collect();
                    pos += na + 1; cur = a[na];
                    if pos + crate::engine::STEP4_ROWS + 2 >= ctx() { break 'q Finish::Length; }
                    let tc = Instant::now();
                    if fused {
                        let (a2, d2) = m.cycle4(&toks, pos - na);
                        a = a2; ds = d2;
                    } else {
                        ds[0] = m.draft_rows(&toks, pos - na);
                        let ch = m.draft_chain(ds[0], pos + 1, toks.len() - 1, 2); ds[1] = ch[0]; ds[2] = ch[1];
                        w_draft += tc.elapsed().as_secs_f64() * 1e3;
                        let tq = Instant::now();
                        a = m.step4([cur, ds[0], ds[1], ds[2]], pos);
                        t_step = tq.elapsed().as_secs_f64() * 1e3;
                        continue;
                    }
                    t_step = tc.elapsed().as_secs_f64() * 1e3;
                };
                m.flush4();
                break 'g f;
            }
            let mut d = m.draft(y, start);
            let (mut pos, mut cur) = (start, y);
            loop {
                if pos + 3 >= ctx() { break 'g Finish::Length; }
                let prof_now = gen_profile && cycles % 8 == 4;
                if prof_now { *m.prof.borrow_mut() = Some(crate::engine::Profile::default()); profiled += 1; }
                let tc = Instant::now();
                let (mut a0, mut a1) = m.step2(cur, d, pos);
                let t_double = tc.elapsed().as_secs_f64() * 1e3;
                if !camp.greedy() {
                    // logits of the two tokens of the double step (before draft/draft2, which overwrite "logits")
                    let l = m.read("logits", 2 * crate::engine::VOCAB);
                    if greedy_accept && a0 == d { a1 = camp.sample(&l[crate::engine::VOCAB..]); }
                    else {
                        a0 = camp.sample(&l[..crate::engine::VOCAB]);
                        if a0 == d { a1 = camp.sample(&l[crate::engine::VOCAB..]); }
                    }
                }
                let t_sample = tc.elapsed().as_secs_f64() * 1e3 - t_double;
                cycles += 1; if a0 == d { accepted += 1; }
                fed = if a0 == d { pos + 2 } else { pos + 1 };
                if a0 == d {
                    if let Some(f) = u.emit(d) { break 'g f; }
                    if let Some(f) = u.emit(a1) { break 'g f; }
                    let td = Instant::now();
                    d = m.draft2(d, a1, pos); row1 = true;
                    if !prof_now { w_draft += td.elapsed().as_secs_f64() * 1e3; }
                    pos += 2; cur = a1;
                } else {
                    let tr = Instant::now();
                    m.restore();
                    if !prof_now { w_restore += tr.elapsed().as_secs_f64() * 1e3; }
                    if let Some(f) = u.emit(a0) { break 'g f; }
                    let td = Instant::now();
                    d = m.draft(a0, pos + 1); row1 = false;
                    if !prof_now { w_draft += td.elapsed().as_secs_f64() * 1e3; }
                    pos += 1; cur = a0;
                }
                if let Some(f) = probe.as_mut() {
                    use std::io::Write;
                    let chain = m.draft_chain(d, pos + 1, row1 as usize, 2);
                    let mut h = toks.clone(); h.extend_from_slice(&u.emitted);
                    let (len, ng) = ngram_next(&h, 3);
                    let _ = writeln!(f, "c {pos} {} {cur} {d} {} {} {len} {}", h.len(), chain[0], chain[1],
                                     ng.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
                }
                if !prof_now { w_double += t_double; w_sample += t_sample; w_cycles += 1; }
                else { let p = m.prof.borrow_mut().take(); *m.prof.borrow_mut() = None; acc_prof(&mut prof_sum, p); }
            }
        };
        if !u.cut {
            if let Some(f) = probe.as_mut() {
                use std::io::Write;
                let _ = writeln!(f, "g {n} {}", u.emitted.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(","));
                let _ = f.flush();
            }
            break 'all f;
        }
        // Thinking budget reached: the closing text goes to the client and into the model (with the emitted
        // token not processed yet, if any), one step at a time like the tail of the prefill; then generation goes on.
        u.cut = false; u.budget = 0; budget_cuts += 1;
        let mut feed: Vec<u32> = u.emitted[fed - n..].to_vec();
        for &t in &close { if let Some(f) = u.emit(t) { break 'all f; } }
        feed.extend_from_slice(&close);
        if fed + feed.len() + 4 >= ctx() { break 'all Finish::Length; }
        let mut last = 0;
        for (j, &t) in feed.iter().enumerate() {
            last = m.step(t, fed + j);
            if j + 1 < feed.len() { m.draft(feed[j + 1], fed + j + 1); }
        }
        start = fed + feed.len();
        y = if camp.greedy() { last } else { camp.sample(&m.read("logits", crate::engine::VOCAB)) };
        eprintln!("thinking budget {} reached at position {fed}: reasoning closed ({} tokens fed)", p.reasoning_budget, feed.len());
        };
        if budget_cuts == 0 && thinking && p.reasoning_budget > 0 { eprintln!("thinking budget {}: not reached ({} reasoning tokens)", p.reasoning_budget, u.thought); }
        if w_cycles > 0 {
            let c = w_cycles as f64;
            eprintln!("gen time per cycle ({w_cycles} cycles): double step {:.1} ms · logits+sampling {:.1} ms · draft {:.1} ms · restore {:.1} ms",
                w_double / c, w_sample / c, w_draft / c, w_restore / c);
        }
        if profiled > 0 { print_prof(&prof_sum, profiled); }
        if cycles > 0 {
            eprintln!("mtp: {accepted}/{cycles} drafts accepted ({:.0}%) · accept={} · kv={}", 100.0 * accepted as f64 / cycles as f64,
                if greedy_accept { "greedy" } else { "exact" }, crate::engine::kv_format().name());
        }
        if steps4 > 0 { eprintln!("step4: {steps4} steps · {:.2} tokens per step", (steps4 + accepted) as f64 / steps4 as f64); }
        if let Some(t) = trace::summary(trace_from) { eprintln!("{t}"); }
        trace::flush();
        // unprinted remainder (only if the generation ends in the middle of a character)
        if u.k0 < u.emitted.len() && end != Finish::Cancelled {
            let s = u.tk.decode(&u.emitted[u.k0..], false).unwrap_or_default();
            if !s.is_empty() { (u.piece)(&s); }
        }
        Ok(Outcome { token_prompt: n, reused_tokens: done, generated_tokens: u.emitted.len(), end,
                   ms_prefill, ms_generation: t1.elapsed().as_secs_f64() * 1e3 })
    }
}

/// `mechta test-session <prompt.txt> [max_token]`: two requests on the same Session (the second extends the first with
/// a fake answer and a new user message: it must reuse the prefix), then the second again WITHOUT reuse: the
/// text must be the same.
pub fn test_session(file: &str, max: usize) {
    let t = Instant::now();
    let mut s = Session::new();
    println!("loaded in {:.1} s · max context {}", t.elapsed().as_secs_f64(), s.max_context());
    let p1 = std::fs::read_to_string(file).expect("prompt");
    // tokenizer check: the special tokens must be single tokens
    let ids = s.tokenize("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n");
    println!("special tokens: {:?} (expected {IM_START} at the start, {IM_END} after 'hello')", ids);
    let trim = p1.rfind("<|im_start|>assistant").expect("the prompt must end with <|im_start|>assistant");
    let p2 = format!("{}<|im_start|>assistant\nFake answer: I have read the file.<|im_end|>\n<|im_start|>user\nNow summarize the file in one sentence.<|im_end|>\n<|im_start|>assistant\n<think>\n", &p1[..trim]);
    let par = Params { max_token: max, ..Default::default() };
    let run_one = |s: &mut Session, name: &str, p: &str| -> String {
        let mut text = String::new();
        let e = s.generate(p, &par, &mut |x| { text.push_str(x); true }).expect("generate");
        println!("--- {name}: prompt {} tokens, reused {}, generated {}, end {:?} · prefill {:.0} ms ({:.2} ms/new token) · generation {:.0} ms = {:.1} token/s",
            e.token_prompt, e.reused_tokens, e.generated_tokens, e.end, e.ms_prefill, e.ms_prefill / (e.token_prompt - e.reused_tokens).max(1) as f64,
            e.ms_generation, e.generated_tokens as f64 / (e.ms_generation / 1e3));
        println!("{name} text: {:?}", text.chars().take(400).collect::<String>());
        text
    };
    run_one(&mut s, "request 1", &p1);
    let a = run_one(&mut s, "request 2 (reuse)", &p2);
    s.forget();
    let b = run_one(&mut s, "request 2 (from scratch)", &p2);
    let common = a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count();
    println!("request 2 with and without reuse: {}", if a == b { "SAME TEXT".to_string() } else {
        format!("text differs from char {common} of {} · with reuse: {:?} · from scratch: {:?}", a.chars().count(),
            a.chars().skip(common.saturating_sub(40)).take(80).collect::<String>(), b.chars().skip(common.saturating_sub(40)).take(80).collect::<String>()) });
}

/// `mechta test-budget <budget> [max_token]`: checks the thinking budget at temperature 0.
/// 1. One request with the budget: the reasoning must be cut and closed with `CLOSE_THINKING`.
/// 2. The same prompt plus the output up to the end of the closing text, without budget and without reuse (all
///    prefilled from zero): the answer must be the same text as in 1. This proves that the model state after the cut
///    (the closing text fed one token at a time after an MTP cycle) equals the state of a normal prefill.
pub fn test_budget(budget: usize, max: usize) {
    let mut s = Session::new();
    let prompt = "<|im_start|>user\nA farmer has 17 sheep. All but 9 run away, then he buys twice as many as remain and sells 5. \
Then a storm takes one third of the flock, rounded down. How many sheep are left? Check every step twice.<|im_end|>\n<|im_start|>assistant\n<think>\n";
    let run = |s: &mut Session, p: &str, budget: usize| -> (String, Outcome) {
        let par = Params { max_token: max, temperature: 0.0, reasoning_budget: budget, ..Default::default() };
        let mut out = String::new();
        let e = s.generate(p, &par, &mut |x| { out.push_str(x); true }).expect("generate");
        (out, e)
    };
    if budget == 0 {
        // control (no budget): the same comparison with a plain cut of the output at a line end, to see how far a
        // continuation after a full prefill follows the generated text when nothing is injected
        let (o1, e1) = run(&mut s, prompt, 0);
        println!("--- control 1: generated {} tokens, end {:?}", e1.generated_tokens, e1.end);
        let cut = o1.char_indices().map(|(i, _)| i).filter(|&i| o1[i..].starts_with('\n')).find(|&i| s.tokenize(&o1[..i]).len() >= 280).map(|i| i + 1).unwrap_or(o1.len());
        s.forget();
        let (o2, e2) = run(&mut s, &format!("{prompt}{}", &o1[..cut]), 0);
        println!("--- control 2: prompt {} tokens, generated {}", e2.token_prompt, e2.generated_tokens);
        let a = &o1[cut..];
        let same = a.chars().zip(o2.chars()).take_while(|(x, y)| x == y).count();
        if o2.starts_with(a) || a.starts_with(&o2) { println!("RESULT control: SAME TEXT ({} chars compared)", a.chars().count().min(o2.chars().count())); }
        else { println!("RESULT control: differ from char {same}: generated {:?} · full prefill {:?}", a.chars().skip(same.saturating_sub(30)).take(80).collect::<String>(), o2.chars().skip(same.saturating_sub(30)).take(80).collect::<String>()); }
        return;
    }
    let (o1, e1) = run(&mut s, prompt, budget);
    println!("--- 1. budget {budget}: generated {} tokens, end {:?}", e1.generated_tokens, e1.end);
    let Some(k) = o1.find(CLOSE_THINKING) else { println!("RESULT: the reasoning was not cut (closing text not found)"); return; };
    let cut = k + CLOSE_THINKING.len();
    let thought = s.tokenize(&o1[..k]).len();
    println!("reasoning before the cut: {thought} tokens (budget {budget})");
    println!("end of the reasoning: {:?}", o1[..k].chars().rev().take(200).collect::<String>().chars().rev().collect::<String>());
    println!("answer: {:?}", o1[cut..].chars().take(600).collect::<String>());
    s.forget();
    let p2 = format!("{prompt}{}", &o1[..cut]);
    let (o2, e2) = run(&mut s, &p2, 0);
    println!("--- 2. same text from a full prefill: prompt {} tokens, reused {}, generated {}", e2.token_prompt, e2.reused_tokens, e2.generated_tokens);
    let a = &o1[cut..];
    let same = a.chars().zip(o2.chars()).take_while(|(x, y)| x == y).count();
    if a == o2 { println!("RESULT: SAME ANSWER ({} chars)", a.chars().count()); }
    else { println!("RESULT: answers differ from char {same} of {} / {}: after the cut {:?} · full prefill {:?}", a.chars().count(), o2.chars().count(),
        a.chars().skip(same.saturating_sub(30)).take(80).collect::<String>(), o2.chars().skip(same.saturating_sub(30)).take(80).collect::<String>()); }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ngram_drafts() {
        // suffix [3,4,5,6] seen before, followed by 7,8,9
        assert_eq!(ngram_next(&[1, 3, 4, 5, 6, 7, 8, 9, 2, 3, 4, 5, 6], 3), (4, vec![7, 8, 9]));
        // only [5,6] repeats: the most recent earlier match wins
        assert_eq!(ngram_next(&[5, 6, 1, 0, 5, 6, 2, 9, 5, 6], 3), (2, vec![2, 9, 5]));
        // match near the end: fewer than k tokens follow
        assert_eq!(ngram_next(&[7, 7, 7], 3), (2, vec![7]));
        assert_eq!(ngram_next(&[1, 2, 3], 3), (0, vec![]));
    }
    #[test]
    fn model_ids() {
        assert_eq!(model_id("/x/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf"), "qwen3.6-35b-a3b");
        assert_eq!(model_id("/x/Flint-Q4_K_M.gguf"), "flint-q4_k_m");
        assert_eq!(model_id("/x/Tiel-Coder-35B-A3B-MTP-UD-Q4_K_S.gguf"), "tiel-coder-35b-a3b");
        assert_eq!(model_id("/x/Ornith-1.5-35B-A3B-APEX-MTP-Compact.gguf"), "ornith-1.5-35b-a3b-apex");
        assert_eq!(model_id("/x/Other-Model-Q4.gguf"), "other-model-q4");
    }
    /// The sampler frequencies must follow softmax(logit / t) restricted to top_k and top_p.
    #[test]
    fn frequencies() {
        let logit = vec![2.0f32, 1.0, 0.5, 0.0, -1.0, -3.0];
        let par = Params { temperature: 0.7, top_k: 4, top_p: 0.95, seed: 42, ..Default::default() };
        let mut c = Sampler::new(&par);
        let mut n = [0usize; 6]; let rounds = 200_000;
        for _ in 0..rounds { n[c.sample(&logit) as usize] += 1; }
        // expected: softmax over the best 4 with t = 0.7, then cut at 0.95 cumulative
        let e: Vec<f64> = logit[..4].iter().map(|&x| ((x as f64 - 2.0) / 0.7).exp()).collect();
        let z: f64 = e.iter().sum(); let pr: Vec<f64> = e.iter().map(|x| x / z).collect();
        let mut cum = 0.0; let mut kept = 0; for q in &pr { cum += q; kept += 1; if cum >= 0.95 { break; } }
        let z2: f64 = pr[..kept].iter().sum();
        for i in 0..6 {
            let expected = if i < kept { pr[i] / z2 } else { 0.0 };
            let seen = n[i] as f64 / rounds as f64;
            println!("token {i}: expected {expected:.4} seen {seen:.4}");
            assert!((expected - seen).abs() < 0.005, "token {i}: expected {expected} seen {seen}");
        }
    }
}

/// MECHTA_GEN_PROFILE: sums the per-kernel GPU times of the profiled cycles.
fn acc_prof(sum: &mut std::collections::HashMap<String, (f64, u32)>, p: Option<crate::engine::Profile>) {
    if let Some(p) = p { for (k, (ms, _, n)) in p.gpu { let e = sum.entry(k).or_default(); e.0 += ms; e.1 += n; } }
}
/// Prints the GPU time per profiled cycle by kernel (largest first) and by group (attention / experts / dense / other).
fn print_prof(sum: &std::collections::HashMap<String, (f64, u32)>, cycles: usize) {
    let c = cycles as f64;
    let mut v: Vec<_> = sum.iter().collect();
    v.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
    let tot: f64 = v.iter().map(|x| x.1.0).sum::<f64>() / c;
    let mut groups: std::collections::BTreeMap<&str, f64> = std::collections::BTreeMap::new();
    for (k, (ms, _)) in &v {
        let g = if k.contains("attn") { "attention" } else if k.contains("exps") || k.contains("experts") || k.contains("merge") { "experts" }
            else if k.contains("delta") || k.contains("conv") { "deltanet" } else if k.contains("mv") || k.contains("blk.") || k.contains("output") { "dense" } else { "other" };
        *groups.entry(g).or_default() += ms / c;
    }
    eprintln!("gen profile ({cycles} cycles, GPU ms per cycle, one pass per dispatch): total {tot:.1} · {}",
        groups.iter().map(|(g, ms)| format!("{g} {ms:.1}")).collect::<Vec<_>>().join(" · "));
    for (k, (ms, n)) in v.iter().take(25) { eprintln!("  {:48} {:7.2} ms  ({} dispatches)", k, ms / c, *n as usize / cycles); }
}
