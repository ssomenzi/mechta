//! F5 of PLAN-FLINT.md: Flint behind the trait `mechta_api::Engine`, for the HTTP server.
//!
//! A request = a prompt already formatted by Flint's template (crate chat, `Family::Flint`). Steps:
//! 1. reuse: Flint has no recurrent state, only the K/V caches. The full layers keep every position; the sliding
//!    layers keep a ring of RING_SWA rows. The common prefix k with the tokens already in the cache is reused as is
//!    if the ring still holds the 512 positions before k (at most 512 positions written after k). Otherwise the photo
//!    of the rings (taken at the end of the prompt of the request before, where its answer starts) if the prompt
//!    starts with its tokens; otherwise the root (system prompt + tools, saved on disk); otherwise from zero.
//! 2. prefill in blocks up to the last token, which goes alone (its logits start the generation).
//! 3. generation one token at a time (Flint has no MTP), stop on the four end tokens of Flint.
//! The model decides by itself whether to reason (no switch, no budget: `reasoning_budget` is ignored).
use crate::flint::{Flint, FlintRoot, WINDOW, RING_SWA};
use crate::session::Sampler;
use mechta_api::{Outcome, Finish, Engine, Params};
use std::time::Instant;

const USER: u32 = 150_003;
const DEVELOPER: u32 = 150_018;
/// <|end_of_text|>, <|user|>, <|observation|>, <|developer|> (eog_token_ids of the GGUF)
const END: [u32; 4] = [150_001, 150_003, 150_005, 150_018];
/// a root is taken only if the system part is at least this long (as the Qwen session)
const ROOT_MIN: usize = 2048;
/// Folder of the saved roots: `MECHTA_CACHE`.
fn root_dir() -> String { crate::cache_dir() }

pub struct FlintSession {
    f: Flint,
    tk: tokenizers::Tokenizer,
    /// tokens whose K/V are in the caches now, position by position (the last generated token is not: never fed)
    in_cache: Vec<u32>,
    /// tokens covered by the photo of the rings (empty = no photo)
    photo_tok: Vec<u32>,
    root: Option<(Vec<u32>, FlintRoot)>,
    use_root: bool,
    name: String,
}

/// Same key as the Qwen roots (FNV-1a 64 of the tokens); the file name carries "flint" and the model path key.
fn root_file(tok: &[u32]) -> String {
    let h = |it: &mut dyn Iterator<Item = u8>| { let mut h: u64 = 0xcbf29ce484222325; for b in it { h ^= b as u64; h = h.wrapping_mul(0x100000001b3); } h };
    let m = h(&mut crate::model_path().into_bytes().into_iter());
    let t = h(&mut tok.iter().flat_map(|x| x.to_le_bytes()));
    format!("{}/root-flint-{}-{m:016x}-{t:016x}.bin", root_dir(), if crate::flint::kv_f16() { "f16" } else { "f32" })
}

fn write_root_file(tok: &[u32], r: &FlintRoot) -> std::io::Result<()> {
    use std::io::Write;
    let file = root_file(tok);
    std::fs::create_dir_all(root_dir())?;
    let tmp = format!("{file}.tmp");
    let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
    f.write_all(b"MECHTA-ROOT-FLINT-1\n")?;
    f.write_all(&(r.p as u64).to_le_bytes())?; f.write_all(&(r.kv.len() as u64).to_le_bytes())?;
    for t in tok { f.write_all(&t.to_le_bytes())?; }
    for (k, v) in &r.kv {
        for x in [k, v] { f.write_all(&(x.len() as u64).to_le_bytes())?; for y in x { f.write_all(&y.to_le_bytes())?; } }
    }
    f.flush()?; drop(f);
    std::fs::rename(&tmp, &file)?;
    Ok(())
}

fn read_root_file(path: &str) -> Option<(Vec<u32>, FlintRoot)> {
    let b = std::fs::read(path).ok()?;
    let header = b"MECHTA-ROOT-FLINT-1\n";
    if !b.starts_with(header) { return None; }
    let mut o = header.len();
    let u = |o: &mut usize| -> Option<u64> { let x = u64::from_le_bytes(b.get(*o..*o + 8)?.try_into().ok()?); *o += 8; Some(x) };
    let p = u(&mut o)? as usize; let nl = u(&mut o)? as usize;
    let tok: Vec<u32> = b.get(o..o + 4 * p)?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect(); o += 4 * p;
    let mut kv = vec![];
    for _ in 0..nl {
        let mut due = vec![];
        for _ in 0..2 {
            let n = u(&mut o)? as usize;
            due.push(b.get(o..o + 4 * n)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f32>>()); o += 4 * n;
        }
        let v = due.pop()?; let k = due.pop()?; kv.push((k, v));
    }
    Some((tok, FlintRoot { p, kv }))
}

/// End of the system part (tools included): the first `<|user|>` or `<|developer|>` after the start.
fn root_point(toks: &[u32]) -> Option<usize> { toks.iter().skip(1).position(|t| *t == USER || *t == DEVELOPER).map(|q| q + 1) }

impl FlintSession {
    pub fn new() -> Self {
        let f = Flint::new(&crate::model_path(), crate::engine::ctx());
        let tk = tokenizers::Tokenizer::from_file(crate::tokenizer_path()).expect("Flint tokenizer");
        let use_root = std::env::var("MECHTA_ROOT").map(|v| v != "no").unwrap_or(true);
        FlintSession { f, tk, in_cache: vec![], photo_tok: vec![], root: None, use_root, name: crate::session::model_id(&crate::model_path()) }
    }
    pub fn tokenize(&self, s: &str) -> Vec<u32> { self.tk.encode(s, false).expect("encode").get_ids().to_vec() }
    /// Like `tokenize`, but a tokenizer error goes back to the caller (this is the path of the
    /// HTTP requests, where a panic would make the server exit).
    pub fn tokenize_ok(&self, s: &str) -> Result<Vec<u32>, String> { Ok(self.tk.encode(s, false).map_err(|e| e.to_string())?.get_ids().to_vec()) }

    /// Feeds toks[da..a] at their positions: prefill in blocks, then the cache covers toks[..a]. `alive` is asked between
    /// submits; if it says no, the prefill stops, the cache covers the blocks completed, and the result is false.
    fn feed(&mut self, toks: &[u32], da: usize, a: usize, alive: &mut dyn FnMut() -> bool) -> bool {
        let fed = if a > da { self.f.prefill_cancellable(&toks[da..a], da, alive) } else { 0 };
        self.in_cache.truncate(da);
        self.in_cache.extend_from_slice(&toks[da..da + fed]);
        fed == a - da
    }
}

impl Default for FlintSession { fn default() -> Self { Self::new() } }

impl Engine for FlintSession {
    fn model_name(&self) -> &str { &self.name }
    fn max_context(&self) -> usize { self.f.ctx }
    fn count_tokens(&self, text: &str) -> Result<usize, String> { Ok(self.tokenize_ok(text)?.len()) }

    fn generate(&mut self, prompt: &str, p: &Params, piece: &mut dyn FnMut(&str) -> bool) -> Result<Outcome, String> {
        let t0 = Instant::now();
        let toks = self.tokenize_ok(prompt)?;
        let n = toks.len();
        let ctx = self.f.ctx;
        if n == 0 { return Err("empty prompt".into()); }
        if n + 4 >= ctx { return Err(format!("prompt of {n} tokens: beyond the maximum context of {ctx}")); }
        // 1. reuse
        let m = self.in_cache.len();
        let k = self.in_cache.iter().zip(&toks).take_while(|(a, b)| a == b).count().min(n - 1);
        // the 512 positions before k still in the rings. MECHTA_FLINT_REUSE=photo (tests only): never, so the photo is used
        let ring_ok = k > 0 && m - k + WINDOW <= RING_SWA + 1 && std::env::var("MECHTA_FLINT_REUSE").map(|v| v != "photo").unwrap_or(true);
        let kf = self.photo_tok.len();
        let photo_ok = kf > 0 && kf <= k && toks[..kf] == self.photo_tok[..];
        let q = if self.use_root { root_point(&toks).filter(|&q| q >= ROOT_MIN && q < n) } else { None };
        if let Some(q) = q {
            let in_memory = matches!(&self.root, Some((rt, _)) if rt[..] == toks[..q]);
            let path = root_file(&toks[..q]);
            if !in_memory && std::path::Path::new(&path).exists() {
                let tl = Instant::now();
                if let Some(r) = read_root_file(&path) {
                    eprintln!("root loaded from {path}: {} tokens in {:.0} ms", r.0.len(), tl.elapsed().as_secs_f64() * 1e3);
                    self.root = Some(r);
                }
            }
        }
        let kr = match &self.root { Some((rt, _)) if rt.len() < n && toks[..rt.len()] == rt[..] => rt.len(), _ => 0 };
        let done = if ring_ok && k >= kf.max(kr) { k }
            else if photo_ok && kf >= kr { self.f.restore_photo(); kf }
            else if kr > 0 { self.f.write_root(&self.root.as_ref().unwrap().1); kr }
            else { 0 };
        self.in_cache.truncate(done);
        let ms_rip = t0.elapsed().as_secs_f64() * 1e3;
        // 2. prefill: stops at the new root (saved), then the last token alone; photo at the end of the prompt
        let r = q.filter(|&q| q > done && kr != q);
        let mut stages: Vec<usize> = r.into_iter().collect();
        stages.push(n - 1); stages.sort_unstable(); stages.dedup();
        let mut i = done;
        for t in stages {
            // an empty piece is a probe: it fails when the client has gone (the server drops the channel)
            if !self.feed(&toks, i, t, &mut || piece("")) {
                eprintln!("prefill stopped at token {} of {n}: the client has gone", self.in_cache.len());
                return Ok(Outcome { token_prompt: n, reused_tokens: done, generated_tokens: 0, end: Finish::Cancelled,
                                  ms_prefill: t0.elapsed().as_secs_f64() * 1e3, ms_generation: 0.0 });
            }
            i = t;
            if r == Some(t) {
                let tr = Instant::now();
                let new_root = self.f.read_root(t);
                let tok_r = toks[..t].to_vec();
                eprintln!("new root: {t} tokens, read in {:.0} ms", tr.elapsed().as_secs_f64() * 1e3);
                let (t2, r2) = (tok_r.clone(), FlintRoot { p: new_root.p, kv: new_root.kv.clone() });
                std::thread::spawn(move || match write_root_file(&t2, &r2) { Ok(()) => eprintln!("root saved in {}", root_file(&t2)), Err(e) => eprintln!("root NOT saved: {e}") });
                self.root = Some((tok_r, new_root));
            }
        }
        let mut lg = self.f.token(toks[n - 1], n - 1);
        self.in_cache.push(toks[n - 1]);
        self.f.photograph(); self.photo_tok = toks.clone();
        let ms_prefill = t0.elapsed().as_secs_f64() * 1e3;
        eprintln!("prefill: reused {done} of {n} ({}) · tokenize+restore {ms_rip:.0} ms · total {ms_prefill:.0} ms · longest submit {:.0} ms",
                  if done == 0 { "none" } else if done == k && ring_ok { "common prefix" } else if done == kf && photo_ok { "photo" } else { "root" }, self.f.submit_max.get());
        if p.reasoning_budget > 0 { eprintln!("thinking budget {} ignored: Flint has no switch for its reasoning", p.reasoning_budget); }
        // 3. generation
        let t1 = Instant::now();
        let mut camp = Sampler::new(p);
        let (mut emitted, mut k0) = (Vec::<u32>::new(), 0usize);
        let max = p.max_token.max(1);
        let mut pos = n;
        let end = loop {
            let y = if camp.greedy() { lg.iter().enumerate().fold((0usize, f32::NEG_INFINITY), |m, (i, v)| if *v > m.1 { (i, *v) } else { m }).0 as u32 } else { camp.sample(&lg) };
            if END.contains(&y) { break Finish::Stop; }
            emitted.push(y);
            // incremental decode: wait for the next token if the text ends in the middle of a character
            let s = self.tk.decode(&emitted[k0..], false).unwrap_or_default();
            if !s.ends_with('\u{FFFD}') {
                k0 = emitted.len();
                if !s.is_empty() && !piece(&s) { break Finish::Cancelled; }
            }
            if emitted.len() >= max || pos + 1 >= ctx { break Finish::Length; }
            lg = self.f.token(y, pos);
            self.in_cache.push(y);
            pos += 1;
            // progress of a long generation (the server logs a request only when it ends)
            if emitted.len() % 500 == 0 {
                let s = t1.elapsed().as_secs_f64();
                eprintln!("generating: {} tokens in {s:.0} s ({:.1} t/s), context {pos}", emitted.len(), emitted.len() as f64 / s);
            }
        };
        if k0 < emitted.len() && end != Finish::Cancelled {
            let s = self.tk.decode(&emitted[k0..], false).unwrap_or_default();
            if !s.is_empty() { piece(&s); }
        }
        Ok(Outcome { token_prompt: n, reused_tokens: done, generated_tokens: emitted.len(), end, ms_prefill, ms_generation: t1.elapsed().as_secs_f64() * 1e3 })
    }
}
