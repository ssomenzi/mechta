//! mechta — inference engine for Qwen3.6-35B-A3B. For now: GGUF inspection and tokenizer.
#![forbid(unsafe_code)]
use mechta_gguf::Gguf;
use std::collections::BTreeMap;
mod engine;
mod tools;
mod memcheck;
mod compare;
mod attn_dump;
mod dmx;
mod wgsl_check;
pub mod session;
pub mod flint;
pub mod session_flint;

/// No path is written in the code any more. Three variables are required:
/// `MECHTA_MODEL` (the GGUF), `MECHTA_TOKENIZER` (its tokenizer.json), `MECHTA_CACHE` (folder of the saved roots).
/// A missing one stops the program with a message that names it.
fn required_var(name: &str, what: &str) -> String {
    match std::env::var(name) {
        Ok(v) if !v.is_empty() => v,
        _ => { eprintln!("missing environment variable {name}: {what}"); std::process::exit(2) }
    }
}
/// The model file of this run (qwen35moe: 41 layers with the MTP one, quant types Q8_0, Q3_K, Q4_K, Q6_K, F32; or Flint).
pub fn model_path() -> String { required_var("MECHTA_MODEL", "the .gguf file of the model") }
/// The tokenizer.json of the model in use (Qwen3.6 or Flint).
pub fn tokenizer_path() -> String { required_var("MECHTA_TOKENIZER", "the tokenizer.json of the model") }
/// Folder of the saved roots (prefix cache between sessions); created if missing.
pub fn cache_dir() -> String { required_var("MECHTA_CACHE", "the folder of the prefix cache (saved roots)") }
/// general.architecture of a GGUF ("qwen35moe", "flint", ...).
pub fn architecture(path: &str) -> String {
    Gguf::open(path).ok().and_then(|g| g.meta.get("general.architecture").and_then(|v| v.as_str()).map(String::from)).unwrap_or_default()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("inspect") => inspect(&args.get(2).cloned().unwrap_or_else(model_path)),
        Some("token") => token(&args[2], &args[3]),
        Some("flint-logit-dump") => flint::logit_dump(&args[2], args[3].parse().expect("prefill"), args[4].parse().expect("steps"), &args[5]),
        Some("logit-dump") => tools::logit_dump(&args[2], args[3].parse().unwrap(), args[4].parse().unwrap(), &args[5], args.get(6).is_some_and(|m| m == "double")),
        Some("kl") => compare::kl(&args[2], &args[3], &args[4..].iter().map(|a| compare::parse_limit(a).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) })).collect::<Vec<_>>()),
        // OpenAI-compatible server: 127.0.0.1:4131, thinking on; MECHTA_ADDRESS / MECHTA_THINKING=no
        // `serve [--ctx N]`: context size at start (or MECHTA_CTX; default 65536). Before loading the model
        // the memory check runs: if the context does not fit, the server does not start and says the largest that fits.
        Some("serve") => {
            if let Some(i) = args.iter().position(|a| a == "--ctx") {
                let v = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(0);
                if let Err(e) = engine::check_ctx(v) { eprintln!("--ctx: {e}"); std::process::exit(2); }
                engine::set_ctx(v);
            }
            let (ctx, kv) = (engine::ctx(), engine::kv_format());
            // a Flint GGUF gets the Flint session and Flint's chat format (F5 of PLAN-FLINT.md)
            let flint = architecture(&model_path()) == "flint";
            match if flint { memcheck::check_flint(ctx, &model_path()) } else { memcheck::check(ctx, kv) } {
                Ok(m) => eprintln!("{m}"),
                Err(m) => { eprintln!("{m}"); std::process::exit(1); }
            }
            eprintln!("model: {}", model_path());
            let before = memcheck::used_now();
            let mut cfg = mechta_server::Config::default();
            if flint { cfg.family = mechta_server::Family::Flint; }
            if let Ok(a) = std::env::var("MECHTA_ADDRESS") { cfg.address = a; }
            cfg.api_key = std::env::var("MECHTA_API_KEY").ok().filter(|k| !k.is_empty());
            if std::env::var("MECHTA_THINKING").is_ok_and(|v| v == "no") { cfg.thinking = false; }
            // default thinking budget in tokens (0 or unset = no limit)
            if let Some(n) = std::env::var("MECHTA_THINK_BUDGET").ok().and_then(|v| v.parse().ok()) { cfg.reasoning_budget = n; }
            mechta_server::start(Box::new(move || {
                let s: Box<dyn mechta_api::Engine> = if flint { Box::new(session_flint::FlintSession::new()) } else { Box::new(session::Session::new()) };
                // measured use after loading, to correct the estimate of memcheck (BASE_BYTES)
                if let (Some(b), Some(a)) = (before, memcheck::used_now()) {
                    let estimate = if flint { memcheck::need_bytes_flint(ctx, &model_path()) } else { memcheck::need_bytes(ctx, kv) };
                    eprintln!("context {ctx}, KV {}: GPU memory taken {:.2} GB (estimate {:.2} GB)", if flint { if flint::kv_f16() { "f16" } else { "f32" } } else { kv.name() }, a.saturating_sub(b) as f64 / 1e9, estimate as f64 / 1e9);
                }
                s
            }), cfg)
        }
        // `check-memory [ctx]`: the memory check alone, without the GPU (mechta-serve.sh runs it before every start)
        Some("check-memory") => {
            let ctx = match args.get(2) { Some(v) => engine::check_ctx(v.parse().unwrap_or(0)).unwrap_or_else(|e| { eprintln!("{e}"); std::process::exit(2) }), None => engine::ctx() };
            let r = if architecture(&model_path()) == "flint" { memcheck::check_flint(ctx, &model_path()) } else { memcheck::check(ctx, engine::kv_format()) };
            match r {
                Ok(m) => println!("{m}"),
                Err(m) => { println!("{m}"); std::process::exit(1); }
            }
        }
        Some("dmx-bench") => dmx::dmx_bench(&args[2], &args[3]),
        Some("quad-check") => engine::quad_check(&args[2], args[3].parse().expect("prefill"), args[4].parse().expect("cycles")),
        Some("dmx-ab") => dmx::dmx_ab(&args[2], &args[3], args[4].parse().expect("steps"), args[5].parse().expect("rounds")),
        Some("sparse-check") => attn_dump::sparse_check(&args[2], &args[3], args[4].parse().expect("steps"), &args[5]),
        Some("sparse-recall") => attn_dump::sparse_recall(&args[2], args[3].parse().expect("tokens"), &args[4]),
        Some("attn-dump") => attn_dump::attn_dump(&args[2], &args[3], args[4].parse().expect("steps"), &args[5]),
        Some("test-budget") => session::test_budget(args.get(2).and_then(|v| v.parse().ok()).unwrap_or(256), args.get(3).and_then(|v| v.parse().ok()).unwrap_or(1500)),
        Some("deq") => deq(&args[2], args[3].parse().expect("N"), &args[4]),
        _ => eprintln!("usage: mechta serve [--ctx N] | check-memory [ctx] | logit-dump | kl | flint-logit-dump | test-budget | inspect [file.gguf] | token <text.txt> <llamacpp_ids.bin> | deq <tensor> <N> <reference.f32>"),
    }
}


fn inspect(path: &str) {
    let g = match Gguf::open(path) { Ok(g) => g, Err(e) => { eprintln!("{path}: {e}"); std::process::exit(1) } };
    let mut per_kind: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    let mut max_end = 0u64;
    for t in &g.tensors {
        let e = per_kind.entry(format!("{:?}", t.ttype)).or_default(); e.0 += 1; e.1 += t.nbytes;
        max_end = max_end.max(t.offset + t.nbytes);
    }
    println!("GGUF v{} · {} metadata · {} tensors · data from byte {}", g.version, g.meta.len(), g.tensors.len(), g.data_start);
    for (k, (n, b)) in &per_kind { println!("  {k:8} {n:4} tensors · {:8.2} MiB", *b as f64 / 1048576.0); }
    let wait = g.data_start + max_end; let actual = g.file_len().unwrap();
    println!("end of the last tensor {wait} · file length {actual} · {}", if wait == actual { "MATCH" } else { "do NOT match" });
    for k in ["general.architecture", "qwen35moe.block_count", "qwen35moe.nextn_predict_layers", "qwen35moe.embedding_length"] {
        println!("  {k} = {:?}", g.meta.get(k));
    }
}

fn token(text: &str, reference: &str) {
    let tk = tokenizers::Tokenizer::from_file(tokenizer_path()).expect("tokenizer");
    let s = std::fs::read_to_string(text).expect("text");
    let ids = tk.encode(s, false).expect("encode").get_ids().to_vec();
    let b = std::fs::read(reference).expect("reference");
    let r: Vec<u32> = b.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect();
    let different = ids.iter().zip(&r).filter(|(a, b)| a != b).count();
    let first = ids.iter().zip(&r).position(|(a, b)| a != b);
    println!("mechta {} tokens · llama.cpp {} tokens · different {} · first different {:?}", ids.len(), r.len(), different, first);
}

/// Step 2 of M1: decodes on the GPU the first N values of a tensor and compares them with ggml.
fn deq(name: &str, n: u64, reference: &str) {
    use mechta_gguf::TType;
    let g = Gguf::open(model_path()).expect("GGUF");
    let t = g.tensor(name).unwrap_or_else(|| panic!("tensor {name} not found")).clone();
    let (per, byte) = t.ttype.block().expect("kind");
    assert!(n % per == 0, "N must be a multiple of {per}");
    let nb = n / per;
    let mut raw = Vec::new();
    g.read(&t, 0, nb * byte, &mut raw).expect("read");
    let entry = match t.ttype { TType::Q8_0 => "q8_0", TType::Q4_K => "q4_k", TType::Q6_K => "q6_k", x => panic!("type {x:?} not handled") };
    let gpu = mechta_gpu::Gpu::new();
    let p = gpu.pipeline(include_str!("wgsl/dequant.wgsl"), entry);
    let b_raw = gpu.buffer_byte("raw", &raw);
    let b_out = gpu.empty_buffer("out", n * 4);
    let u = gpu.uniform(&[nb as u32, 0, 0, 0]);
    gpu.run(&p, &[&b_raw, &b_out, &u], ((nb as u32).div_ceil(64), 1, 1));
    let v = gpu.read_f32(&b_out, n as usize);
    let r: Vec<f32> = std::fs::read(reference).expect("reference").chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(r.len(), v.len(), "different lengths");
    let (mut equal, mut max_ass, mut max_ulp) = (0usize, 0f32, 0u32);
    for (a, b) in v.iter().zip(&r) {
        if a.to_bits() == b.to_bits() { equal += 1; }
        max_ass = max_ass.max((a - b).abs());
        max_ulp = max_ulp.max((a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32);
    }
    println!("{name} ({:?}, {n} values, {} blocks) on {}: bit-identical {equal}/{} ({:.4}%) · max absolute diff {max_ass:e} · max ulp {max_ulp}",
        t.ttype, nb, gpu.info.name, v.len(), 100.0 * equal as f64 / v.len() as f64);
}
