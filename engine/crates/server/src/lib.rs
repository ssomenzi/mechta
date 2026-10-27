//! OpenAI-compatible HTTP server.
//!
//! Architecture: ONE thread for the engine (it creates and owns it), one thread per HTTP connection. Connections
//! send a `Job` on the engine channel; the engine runs them one at a time, in order of arrival, and sends the generated
//! pieces back on a channel private to the job. If the client disappears, the connection drops its channel: the
//! engine's next `piece` fails to send, returns false and the generation stops.
#![forbid(unsafe_code)]
mod titles;

mod http;
mod trace_page;

use std::net::{TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mechta_api::{Outcome, Finish, Engine, Params};
use mechta_chat::{Event, Options, Parser};
/// The model family of `Config::family`, for the engine crate (which does not use the chat crate)
pub use mechta_chat::Family;
use serde_json::{Value, json};

/// Server configuration.
#[derive(Clone, Debug)]
pub struct Config {
    /// listening address, default "127.0.0.1:4131"
    pub address: String,
    /// thinking on by default (overridable per request if opencode sends it)
    pub thinking: bool,
    /// default thinking budget in tokens (0 = no limit), MECHTA_THINK_BUDGET; a request can change it
    pub reasoning_budget: usize,
    /// chat template and tool-call format of the model (Qwen by default; Flint for a Flint GGUF)
    pub family: mechta_chat::Family,
    /// API key (`MECHTA_API_KEY`). With one set, `/v1/*` wants `Authorization: Bearer <key>`;
    /// `/health` stays open. None = no check (a server on 127.0.0.1 for one user).
    pub api_key: Option<String>,
}
impl Default for Config { fn default() -> Self { Config { address: "127.0.0.1:4131".into(), thinking: true, reasoning_budget: 0, family: mechta_chat::Family::Qwen, api_key: None } } }

/// A job for the engine thread.
struct Job {
    prompt: String,
    params: Params,
    replies: mpsc::Sender<FromEngine>,
}

/// Messages from the engine to the connection that requested the job.
enum FromEngine {
    /// the job starts: prompt tokens and effective max_token (already reduced to fit in the context)
    Start { token_prompt: usize, max_token: usize },
    /// the prompt alone does not fit in the context
    Rejected(String),
    Piece(String),
    Finish(Result<Outcome, String>),
}

/// Fixed engine data, read once at start-up.
struct Info {
    name: String,
    context: usize,
}

struct State {
    info: Info,
    cfg: Config,
}

static COUNTER: AtomicU64 = AtomicU64::new(0);

/// Connections served at the same time. Each connection is a thread and can hold in RAM up to
/// `MAX_BODY` (http.rs): without a limit 100 slow clients held 1.6 GB and 100 threads. Beyond the
/// limit the answer is 503 immediately. A client queued for the engine takes a slot: 32 are enough for opencode and a few
/// other clients together.
const MAX_CONNECTIONS: usize = 32;
static CONNECTIONS: AtomicUsize = AtomicUsize::new(0);
/// Frees the slot when the connection thread ends, however it ends (even with a panic).
struct ConnectionSlot;
impl Drop for ConnectionSlot { fn drop(&mut self) { CONNECTIONS.fetch_sub(1, Ordering::AcqRel); } }

/// Starts the server and does not return. `create_engine` is called ONCE, on the thread dedicated to the engine (the engine is not
/// Send). HTTP requests pass to the engine one at a time.
pub fn start(create_engine: Box<dyn FnOnce() -> Box<dyn mechta_api::Engine> + Send>, cfg: Config) -> ! {
    let (tx_jobs, rx_jobs) = mpsc::channel::<Job>();
    let (tx_info, rx_info) = mpsc::channel::<Info>();

    let engine_thread = match std::thread::Builder::new()
        .name("engine".into())
        .spawn(move || engine_loop(create_engine, tx_info, rx_jobs))
    {
        Ok(t) => t,
        Err(e) => {
            log(&format!("ERROR: cannot create the engine thread: {e}"));
            std::process::exit(1);
        }
    };

    // The port is opened only when the engine is ready: while it loads, whoever connects finds the port closed.
    let info = match rx_info.recv() {
        Ok(i) => i,
        Err(_) => {
            // the thread died during creation (the panic has already been printed by the default hook)
            log("ERROR: the engine did not start (creation failed)");
            std::process::exit(1);
        }
    };

    // Watchdog: if the engine thread dies (panic or other), the server must not stay alive and silent.
    std::thread::spawn(move || {
        let outcome = engine_thread.join();
        match outcome {
            Err(_) => log("FATAL: the engine thread panicked; exiting with code 101"),
            Ok(()) => log("FATAL: the engine thread ended; exiting with code 101"),
        }
        std::process::exit(101);
    });

    let ascolto = match TcpListener::bind(&cfg.address) {
        Ok(a) => a,
        Err(e) => {
            log(&format!("ERROR: cannot listen on {}: {e}", cfg.address));
            std::process::exit(1);
        }
    };
    set_busy(false);   // a stale marker of a server that died in the middle of a request
    log(&format!(
        "ready on http://{}/v1 — model {} — context {} — thinking {} — thinking budget {}",
        cfg.address,
        info.name,
        info.context,
        if cfg.thinking { "on" } else { "off" },
        if cfg.reasoning_budget > 0 { cfg.reasoning_budget.to_string() } else { "none".into() }
    ));
    if cfg.api_key.is_some() { log("API key required on /v1/* (MECHTA_API_KEY)"); }

    let state = Arc::new(State { info, cfg });
    for conn in ascolto.incoming() {
        match conn {
            Ok(mut sock) => {
                if CONNECTIONS.fetch_add(1, Ordering::AcqRel) >= MAX_CONNECTIONS {
                    CONNECTIONS.fetch_sub(1, Ordering::AcqRel);
                    log(&format!("503: more than {MAX_CONNECTIONS} connections at once, refused"));
                    let _ = sock.set_write_timeout(Some(Duration::from_secs(2)));
                    let _ = http::respond_json(&mut sock, 503, &error_json("Too many connections: retry shortly", "server_error", None));
                    continue;
                }
                let slot = ConnectionSlot;
                let state = Arc::clone(&state);
                let tx = tx_jobs.clone();
                let r = std::thread::Builder::new().name("http".into()).spawn(move || { let _slot = slot; mechta_trace::set_thread(mechta_trace::HTTP); handle(sock, &state, &tx) });
                if let Err(e) = r {
                    log(&format!("ERROR: connection thread not created: {e}"));
                }
            }
            Err(e) => {
                // transient errors (too many open files...): wait a moment and go on
                log(&format!("accept error: {e}"));
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
    log("FATAL: the listening socket closed");
    std::process::exit(1)
}

/// The engine thread: creates the engine, publishes the fixed data, then runs the jobs one at a time.
fn engine_loop(
    create_engine: Box<dyn FnOnce() -> Box<dyn Engine> + Send>,
    tx_info: mpsc::Sender<Info>,
    rx_jobs: mpsc::Receiver<Job>,
) {
    mechta_trace::mark_engine_thread();
    let mut engine = create_engine();
    let context = engine.max_context();
    if tx_info.send(Info { name: engine.model_name().to_string(), context }).is_err() {
        return;
    }
    for job in rx_jobs {
        // a tokenizer error goes back to the client as 400, it does not bring down the engine thread
        let token_prompt = match engine.count_tokens(&job.prompt) {
            Ok(n) => n,
            Err(e) => { let _ = job.replies.send(FromEngine::Rejected(format!("prompt cannot be tokenized: {e}"))); continue; }
        };
        if token_prompt >= context {
            let _ = job.replies.send(FromEngine::Rejected(format!(
                "The prompt is {token_prompt} tokens, the maximum context is {context}: shorten the conversation."
            )));
            continue;
        }
        let mut p = job.params;
        p.max_token = p.max_token.min(context - token_prompt).max(1);
        // if the client already left while waiting in the queue, the job is skipped
        if job.replies.send(FromEngine::Start { token_prompt, max_token: p.max_token }).is_err() {
            continue;
        }
        let tx = job.replies.clone();
        set_busy(true);
        let outcome = engine.generate(&job.prompt, &p, &mut |text: &str| tx.send(FromEngine::Piece(text.to_string())).is_ok());
        set_busy(false);
        let _ = job.replies.send(FromEngine::Finish(outcome));
    }
}

/// A connection: reads the request, dispatches it, answers, closes.
fn handle(mut sock: TcpStream, state: &State, tx: &mpsc::Sender<Job>) {
    let _ = sock.set_nodelay(true);
    let _ = sock.set_read_timeout(Some(Duration::from_secs(60)));
    let _ = sock.set_write_timeout(Some(Duration::from_secs(60)));
    let request = match http::read(&mut sock) {
        Ok(r) => r,
        Err(http::ReadError(0, _)) => return,
        Err(http::ReadError(code, msg)) => {
            let _ = http::respond_json(&mut sock, code, &error_json(&msg, "invalid_request_error", None));
            return;
        }
    };
    let method = request.method.as_str();
    let path = request.path.trim_end_matches('/');
    if let Some(key) = &state.cfg.api_key
        && path.starts_with("/v1")
        && !authorized(request.authorization.as_deref(), key)
    {
        log("401: API key missing or wrong");
        let _ = http::respond_json(&mut sock, 401, &error_json("API key missing or wrong (Authorization: Bearer ...)", "authentication_error", Some("invalid_api_key")));
        return;
    }
    if trace_page::route(&mut sock, method, path) { return; }
    match path {
        "/v1/models" if method == "GET" => {
            let body = json!({
                "object": "list",
                "data": [{
                    "id": state.info.name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "mechta",
                    "context_length": state.info.context,
                    // mechta-chat follows the server default unless /think changes it
                    "thinking_default": state.cfg.thinking,
                    "thinking_budget": state.cfg.reasoning_budget,
                }]
            });
            let _ = http::respond_json(&mut sock, 200, &body.to_string());
        }
        "/health" if method == "GET" => {
            let _ = http::respond_json(&mut sock, 200, r#"{"status":"ok"}"#);
        }
        "/v1/chat/completions" if method == "POST" => completion(&mut sock, &request.body, state, tx),
        "/v1/models" | "/health" | "/v1/chat/completions" => {
            let _ = http::respond_json(&mut sock, 405, &error_json("Method not allowed", "invalid_request_error", None));
        }
        other => {
            // the path comes from the client; in the log and in the response it goes without control characters
            let msg = format!("Unknown path: {} {}", sanitize(method), sanitize(other));
            let _ = http::respond_json(&mut sock, 404, &error_json(&msg, "invalid_request_error", None));
        }
    }
}

/// `Authorization: Bearer <key>`, compared in constant time (does not reveal how many characters match).
fn authorized(header: Option<&str>, key: &str) -> bool {
    let Some(h) = header else { return false };
    let Some(given) = h.strip_prefix("Bearer ").or_else(|| h.strip_prefix("bearer ")) else { return false };
    let (a, b) = (given.trim().as_bytes(), key.as_bytes());
    let mut diff = a.len() ^ b.len();
    for i in 0..a.len().max(b.len()) { diff |= usize::from(a.get(i).copied().unwrap_or(0) ^ b.get(i).copied().unwrap_or(0)); }
    diff == 0
}

/// OpenAI-style error.
fn error_json(message: &str, kind: &str, code: Option<&str>) -> String {
    json!({ "error": { "message": message, "type": kind, "param": null, "code": code } }).to_string()
}

/// Options of a /v1/chat/completions request, already extracted from the JSON.
struct ChatRequest {
    prompt: String,
    tools: Option<Value>,
    thinking: bool,
    params: Params,
    stream: bool,
    usage_in_stream: bool,
    /// MECHTA_SAVE_REQUESTS: number of the saved request file (the raw output gets the same number)
    number: Option<u128>,
}

fn read_chat_request(body: &[u8], cfg: &Config) -> Result<ChatRequest, String> {
    let v: Value = serde_json::from_slice(body).map_err(|e| format!("invalid JSON: {e}"))?;
    // MECHTA_SAVE_REQUESTS=<folder>: saves the body of every request (to study what the client sends)
    let mut number = None;
    if let Ok(d) = std::env::var("MECHTA_SAVE_REQUESTS") {
        let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|t| t.as_millis()).unwrap_or(0);
        let _ = std::fs::write(format!("{d}/request-{n}.json"), body);
        number = Some(n);
    }
    let messages = &v["messages"];
    if !messages.is_array() || messages.as_array().is_some_and(|a| a.is_empty()) {
        return Err("'messages' must be a non-empty array".into());
    }
    // tools: only if there is at least one tool and tool_choice is not "none"
    let tools = match &v["tools"] {
        Value::Array(a) if !a.is_empty() && v["tool_choice"].as_str() != Some("none") => Some(v["tools"].clone()),
        _ => None,
    };
    // thinking: default from the Config; per request with chat_template_kwargs.enable_thinking (convention of vLLM and
    // llama.cpp) or, failing that, enable_thinking at the top level
    let thinking = v["chat_template_kwargs"]["enable_thinking"]
        .as_bool()
        .or_else(|| v["enable_thinking"].as_bool())
        .unwrap_or(cfg.thinking);

    let mut p = Params::default();
    if let Some(n) = v["max_completion_tokens"].as_u64().or_else(|| v["max_tokens"].as_u64()) {
        p.max_token = n as usize;
    }
    if let Some(t) = v["temperature"].as_f64() { p.temperature = t as f32; }
    if let Some(t) = v["top_p"].as_f64() { p.top_p = t as f32; }
    if let Some(k) = v["top_k"].as_u64() { p.top_k = k as usize; }
    if let Some(b) = v["mtp"].as_bool() { p.mtp = b; }   // mechta-only field, for measurements
    if let Some(s) = v["seed"].as_u64() { p.seed = s; } else if let Some(s) = v["seed"].as_i64() { p.seed = s as u64; }
    // thinking budget: `thinking_budget` (name of the Qwen cloud API) at the top level or in chat_template_kwargs; 0 = no
    // limit; otherwise the server default. Only with the thinking on.
    if thinking {
        p.reasoning_budget = v["thinking_budget"].as_u64()
            .or_else(|| v["chat_template_kwargs"]["thinking_budget"].as_u64())
            .map(|n| n as usize).unwrap_or(cfg.reasoning_budget);
    }

    // preserve_thinking: on by default (keeps the prefix cache valid across user turns); a client can turn it off
    let preserve_thinking = v["chat_template_kwargs"]["preserve_thinking"].as_bool().unwrap_or(true);
    let prompt = mechta_chat::prompt(messages, tools.as_ref(), &Options { thinking, preserve_thinking, family: cfg.family })?;
    Ok(ChatRequest {
        prompt,
        tools,
        thinking,
        params: p,
        stream: v["stream"].as_bool().unwrap_or(false),
        usage_in_stream: v["stream_options"]["include_usage"].as_bool().unwrap_or(false),
        number,
    })
}

/// MECHTA_SAVE_OUTPUTS=<folder>: the raw text of the model (before the parser) of every request, written
/// when the request ends in any way (also client gone or error), to see what the model really wrote in a tool call.
/// File `output-<n>.txt`, with the same number as `request-<n>.json` when MECHTA_SAVE_REQUESTS is set too.
/// At the end: the tool calls as the parser read them (name and argument keys with their length).
struct RawOutput { dir: Option<String>, n: u128, text: String, calls: Vec<String> }
impl RawOutput {
    fn new(number: Option<u128>) -> Self {
        let n = number.unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map(|t| t.as_millis()).unwrap_or(0));
        RawOutput { dir: std::env::var("MECHTA_SAVE_OUTPUTS").ok(), n, text: String::new(), calls: Vec::new() }
    }
    fn push(&mut self, t: &str) { if self.dir.is_some() { self.text.push_str(t); } }
    fn call(&mut self, name: &str, args_json: &str) {
        if self.dir.is_none() { return; }
        let keys = serde_json::from_str::<Value>(args_json).ok().and_then(|v| v.as_object().map(|o| o.iter()
            .map(|(k, v)| format!("{k}={}", v.as_str().map(|s| s.chars().count()).unwrap_or(v.to_string().len()))).collect::<Vec<_>>().join(" ")))
            .unwrap_or_else(|| "(arguments not JSON)".into());
        self.calls.push(format!("{name}: {keys}"));
    }
}
impl Drop for RawOutput {
    fn drop(&mut self) {
        let Some(d) = &self.dir else { return };
        let body = format!("{}\n=====PARSED CALLS ({})\n{}\n", self.text, self.calls.len(), self.calls.join("\n"));
        if let Err(e) = std::fs::write(format!("{d}/output-{}.txt", self.n), body) { log(&format!("raw output not saved: {e}")); }
    }
}

/// True if the client closed the connection (non-blocking read: 0 bytes = end of stream).
fn client_closed(sock: &TcpStream) -> bool {
    if sock.set_nonblocking(true).is_err() {
        return true;
    }
    let mut b = [0u8; 1];
    let closed = match sock.peek(&mut b) {
        Ok(0) => true,
        Ok(_) => false,
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => false,
        Err(_) => true,
    };
    let _ = sock.set_nonblocking(false);
    closed
}

/// Collects the output events and translates them into OpenAI pieces.
struct Output {
    id: String,
    created: u64,
    model_path: String,
    n_calls: usize,
    content: String,
    reasoning: String,
    calls: Vec<Value>,
}

impl Output {
    /// OpenAI delta of an event; None if empty
    fn delta(&mut self, ev: Event) -> Option<Value> {
        match ev {
            Event::Reasoning(t) if !t.is_empty() => {
                self.reasoning.push_str(&t);
                Some(json!({ "reasoning_content": t }))
            }
            Event::Text(t) if !t.is_empty() => {
                self.content.push_str(&t);
                Some(json!({ "content": t }))
            }
            Event::ToolCall { id, name, arguments_json } => {
                let index = self.n_calls;
                self.n_calls += 1;
                let call = json!({
                    "id": id, "type": "function", "function": { "name": name, "arguments": arguments_json }
                });
                self.calls.push(call.clone());
                let mut with_index = call;
                with_index["index"] = json!(index);
                Some(json!({ "tool_calls": [with_index] }))
            }
            _ => None,
        }
    }

    fn piece(&self, delta: Value, end: Option<&str>) -> String {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model_path,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": end }]
        })
        .to_string()
    }

    fn finish_reason(&self, end: Finish) -> &'static str {
        if self.n_calls > 0 {
            return "tool_calls";
        }
        match end {
            Finish::Length => "length",
            Finish::Stop | Finish::Cancelled => "stop",
        }
    }
}

fn usage(e: &Outcome) -> Value {
    json!({
        "prompt_tokens": e.token_prompt,
        "completion_tokens": e.generated_tokens,
        "total_tokens": e.token_prompt + e.generated_tokens,
        "prompt_tokens_details": { "cached_tokens": e.reused_tokens },
    })
}

/// POST /v1/chat/completions
fn completion(sock: &mut TcpStream, body: &[u8], state: &State, tx: &mpsc::Sender<Job>) {
    let start = Instant::now();
    let question = match read_chat_request(body, &state.cfg) {
        Ok(d) => d,
        Err(msg) => {
            log(&format!("400 invalid request: {msg}"));
            let _ = http::respond_json(sock, 400, &error_json(&msg, "invalid_request_error", None));
            return;
        }
    };
    let mut question = question;
    let mut raw = RawOutput::new(question.number);
    let (tx_replies, rx) = mpsc::channel::<FromEngine>();
    // Session title request (opencode): deterministic answer without the engine, so it does not cost a
    // generation and does not break prefix reuse. MECHTA_TITLES=engine to send it to the model.
    if let Some(title) = titles::fake_title(body) {
        question.thinking = false;
        let _ = tx_replies.send(FromEngine::Start { token_prompt: 0, max_token: question.params.max_token });
        let _ = tx_replies.send(FromEngine::Piece(title.clone()));
        let _ = tx_replies.send(FromEngine::Finish(Ok(Outcome { token_prompt: 0, reused_tokens: 0, generated_tokens: 1, end: mechta_api::Finish::Stop, ms_prefill: 0.0, ms_generation: 0.0 })));
        log(&format!("fake title: {title:?}"));
    } else if tx.send(Job { prompt: question.prompt.clone(), params: question.params.clone(), replies: tx_replies }).is_err() {
        let _ = http::respond_json(sock, 503, &error_json("Engine not available", "server_error", None));
        return;
    }

    let mut output = Output {
        id: format!("chatcmpl-{:x}{:04x}", now_seconds(), COUNTER.fetch_add(1, Ordering::Relaxed)),
        created: now_seconds(),
        model_path: state.info.name.clone(),
        n_calls: 0,
        content: String::new(),
        reasoning: String::new(),
        calls: Vec::new(),
    };
    // Streaming: the SSE response opens IMMEDIATELY, so during the queue and the prefill comments can be sent
    // every `KEEPALIVE` and the client does not close for inactivity (opencode closed at 300 s). MECHTA_KEEPALIVE=no:
    // previous behaviour (opening after the job starts, no comments).
    let keepalive = std::env::var("MECHTA_KEEPALIVE").map(|v| v != "no").unwrap_or(true);
    const KEEPALIVE: Duration = Duration::from_secs(10);
    let mut last_byte = Instant::now();
    let mut sse = None;
    if question.stream && keepalive {
        match http::Sse::open(sock) {
            Ok(mut s) => {
                if s.data(&output.piece(json!({ "role": "assistant", "content": "" }), None)).is_ok() { sse = Some(s); }
                else { log("client gone when opening the stream"); return; }
            }
            Err(_) => return,
        }
    }

    // Waiting for the turn (the engine may be busy with another request). If the client leaves in the meantime,
    // the channel is dropped: the engine will skip the job.
    let poll = Duration::from_millis(250);
    let (token_prompt, max_token) = loop {
        match rx.recv_timeout(poll) {
            Ok(FromEngine::Start { token_prompt, max_token }) => break (token_prompt, max_token),
            Ok(FromEngine::Rejected(msg)) => {
                log(&format!("400 context exceeded: {msg}"));
                if let Some(mut s) = sse.take() {
                    let _ = s.data(&error_json(&msg, "invalid_request_error", Some("context_length_exceeded")));
                    let _ = s.data("[DONE]"); let _ = s.close();
                    return;
                }
                let _ = http::respond_json(
                    sock,
                    400,
                    &error_json(&msg, "invalid_request_error", Some("context_length_exceeded")),
                );
                return;
            }
            Ok(_) => {
                // should not happen: no piece before Start
                let _ = http::respond_json(sock, 500, &error_json("Unexpected sequence from the engine", "server_error", None));
                return;
            }
            Err(RecvTimeoutError::Timeout) => {
                let mut via = client_closed(sock);
                if !via && last_byte.elapsed() >= KEEPALIVE && let Some(s) = sse.as_mut() {
                    via = s.comment("queued").is_err(); last_byte = Instant::now();
                }
                if via {
                    log("client gone while waiting in the queue");
                    return;
                }
            }
            Err(RecvTimeoutError::Disconnected) => {
                let _ = http::respond_json(sock, 503, &error_json("Engine not available", "server_error", None));
                return;
            }
        }
    };
    let ms_queue = start.elapsed().as_secs_f64() * 1000.0;

    let mut parser = Parser::per_family(state.cfg.family, question.thinking, question.tools.as_ref());

    // Streaming: the SSE response is opened and the role is sent.
    if question.stream && sse.is_none() {
        match http::Sse::open(sock) {
            Ok(mut s) => {
                if s.data(&output.piece(json!({ "role": "assistant", "content": "" }), None)).is_ok() {
                    sse = Some(s);
                } else {
                    log("client gone when opening the stream");
                    return;
                }
            }
            Err(_) => return,
        }
    }

    // Loop over the pieces. With the stream a write error says that the client has disappeared; without stream (and when the
    // parser holds back the text) the socket is checked every now and then.
    let mut last_check = Instant::now();
    let mut gone = false;
    let outcome = loop {
        let msg = match rx.recv_timeout(poll) {
            Ok(m) => Some(m),
            Err(RecvTimeoutError::Timeout) => None,
            Err(RecvTimeoutError::Disconnected) => break Err("the engine stopped".to_string()),
        };
        let events = match msg {
            Some(FromEngine::Piece(t)) => { raw.push(&t); parser.push(&t) }
            Some(FromEngine::Finish(r)) => break r,
            Some(_) => Vec::new(),
            None => Vec::new(),
        };
        for ev in events {
            if let Event::ToolCall { name, arguments_json, .. } = &ev { raw.call(name, arguments_json); }
            if let Some(d) = output.delta(ev)
                && let Some(s) = sse.as_mut()
            {
                let ts = mechta_trace::start();
                if s.data(&output.piece(d, None)).is_err() { gone = true; } else { last_byte = Instant::now(); }
                mechta_trace::span(mechta_trace::Kind::Send, ts, |e| e.label = "http");
            }
        }
        // the silence is measured on the bytes really sent, not on the pieces of the engine. While the model
        // writes a tool call the parser holds the text back until the call is complete: the engine sends pieces all the
        // time but the client gets nothing, and opencode closed the stream after 300 s and asked again, forever
        // (reproducible with the fake engine and MECHTA_FAKE_TOOL_PIECES).
        if !gone && last_byte.elapsed() >= KEEPALIVE && keepalive && let Some(s) = sse.as_mut() {
            if s.comment("working").is_err() { gone = true; }
            last_byte = Instant::now();
        }
        if !gone && last_check.elapsed() >= poll {
            last_check = Instant::now();
            gone = client_closed(sock);
        }
        if gone {
            // by dropping `rx` the next `piece` fails and the engine stops
            drop(rx);
            log(&format!("client gone during generation ({:.0} ms)", start.elapsed().as_secs_f64() * 1000.0));
            return;
        }
    };

    match outcome {
        Err(e) => {
            log(&format!("ERROR from the engine: {e}"));
            match sse {
                Some(mut s) => {
                    let _ = s.data(&error_json(&e, "server_error", None));
                    let _ = s.data("[DONE]");
                    let _ = s.close();
                }
                None => {
                    let _ = http::respond_json(sock, 500, &error_json(&e, "server_error", None));
                }
            }
        }
        Ok(outcome) => {
            // what the parser was holding back
            for ev in parser.end() {
                if let Event::ToolCall { name, arguments_json, .. } = &ev { raw.call(name, arguments_json); }
                if let Some(d) = output.delta(ev)
                    && let Some(s) = sse.as_mut()
                {
                    let _ = s.data(&output.piece(d, None));
                }
            }
            let reason = output.finish_reason(outcome.end);
            log(&format!(
                "chat stream={} thinking={} budget={} prompt={} reused={} generated={} max={} queue_ms={:.0} prefill_ms={:.0} gen_ms={:.0} t/s={:.1} finish={} ({:?})",
                question.stream,
                question.thinking,
                question.params.reasoning_budget,
                outcome.token_prompt,
                outcome.reused_tokens,
                outcome.generated_tokens,
                max_token,
                ms_queue,
                outcome.ms_prefill,
                outcome.ms_generation,
                if outcome.ms_generation > 0.0 { outcome.generated_tokens as f64 * 1000.0 / outcome.ms_generation } else { 0.0 },
                reason,
                outcome.end
            ));
            let _ = token_prompt;
            write_last_request(&outcome);
            match sse {
                Some(mut s) => {
                    let _ = s.data(&output.piece(json!({}), Some(reason)));
                    if question.usage_in_stream {
                        let u = json!({
                            "id": output.id, "object": "chat.completion.chunk", "created": output.created,
                            "model": output.model_path, "choices": [], "usage": usage(&outcome)
                        });
                        let _ = s.data(&u.to_string());
                    }
                    let _ = s.data("[DONE]");
                    let _ = s.close();
                }
                None => {
                    let mut message = json!({ "role": "assistant", "content": output.content });
                    if !output.reasoning.is_empty() {
                        message["reasoning_content"] = json!(output.reasoning);
                    }
                    if !output.calls.is_empty() {
                        message["tool_calls"] = json!(output.calls);
                        if output.content.is_empty() {
                            message["content"] = Value::Null;
                        }
                    }
                    let body = json!({
                        "id": output.id,
                        "object": "chat.completion",
                        "created": output.created,
                        "model": output.model_path,
                        "choices": [{ "index": 0, "message": message, "finish_reason": reason }],
                        "usage": usage(&outcome),
                    });
                    let _ = http::respond_json(sock, 200, &body.to_string());
                }
            }
        }
    }
}

fn now_seconds() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

/// Text coming from the client, without control characters (ANSI sequences in the journal, fake lines
/// in the log) and not longer than 200 characters.
fn sanitize(s: &str) -> String {
    s.chars().take(200).map(|c| if c.is_control() { '?' } else { c }).collect()
}

/// A log line on stderr with the UTC time.
fn log(msg: &str) {
    let now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default();
    let s = now.as_secs();
    let (g, resto) = ((s / 86400) as i64, s % 86400);
    // from days since 1970 to civil date (Howard Hinnant's algorithm)
    let z = g + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + if month <= 2 { 1 } else { 0 };
    eprintln!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{:03}Z mechta-server: {msg}",
        resto / 3600,
        resto % 3600 / 60,
        resto % 60,
        now.subsec_millis()
    );
}

/// The numbers of the last request in `~/.cache/mechta/last.json`, for a status bar widget.
/// Written to a temporary file and renamed, so a reader never sees half a file. Errors are only logged.
fn write_last_request(e: &Outcome) {
    let Some(home) = std::env::var_os("HOME") else { return };
    let dir = std::path::Path::new(&home).join(".cache/mechta");
    let new_tokens = e.token_prompt.saturating_sub(e.reused_tokens);
    let rate = |n: usize, ms: f64| if ms > 0.0 { n as f64 * 1000.0 / ms } else { 0.0 };
    let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let body = json!({
        "ts": ts, "prompt_tokens": e.token_prompt, "reused_tokens": e.reused_tokens, "new_tokens": new_tokens,
        // prefill speed over ALL the prompt tokens, in cache or not (a design choice)
        "prefill_ms": e.ms_prefill, "prefill_tps": rate(e.token_prompt, e.ms_prefill),
        "generated_tokens": e.generated_tokens, "gen_ms": e.ms_generation, "gen_tps": rate(e.generated_tokens, e.ms_generation),
    });
    let res = std::fs::create_dir_all(&dir).and_then(|_| {
        let tmp = dir.join("last.json.tmp");
        std::fs::write(&tmp, body.to_string())?;
        std::fs::rename(&tmp, dir.join("last.json"))
    });
    if let Err(err) = res { eprintln!("last.json not written: {err}"); }
}

/// `~/.cache/mechta/busy` exists while the engine works on a request (a status bar widget can use it to
/// ask for confirmation before stopping the server in the middle of an answer). Removed at start-up too, in case a
/// previous server died while busy.
pub fn set_busy(busy: bool) {
    let Some(home) = std::env::var_os("HOME") else { return };
    let dir = std::path::Path::new(&home).join(".cache/mechta");
    if busy {
        let ts = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        let _ = std::fs::create_dir_all(&dir).and_then(|_| std::fs::write(dir.join("busy"), ts.to_string()));
    } else {
        let _ = std::fs::remove_file(dir.join("busy"));
    }
}
