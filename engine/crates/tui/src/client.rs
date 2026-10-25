//! HTTP client of the chat TUI: one streaming request to the mechta server (OpenAI-compatible, 127.0.0.1:4131).
//! The server answers with Server-Sent Events in chunked transfer encoding, one request per connection; no HTTP
//! library is needed for that. Every piece of the answer is sent to the UI thread as an `Event`.
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Server address: MECHTA_CHAT_ADDR, default 127.0.0.1:4131 (the fake engine of the tests listens on 4132).
pub fn addr() -> String { std::env::var("MECHTA_CHAT_ADDR").unwrap_or_else(|_| "127.0.0.1:4131".to_string()) }

pub enum Event {
    /// a piece of the thinking (reasoning_content) or of the answer (content)
    Reasoning(String),
    Content(String),
    /// the answer is over (normally, or stopped with Esc)
    Done { stopped: bool },
    Error(String),
}

/// Sends the conversation and streams the answer to `tx`. `cancel` set to true closes the connection: the server
/// notices that the client is gone and stops generating.
pub fn stream_chat(messages: serde_json::Value, thinking: bool, tx: Sender<Event>, cancel: Arc<AtomicBool>) {
    let body = serde_json::json!({
        "model": "qwen3.6-35b-a3b", "stream": true, "messages": messages,
        "chat_template_kwargs": { "enable_thinking": thinking },
    }).to_string();
    let result = (|| -> std::io::Result<bool> {
        let a = addr();
        let mut s = TcpStream::connect(&a)?;
        // MECHTA_API_KEY, the same variable as the server, if the server asks for a key
        let key = std::env::var("MECHTA_API_KEY").ok().filter(|k| !k.is_empty()).map(|k| format!("Authorization: Bearer {k}\r\n")).unwrap_or_default();
        write!(s, "POST /v1/chat/completions HTTP/1.1\r\nHost: {a}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n{key}Connection: close\r\n\r\n{body}", body.len())?;
        s.set_read_timeout(Some(std::time::Duration::from_millis(200)))?;
        let mut r = BufReader::new(s);
        // status line and headers
        let mut line = String::new();
        read_line(&mut r, &mut line, &cancel)?;
        if !line.contains(" 200 ") {
            let mut rest = String::new();
            let _ = r.read_to_string(&mut rest);
            return Err(std::io::Error::other(format!("{} {}", line.trim(), rest.lines().last().unwrap_or(""))));
        }
        loop { line.clear(); read_line(&mut r, &mut line, &cancel)?; if line.trim().is_empty() { break; } }
        // chunked body: "<hex size>\r\n<data>\r\n", data = SSE lines
        let mut sse = String::new();
        loop {
            if cancel.load(Ordering::Relaxed) { return Ok(true); }
            line.clear();
            read_line(&mut r, &mut line, &cancel)?;
            let size = usize::from_str_radix(line.trim(), 16).map_err(|_| std::io::Error::other(format!("bad chunk size {:?}", line.trim())))?;
            if size == 0 { return Ok(false); }
            let mut buf = vec![0u8; size + 2];
            read_exact(&mut r, &mut buf, &cancel)?;
            sse.push_str(&String::from_utf8_lossy(&buf[..size]));
            while let Some(end) = sse.find("\n\n") {
                let ev: String = sse.drain(..end + 2).collect();
                for l in ev.lines() {
                    let Some(data) = l.strip_prefix("data: ") else { continue };   // ": keep-alive" comments too
                    if data == "[DONE]" { return Ok(false); }
                    let Ok(v) = serde_json::from_str::<serde_json::Value>(data) else { continue };
                    if let Some(e) = v.get("error") { return Err(std::io::Error::other(e.to_string())); }
                    let d = &v["choices"][0]["delta"];
                    if let Some(t) = d["reasoning_content"].as_str() { if !t.is_empty() { let _ = tx.send(Event::Reasoning(t.to_string())); } }
                    if let Some(t) = d["content"].as_str() { if !t.is_empty() { let _ = tx.send(Event::Content(t.to_string())); } }
                }
            }
        }
    })();
    let _ = match result {
        Ok(stopped) => tx.send(Event::Done { stopped }),
        Err(e) if cancel.load(Ordering::Relaxed) => { let _ = e; tx.send(Event::Done { stopped: true }) }
        Err(e) => tx.send(Event::Error(e.to_string())),
    };
}

// Reads with a short timeout so that `cancel` is seen even while the server is busy with a long prefill.
fn read_line<R: BufRead>(r: &mut R, line: &mut String, cancel: &AtomicBool) -> std::io::Result<()> {
    loop {
        match r.read_line(line) {
            Ok(0) => return Err(std::io::Error::other("connection closed")),
            Ok(_) => return Ok(()),
            Err(e) if is_timeout(&e) => { if cancel.load(Ordering::Relaxed) { return Err(e); } }
            Err(e) => return Err(e),
        }
    }
}
fn read_exact<R: Read>(r: &mut R, buf: &mut [u8], cancel: &AtomicBool) -> std::io::Result<()> {
    let mut got = 0;
    while got < buf.len() {
        match r.read(&mut buf[got..]) {
            Ok(0) => return Err(std::io::Error::other("connection closed")),
            Ok(n) => got += n,
            Err(e) if is_timeout(&e) => { if cancel.load(Ordering::Relaxed) { return Err(e); } }
            Err(e) => return Err(e),
        }
    }
    Ok(())
}
fn is_timeout(e: &std::io::Error) -> bool { matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) }

/// Speeds of the last request, from the file the server writes at the end of each one.
pub fn last_speeds() -> Option<(f64, u64, f64, u64)> {
    let home = std::env::var_os("HOME")?;
    let s = std::fs::read_to_string(std::path::Path::new(&home).join(".cache/mechta/last.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&s).ok()?;
    Some((v["prefill_tps"].as_f64()?, v["new_tokens"].as_u64()?, v["gen_tps"].as_f64()?, v["generated_tokens"].as_u64()?))
}

/// Context length and default thinking of the server, from GET /v1/models (None if it does not answer).
pub fn server_info() -> Option<(u64, bool)> {
    let a = addr();
    let mut s = TcpStream::connect_timeout(&a.parse().ok()?, std::time::Duration::from_millis(500)).ok()?;
    s.set_read_timeout(Some(std::time::Duration::from_secs(2))).ok()?;
    write!(s, "GET /v1/models HTTP/1.1\r\nHost: {a}\r\nConnection: close\r\n\r\n").ok()?;
    let mut body = String::new();
    s.read_to_string(&mut body).ok()?;
    let json = body.split("\r\n\r\n").nth(1)?;
    let v: serde_json::Value = serde_json::from_str(json).ok()?;
    let m = &v["data"][0];
    Some((m["context_length"].as_u64()?, m["thinking_default"].as_bool().unwrap_or(true)))
}

/// Tokens of the last request (prompt + generated), from the file the server writes at the end of each one.
pub fn last_tokens() -> Option<u64> {
    let home = std::env::var_os("HOME")?;
    let s = std::fs::read_to_string(std::path::Path::new(&home).join(".cache/mechta/last.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&s).ok()?;
    Some(v["prompt_tokens"].as_u64()? + v["generated_tokens"].as_u64()?)
}

