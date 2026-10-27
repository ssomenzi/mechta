//! Minimal HTTP/1.1 on std::net: just enough for a local OpenAI client (opencode, curl).
//! One request per connection (`Connection: close`): no keep-alive, no state between requests.
//! Chosen instead of a library because SSE streaming must go out immediately, piece by piece, and a write
//! error must tell the engine that the client has gone away.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::{Duration, Instant};

/// Maximum accepted body. A conversation of 262144 tokens fits in a few MiB, and
/// each connection can hold up to this value in RAM (see `MAX_CONNECTIONS` in lib.rs).
const MAX_BODY: usize = 16 * 1024 * 1024;
/// Maximum headers (request line + headers).
const MAX_HEADERS: usize = 64 * 1024;
/// Maximum time to receive a whole request (headers + body). The socket read timeout
/// (60 s) fires only if the client is completely silent: a client sending one byte per minute would hold a thread and
/// a connection slot forever.
const READ_DEADLINE: Duration = Duration::from_secs(30);

pub struct Request {
    pub method: String,
    /// path without query string
    pub path: String,
    pub body: Vec<u8>,
    /// value of the `Authorization` header, if present (optional API key)
    pub authorization: Option<String>,
}

/// Error reading the request: HTTP status to answer + message. Status 0 = connection to just close.
pub struct ReadError(pub u16, pub String);

/// Reader with a deadline: each `read` on the socket has as timeout the time left, and the deadline is checked after
/// each `read`. `BufReader::read_line` is not suitable: the socket timeout applies to each single `recv`, and a client
/// sending one byte every few seconds restarts it every time (measured: 60 s
/// without closing both with the check after `read_line` and with the timeout set before).
struct Reader<'a> { sock: &'a TcpStream, start: Instant, buf: Vec<u8>, da: usize }

impl Reader<'_> {
    /// Reads more bytes from the socket appended to `buf`; 0 = connection closed.
    fn fill(&mut self) -> Result<usize, ReadError> {
        let resta = READ_DEADLINE.saturating_sub(self.start.elapsed());
        if resta.is_zero() { return Err(ReadError(408, "request too slow".into())); }
        self.sock.set_read_timeout(Some(resta)).map_err(|e| ReadError(0, e.to_string()))?;
        let mut piece = [0u8; 8 * 1024];
        let n = match self.sock.read(&mut piece) {
            Ok(n) => n,
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {
                return Err(ReadError(408, "request too slow".into()))
            }
            Err(e) => return Err(ReadError(0, e.to_string())),
        };
        self.buf.extend_from_slice(&piece[..n]);
        Ok(n)
    }
    /// A header line (without the line terminator). `nread` counts the header bytes for the limit.
    fn row(&mut self, nread: &mut usize) -> Result<String, ReadError> {
        loop {
            if let Some(i) = self.buf[self.da..].iter().position(|&c| c == b'\n') {
                let row = self.buf[self.da..self.da + i].to_vec();
                self.da += i + 1;
                *nread += i + 1;
                if *nread > MAX_HEADERS { return Err(ReadError(431, "headers too large".into())); }
                let row = String::from_utf8(row).map_err(|_| ReadError(400, "headers are not UTF-8".into()))?;
                return Ok(row.trim_end_matches('\r').to_string());
            }
            if self.buf.len() - self.da > MAX_HEADERS { return Err(ReadError(431, "headers too large".into())); }
            if self.fill()? == 0 { return Err(ReadError(0, "connection closed".into())); }
        }
    }
    /// `n` bytes of body: first those already in `buf`, then from the socket, in pieces.
    fn body(&mut self, n: usize) -> Result<Vec<u8>, ReadError> {
        let mut body = Vec::with_capacity(n.min(1 << 20));
        let already = (self.buf.len() - self.da).min(n);
        body.extend_from_slice(&self.buf[self.da..self.da + already]);
        self.da += already;
        while body.len() < n {
            self.buf.clear(); self.da = 0;
            if self.fill()? == 0 { return Err(ReadError(0, "incomplete body: connection closed".into())); }
            let take = self.buf.len().min(n - body.len());
            body.extend_from_slice(&self.buf[..take]);
            self.da = take;
        }
        Ok(body)
    }
}

/// Reads a complete request from the socket.
pub fn read(sock: &mut TcpStream) -> Result<Request, ReadError> {
    let mut reader = Reader { sock, start: Instant::now(), buf: Vec::with_capacity(8 * 1024), da: 0 };
    let mut nread = 0usize;
    let row = reader.row(&mut nread)?;
    let mut parts = row.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let target = parts.next().unwrap_or("").to_string();
    if method.is_empty() || target.is_empty() { return Err(ReadError(400, "invalid request line".into())); }
    let path = target.split('?').next().unwrap_or("").to_string();

    let mut length: Option<usize> = None;
    let mut expect_continue = false;
    let mut chunked = false;
    let mut authorization = None;
    loop {
        let r = reader.row(&mut nread)?;
        if r.is_empty() { break; }
        if let Some((name, value)) = r.split_once(':') {
            let name = name.trim().to_ascii_lowercase();
            let value = value.trim();
            match name.as_str() {
                "content-length" => {
                    length = Some(value.parse().map_err(|_| ReadError(400, "invalid Content-Length".into()))?)
                }
                "expect" if value.eq_ignore_ascii_case("100-continue") => expect_continue = true,
                "transfer-encoding" if value.to_ascii_lowercase().contains("chunked") => chunked = true,
                "authorization" => authorization = Some(value.to_string()),
                _ => {}
            }
        }
    }
    if chunked { return Err(ReadError(411, "chunked body not supported: Content-Length required".into())); }
    let n = length.unwrap_or(0);
    if n > MAX_BODY { return Err(ReadError(413, "body too large".into())); }
    if expect_continue && n > 0 {
        // curl sends "Expect: 100-continue" for bodies over 1 KB and waits 1 s if we do not answer
        let _ = (&*reader.sock).write_all(b"HTTP/1.1 100 Continue\r\n\r\n");
    }
    // The body is read in pieces: memory grows with the bytes that really arrive, not with the declared
    // Content-Length, and a slow client is closed at the deadline.
    let body = reader.body(n)?;
    // the timeout goes back to the connection one (lib.rs sets it before calling `read`)
    let _ = sock.set_read_timeout(Some(Duration::from_secs(60)));
    Ok(Request { method, path, body, authorization })
}

fn status_text(state: u16) -> &'static str {
    match state {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        411 => "Length Required",
        413 => "Payload Too Large",
        431 => "Request Header Fields Too Large",
        500 => "Internal Server Error",
        503 => "Service Unavailable",
        _ => "Status",
    }
}

/// Complete response with the given content type (the trace page and its texts).
pub fn respond(sock: &mut TcpStream, state: u16, content_type: &str, body: &str) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {state} {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n",
        status_text(state),
        body.len()
    );
    let mut all = head.into_bytes();
    all.extend_from_slice(body.as_bytes());
    sock.write_all(&all)?;
    sock.flush()
}

/// Complete response, with JSON body.
pub fn respond_json(sock: &mut TcpStream, state: u16, body: &str) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {state} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        status_text(state),
        body.len()
    );
    let mut all = head.into_bytes();
    all.extend_from_slice(body.as_bytes());
    sock.write_all(&all)?;
    sock.flush()
}

/// Streaming response (Server-Sent Events) with chunked Transfer-Encoding: each event goes out immediately.
/// Owns a copy of the socket (`try_clone`), so the original stays free to check whether the client has disappeared.
pub struct Sse {
    sock: TcpStream,
}

impl Sse {
    pub fn open(original: &TcpStream) -> std::io::Result<Self> {
        let mut sock = original.try_clone()?;
        sock.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
        )?;
        sock.flush()?;
        Ok(Sse { sock })
    }

    /// Writes "data: <data>\n\n" as an HTTP chunk and pushes it out. Error = client gone.
    pub fn data(&mut self, data: &str) -> std::io::Result<()> {
        let load = format!("data: {data}\n\n");
        let mut piece = format!("{:x}\r\n", load.len()).into_bytes();
        piece.extend_from_slice(load.as_bytes());
        piece.extend_from_slice(b"\r\n");
        self.sock.write_all(&piece)?;
        self.sock.flush()
    }

    /// SSE comment (": <text>\n\n"): clients ignore it, but it keeps the connection alive while the engine is queued or
    /// in the prefill (opencode closed a long prefill at exactly 300 s and repeated the request).
    pub fn comment(&mut self, text: &str) -> std::io::Result<()> {
        let load = format!(": {text}\n\n");
        let mut piece = format!("{:x}\r\n", load.len()).into_bytes();
        piece.extend_from_slice(load.as_bytes());
        piece.extend_from_slice(b"\r\n");
        self.sock.write_all(&piece)?;
        self.sock.flush()
    }

    /// Final zero-length chunk.
    pub fn close(mut self) -> std::io::Result<()> {
        self.sock.write_all(b"0\r\n\r\n")?;
        self.sock.flush()
    }
}
