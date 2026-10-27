//! The live trace page: `GET /trace` (the page), `GET /trace/locales/<lang>.json`
//! (its texts), `GET /trace/stream` (SSE: the new events every 50 ms). Only from the machine itself (loopback): the
//! server may listen on the LAN, the trace says what the engine is doing. With MECHTA_TRACE off the page says so.
//!
//! The first stream also starts the sampler: CPU load per core every 100 ms (/proc/stat counts in 10 ms ticks, a
//! shorter sample would only read 0 or 100%) and GPU clock and busy percent every 10 ms (sysfs of the first card that
//! has them). Samples are context for reading the events, not measurements of the engine.
use crate::http;
use mechta_trace as trace;
use std::net::TcpStream;
use std::time::Duration;

const PAGE: &str = include_str!("trace/index.html");
const EN: &str = include_str!("trace/locales/en.json");
const IT: &str = include_str!("trace/locales/it.json");

/// Answers a /trace path; false if the path is not one of them.
pub fn route(sock: &mut TcpStream, method: &str, path: &str) -> bool {
    if path != "/trace" && !path.starts_with("/trace/") { return false; }
    if !sock.peer_addr().map(|a| a.ip().is_loopback()).unwrap_or(false) {
        let _ = http::respond(sock, 403, "text/plain", "the trace page is served only to the machine itself\n");
        return true;
    }
    if method != "GET" { let _ = http::respond(sock, 405, "text/plain", "GET only\n"); return true; }
    match path {
        "/trace" => { let _ = http::respond(sock, 200, "text/html; charset=utf-8", PAGE); }
        "/trace/locales/en.json" => { let _ = http::respond(sock, 200, "application/json", EN); }
        "/trace/locales/it.json" => { let _ = http::respond(sock, 200, "application/json", IT); }
        "/trace/stream" => stream(sock),
        _ => { let _ = http::respond(sock, 404, "text/plain", "not found\n"); }
    }
    true
}

/// SSE: first `{"level":N,...}`, then every 50 ms `{"now":ns,"lost":n,"ev":[...]}` with the events recorded since
/// the previous batch. Ends when the client goes.
fn stream(sock: &mut TcpStream) {
    let Ok(mut s) = http::Sse::open(sock) else { return };
    let Some(t) = trace::global() else {
        let _ = s.data(r#"{"level":0}"#);
        let _ = s.close();
        return;
    };
    start_sampler();
    if s.data(&format!(r#"{{"level":{},"kv_bytes_per_pos":{}}}"#, trace::level(), trace::kv_bytes_per_pos())).is_err() { return; }
    // start from the last 3 s of events, so the page is not empty when opened between two requests
    let now = trace::now_ns();
    let mut next = {
        let r = t.ring.lock().unwrap();
        let (v, end, _) = r.since(0);
        end - v.iter().rev().take_while(|e| e.t1 + 3_000_000_000 >= now).count() as u64
    };
    loop {
        std::thread::sleep(Duration::from_millis(50));
        let (v, end, lost) = t.ring.lock().unwrap().since(next);
        next = end;
        let mut body = format!(r#"{{"now":{},"lost":{lost},"ev":["#, trace::now_ns());
        for (i, e) in v.iter().enumerate() { if i > 0 { body.push(','); } body.push_str(&e.json()); }
        body.push_str("]}");
        if s.data(&body).is_err() { return; }
    }
}

/// The sampler thread (once per process).
fn start_sampler() {
    static STARTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    STARTED.get_or_init(|| {
        std::thread::Builder::new().name("trace-sampler".into()).spawn(sampler).ok();
    });
}

fn sampler() {
    trace::set_thread(trace::SAMPLER);
    trace::set_ctx("sampler", 0, 0, 0);
    let card = (0..8).map(|i| format!("/sys/class/drm/card{i}/device")).find(|d| std::path::Path::new(&format!("{d}/gpu_busy_percent")).exists());
    let mut prev: Vec<(u64, u64)> = vec![];
    let mut prev_engine: Option<u64> = None;
    let mut t_cpu = trace::now_ns();
    let mut tick = 0u64;
    loop {
        let t0 = trace::now_ns();
        if let Some(c) = &card {
            if let Some(mhz) = std::fs::read_to_string(format!("{c}/pp_dpm_sclk")).ok().and_then(|s| current_mhz(&s)) {
                let mut e = trace::Event::new(trace::Kind::GpuClock, t0, t0); e.bytes = mhz; trace::record(e);
            }
            if let Some(b) = std::fs::read_to_string(format!("{c}/gpu_busy_percent")).ok().and_then(|s| s.trim().parse::<u64>().ok()) {
                let mut e = trace::Event::new(trace::Kind::GpuBusy, t0, t0); e.bytes = b; trace::record(e);
            }
        }
        if tick.is_multiple_of(10) && let Ok(stat) = std::fs::read_to_string("/proc/stat") {
            let now = parse_stat(&stat);
            if prev.len() == now.len() {
                for (core, ((b0, a0), (b1, a1))) in prev.iter().zip(&now).enumerate() {
                    let all = a1.saturating_sub(*a0).max(1);
                    let mut e = trace::Event::new(trace::Kind::CpuLoad, t_cpu, t0);
                    e.layer = core as u8; e.bytes = 1000 * b1.saturating_sub(*b0) / all;
                    trace::record(e);
                }
            }
            // the engine thread: core and CPU time (ticks of 10 ms) over the same interval
            let tid = trace::engine_tid();
            if tid != 0 && let Some((ticks, core)) = std::fs::read_to_string(format!("/proc/self/task/{tid}/stat")).ok().and_then(|s| parse_task_stat(&s)) {
                if let Some(p) = prev_engine {
                    let all_ticks = ((t0 - t_cpu) / 10_000_000).max(1);
                    let mut e = trace::Event::new(trace::Kind::EngineCpu, t_cpu, t0);
                    e.layer = core as u8; e.bytes = (1000 * ticks.saturating_sub(p) / all_ticks).min(1000);
                    trace::record(e);
                }
                prev_engine = Some(ticks);
            }
            prev = now; t_cpu = t0;
        }
        tick += 1;
        std::thread::sleep(Duration::from_millis(10).saturating_sub(Duration::from_nanos(trace::now_ns() - t0)));
    }
}

/// The current GPU clock from `pp_dpm_sclk` ("2: 2600Mhz *": the line with the star).
fn current_mhz(s: &str) -> Option<u64> {
    let l = s.lines().find(|l| l.trim_end().ends_with('*'))?;
    l.split_whitespace().nth(1)?.trim_end_matches("Mhz").trim_end_matches("MHz").parse().ok()
}

/// A /proc/<pid>/task/<tid>/stat line: (utime + stime in ticks, core it last ran on). The fields are counted after the
/// `)` that closes the thread name (the name may contain spaces): utime is the 14th field of the line, stime the 15th,
/// processor the 39th (proc(5)).
fn parse_task_stat(s: &str) -> Option<(u64, u32)> {
    let f: Vec<&str> = s.get(s.rfind(')')? + 2..)?.split_whitespace().collect();
    Some((f.get(11)?.parse::<u64>().ok()? + f.get(12)?.parse::<u64>().ok()?, f.get(36)?.parse().ok()?))
}

/// Per core (`cpuN` lines): (busy ticks, all ticks). Busy = all - idle - iowait.
fn parse_stat(s: &str) -> Vec<(u64, u64)> {
    s.lines().filter(|l| l.starts_with("cpu") && l.as_bytes().get(3).is_some_and(|c| c.is_ascii_digit())).map(|l| {
        let v: Vec<u64> = l.split_whitespace().skip(1).filter_map(|x| x.parse().ok()).collect();
        let all: u64 = v.iter().sum();
        let idle = v.get(3).copied().unwrap_or(0) + v.get(4).copied().unwrap_or(0);
        (all - idle, all)
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn clock_line() {
        assert_eq!(current_mhz("0: 2600Mhz \n1: 1100Mhz \n2: 2600Mhz *\n"), Some(2600));
        assert_eq!(current_mhz("0: 800Mhz *\n1: 2600Mhz \n"), Some(800));
        assert_eq!(current_mhz("0: 800Mhz\n"), None);
    }
    #[test]
    fn task_stat() {
        let s = "4242 (engine (x)) R 1 2 3 4 5 6 7 8 9 10 120 30 0 0 20 0 9 0 100 200 300 1 2 3 4 5 6 7 8 9 10 11 12 13 14 7 0 0\n";   // field 36 after the name = 7 (python check)
        assert_eq!(parse_task_stat(s), Some((150, 7)));
    }
    #[test]
    fn stat_lines() {
        let s = "cpu  10 0 10 100 5 0 0 0 0 0\ncpu0 1 0 2 50 3 0 1 0 0 0\ncpu1 4 0 0 40 0 0 0 0 0 0\nintr 1 2\n";
        assert_eq!(parse_stat(s), vec![(4, 57), (4, 44)]);
    }
}
