//! mechta trace: real-time trace: events with start and end on one clock, kept in a ring buffer
//! allocated once. MECHTA_TRACE=0 (default) off, 1 = submits and CPU phases, 2 = also the attention groups inside the pass.
use std::collections::VecDeque;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// What an event is. The names are what the trace page and the file show (`Kind::name`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum Kind {
    /// CPU: building the command buffer of a submit
    Prep, Submit,
    /// CPU: waiting for the end of a submit (busy-wait or sleeping poll)
    Wait,
    /// GPU: a whole submit, from the first to the last dispatch
    Gpu,
    /// GPU, level 2: the attention groups of one layer
    AttnPre, AttnKvWrite, AttnCore, AttnMerge, AttnPost,
    /// only a mark: the attention groups of the layer end here (never recorded)
    AttnEnd,
    /// CPU: engine and session phases
    Inputs, Sample, Emit, Draft, Restore, PrefixReuse, RootLoad, RootSave, BlockChoice, Send,
    /// samples for context (the server's sampler thread): load of one CPU core in permille (`layer` = core, value in
    /// `bytes`), GPU clock in MHz and GPU busy percent as the driver reports them
    CpuLoad, GpuClock, GpuBusy,
    /// sample: the engine thread as the OS sees it, over the sample interval: core it ran on last (`layer`) and its
    /// CPU time in permille of the interval (`bytes`). Busy-waiting for the GPU counts as CPU time here.
    EngineCpu,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Prep => "prep", Kind::Submit => "submit", Kind::Wait => "wait", Kind::Gpu => "gpu",
            Kind::AttnPre => "attn.pre", Kind::AttnKvWrite => "attn.kv_write", Kind::AttnCore => "attn.core",
            Kind::AttnMerge => "attn.merge", Kind::AttnPost => "attn.post", Kind::AttnEnd => "attn.end",
            Kind::Inputs => "inputs", Kind::Sample => "sample", Kind::Emit => "emit", Kind::Draft => "draft",
            Kind::Restore => "restore", Kind::PrefixReuse => "prefix_reuse", Kind::RootLoad => "root_load",
            Kind::RootSave => "root_save", Kind::BlockChoice => "block_choice", Kind::Send => "send",
            Kind::CpuLoad => "cpu.load", Kind::GpuClock => "gpu.clock", Kind::GpuBusy => "gpu.busy", Kind::EngineCpu => "engine.cpu",
        }
    }
}

/// One event. Times in ns on the trace clock (`now_ns`). `label` = the program it belongs to ("double", "prefill"...),
/// `pos` = context position, `ntok` = tokens of the step, `block` = prefill block size (0 = not a block),
/// `layer` = model layer (255 = none), `bytes` = KV bytes read.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Event { pub kind: Kind, pub label: &'static str, pub thread: u8, pub layer: u8, pub t0: u64, pub t1: u64, pub pos: u32, pub ntok: u32, pub block: u32, pub bytes: u64 }

impl Event {
    /// An event with the fields of the current context (`set_ctx`).
    pub fn new(kind: Kind, t0: u64, t1: u64) -> Self {
        let c = CTX.with(|c| c.get());
        Event { kind, label: c.label, thread: THREAD.with(|t| t.get()), layer: 255, t0, t1, pos: c.pos, ntok: c.ntok, block: c.block, bytes: 0 }
    }
}

/// What the engine is running now: the submits of `crates/gpu` take their label and position from here, so they
/// do not need new arguments. One per thread (the engine runs on one thread).
/// `kv` = KV positions the attention reads (default pos + ntok); `open` = the attention group a prefill submit starts
/// inside of, when the previous submit was cut in the middle of a layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Ctx { pub label: &'static str, pub pos: u32, pub ntok: u32, pub block: u32, pub kv: u32, pub open: Option<(Kind, u8)> }

thread_local! { static CTX: std::cell::Cell<Ctx> = const { std::cell::Cell::new(Ctx { label: "", pos: 0, ntok: 0, block: 0, kv: 0, open: None }) }; }

/// The thread an event comes from (`Event::thread`): set once at the start of each thread that records.
pub const ENGINE: u8 = 0;
pub const HTTP: u8 = 1;
pub const ROOT_SAVER: u8 = 2;
pub const SAMPLER: u8 = 3;
thread_local! { static THREAD: std::cell::Cell<u8> = const { std::cell::Cell::new(ENGINE) }; }
pub fn set_thread(id: u8) { THREAD.with(|t| t.set(id)); }

static ENGINE_TID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
/// Called by the engine thread at its start: remembers its Linux thread id (from /proc/thread-self), so the sampler
/// can read the core it runs on and its CPU time.
pub fn mark_engine_thread() {
    let tid = std::fs::read_link("/proc/thread-self").ok().and_then(|p| p.file_name()?.to_str()?.parse().ok()).unwrap_or(0);
    ENGINE_TID.store(tid, std::sync::atomic::Ordering::Relaxed);
}
/// The engine thread id, 0 if unknown.
pub fn engine_tid() -> u32 { ENGINE_TID.load(std::sync::atomic::Ordering::Relaxed) }

/// Sets the context of the next events (cheap: a thread-local write; done also with the trace off).
pub fn set_ctx(label: &'static str, pos: usize, ntok: usize, block: usize) {
    CTX.with(|c| c.set(Ctx { label, pos: pos as u32, ntok: ntok as u32, block: block as u32, kv: (pos + ntok) as u32, open: None }));
}
/// KV positions read by the attention of the current context, when they are not pos + ntok (MTP: from `mtp_start`).
pub fn set_kv(n: usize) { CTX.with(|c| { let mut x = c.get(); x.kv = n as u32; c.set(x); }); }
/// The attention group the next submit starts inside of (None = outside any).
pub fn set_open(m: Option<(Kind, u8)>) { CTX.with(|c| { let mut x = c.get(); x.open = m; c.set(x); }); }
pub fn ctx() -> Ctx { CTX.with(|c| c.get()) }

static KV_BYTES_PER_POS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Bytes of K + V of one position in one attention layer (set by the engine from the KV format).
pub fn set_kv_bytes_per_pos(b: u64) { KV_BYTES_PER_POS.store(b, std::sync::atomic::Ordering::Relaxed); }
pub fn kv_bytes_per_pos() -> u64 { KV_BYTES_PER_POS.load(std::sync::atomic::Ordering::Relaxed) }

/// Fixed-size ring: `push` overwrites the oldest event when full. `seq` counts every event ever pushed,
/// so a reader asks "what came after seq N" and knows how many it lost.
pub struct Ring { buf: Vec<Event>, seq: u64 }

impl Ring {
    pub fn new(cap: usize) -> Self { Ring { buf: Vec::with_capacity(cap.max(1)), seq: 0 } }
    pub fn push(&mut self, e: Event) {
        let cap = self.buf.capacity();
        if self.buf.len() < cap { self.buf.push(e); } else { self.buf[(self.seq % cap as u64) as usize] = e; }
        self.seq += 1;
    }
    /// The events with sequence number >= `from`, oldest first; the next `from`; how many were lost (overwritten).
    pub fn since(&self, from: u64) -> (Vec<Event>, u64, u64) {
        let cap = self.buf.capacity() as u64;
        let oldest = self.seq.saturating_sub(cap);
        let start = from.max(oldest).min(self.seq);
        let lost = oldest.saturating_sub(from);
        let v = (start..self.seq).map(|s| self.buf[(s % cap) as usize]).collect();
        (v, self.seq, lost)
    }
}

/// GPU ticks -> trace clock. The offset is the minimum of (CPU time when the wait ended - GPU end) over the last
/// `window` submits: the GPU end is always before the CPU sees it, so the minimum is the tightest estimate, and
/// the window lets the offset follow a slow drift between the two clocks in both directions.
pub struct ClockAlign { last: VecDeque<i128>, window: usize }

impl ClockAlign {
    pub fn new(window: usize) -> Self { ClockAlign { last: VecDeque::new(), window: window.max(1) } }
    /// `cpu_done`: trace clock (ns) when the wait ended; `gpu_end`: GPU time (ns) of the last timestamp of the submit.
    pub fn update(&mut self, cpu_done: u64, gpu_end: u64) {
        if self.last.len() == self.window { self.last.pop_front(); }
        self.last.push_back(cpu_done as i128 - gpu_end as i128);
    }
    pub fn offset(&self) -> Option<i128> { self.last.iter().copied().min() }
    /// A GPU time (ns) on the trace clock; None before the first submit.
    pub fn to_cpu(&self, gpu_ns: u64) -> Option<u64> { self.offset().map(|o| (gpu_ns as i128 + o).max(0) as u64) }
}

/// MECHTA_TRACE: 0 off (default), 1 submits and CPU phases, 2 also the attention groups. Read once.
pub fn level() -> u8 {
    static L: OnceLock<u8> = OnceLock::new();
    *L.get_or_init(|| std::env::var("MECHTA_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(0).min(2))
}
pub fn on() -> bool { level() > 0 }

/// The zero of the trace clock, and the wall clock at that moment (written in the file header, to align the trace with
/// samples taken outside the process).
fn epoch() -> &'static (Instant, std::time::SystemTime) {
    static T0: OnceLock<(Instant, std::time::SystemTime)> = OnceLock::new();
    T0.get_or_init(|| (Instant::now(), std::time::SystemTime::now()))
}
/// Trace clock: ns since the first call in the process (monotonic).
pub fn now_ns() -> u64 { epoch().0.elapsed().as_nanos() as u64 }

/// Events kept in memory: 1M x ~48 bytes. MECHTA_TRACE_EVENTS changes it.
fn capacity() -> usize { std::env::var("MECHTA_TRACE_EVENTS").ok().and_then(|v| v.parse().ok()).unwrap_or(1 << 20) }

/// The process-wide ring, clock alignment and file writer. Allocated only when the trace is on.
pub struct Trace { pub ring: Mutex<Ring>, pub clock: Mutex<ClockAlign>, file: Option<Mutex<FileOut>> }

pub fn global() -> Option<&'static Trace> {
    static G: OnceLock<Option<Trace>> = OnceLock::new();
    let g = G.get_or_init(|| on().then(|| Trace { ring: Mutex::new(Ring::new(capacity())), clock: Mutex::new(ClockAlign::new(64)), file: FileOut::open() })).as_ref();
    // the writer thread starts after the trace exists (it reads it through `global`)
    static W: OnceLock<()> = OnceLock::new();
    if let Some(t) = g && t.file.is_some() { W.get_or_init(|| { std::thread::spawn(|| loop { std::thread::sleep(std::time::Duration::from_millis(100)); flush(); }); }); }
    g
}

/// The events go to a JSON-lines file: MECHTA_TRACE_FILE, or `trace-<UTC date>.jsonl` in the working directory
/// (MECHTA_TRACE_FILE=none: no file). A thread appends the new events every 100 ms; the engine never waits for the disk.
struct FileOut { w: std::io::BufWriter<std::fs::File>, next: u64, lost: u64 }

impl FileOut {
    fn open() -> Option<Mutex<FileOut>> {
        let path = match std::env::var("MECHTA_TRACE_FILE") {
            Ok(p) if p == "none" => return None,
            Ok(p) if !p.is_empty() => p,
            _ => { let (y, mo, d, h, mi, s) = utc(epoch().1.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)); format!("trace-{y:04}{mo:02}{d:02}-{h:02}{mi:02}{s:02}.jsonl") }
        };
        let f = match std::fs::File::create(&path) { Ok(f) => f, Err(e) => { eprintln!("trace: file {path} not created: {e}"); return None; } };
        let mut w = std::io::BufWriter::new(f);
        let wall = epoch().1.duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        use std::io::Write;
        let _ = writeln!(w, r#"{{"header":1,"clock":"ns since start","start_unix_ns":{wall},"level":{},"kv_bytes_per_pos":{}}}"#, level(), kv_bytes_per_pos());
        eprintln!("trace: level {} -> {path}", level());
        Some(Mutex::new(FileOut { w, next: 0, lost: 0 }))
    }
}

/// Writes to the file the events recorded since the last call (also called by the writer thread every 100 ms).
pub fn flush() {
    use std::io::Write;
    let Some(t) = global() else { return };
    let Some(f) = &t.file else { return };
    let mut f = f.lock().unwrap();
    let (v, next, lost) = t.ring.lock().unwrap().since(f.next);
    f.next = next;
    if lost > 0 { f.lost += lost; let _ = writeln!(f.w, r#"{{"lost":{lost}}}"#); }
    for e in v { let _ = writeln!(f.w, "{}", e.json()); }
    let _ = f.w.flush();
}

impl Event {
    /// One line of the trace file. Labels and kind names are fixed ASCII words: no escaping needed.
    pub fn json(&self) -> String {
        format!(r#"{{"k":"{}","l":"{}","th":{},"ly":{},"t0":{},"t1":{},"pos":{},"n":{},"b":{},"by":{}}}"#,
            self.kind.name(), self.label, self.thread, self.layer, self.t0, self.t1, self.pos, self.ntok, self.block, self.bytes)
    }
}

/// UTC calendar date of a Unix time in seconds (days-to-civil, H. Hinnant's algorithm).
fn utc(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    let (days, rem) = ((secs / 86400) as i64, secs % 86400);
    let z = days + 719468; let era = z.div_euclid(146097); let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; let mo = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = yoe + era * 400 + if mo <= 2 { 1 } else { 0 };
    (y, mo, d, (rem / 3600) as u32, (rem / 60 % 60) as u32, (rem % 60) as u32)
}

/// Records an event (nothing if the trace is off).
pub fn record(e: Event) {
    if let Some(t) = global() { t.ring.lock().unwrap().push(e); }
}

/// Start of a CPU span: now on the trace clock, or 0 with the trace off (no clock read).
pub fn start() -> u64 { if on() { now_ns() } else { 0 } }

/// Records a CPU span of `kind` from `t0` (trace clock) to now, with the given fields applied by `f`.
pub fn span(kind: Kind, t0: u64, f: impl FnOnce(&mut Event)) {
    if !on() { return; }
    let mut e = Event::new(kind, t0, now_ns());
    f(&mut e);
    record(e);
}

/// The current end of the ring (sequence number of the next event); 0 with the trace off.
pub fn mark() -> u64 { global().map(|t| t.ring.lock().unwrap().seq).unwrap_or(0) }

/// One line per (label, kind) of the events after `from`: count and total ms; plus the GPU idle time between
/// consecutive GPU spans. For the server log at the end of a request.
pub fn summary(from: u64) -> Option<String> {
    let t = global()?;
    let (v, _, lost) = t.ring.lock().unwrap().since(from);
    if v.is_empty() { return None; }
    let mut tot: std::collections::BTreeMap<(&str, &str), (usize, f64)> = std::collections::BTreeMap::new();
    for e in &v { let x = tot.entry((e.label, e.kind.name())).or_default(); x.0 += 1; x.1 += (e.t1 - e.t0) as f64 * 1e-6; }
    let mut gpu: Vec<(u64, u64)> = v.iter().filter(|e| e.kind == Kind::Gpu).map(|e| (e.t0, e.t1)).collect();
    gpu.sort_unstable();
    let busy: f64 = gpu.iter().map(|g| (g.1 - g.0) as f64).sum::<f64>() * 1e-6;
    let wall = gpu.last().zip(gpu.first()).map(|(l, f)| (l.1 - f.0) as f64 * 1e-6).unwrap_or(0.0);
    let mut s = format!("trace: {} events ({lost} lost) · GPU busy {busy:.0} of {wall:.0} ms between the first and the last submit", v.len());
    for ((l, k), (n, ms)) in &tot { s.push_str(&format!("\n  {l:>10} {k:<13} {n:7} x {ms:10.1} ms")); }
    // attention share of the GPU time per program, and the KV read speed of attn.core
    let attn = [Kind::AttnPre, Kind::AttnKvWrite, Kind::AttnCore, Kind::AttnMerge, Kind::AttnPost];
    let mut per: std::collections::BTreeMap<&str, (f64, f64, f64, u64)> = std::collections::BTreeMap::new();   // gpu, attention, core ms, core bytes
    for e in &v {
        let x = per.entry(e.label).or_default(); let ms = (e.t1 - e.t0) as f64 * 1e-6;
        if e.kind == Kind::Gpu { x.0 += ms; }
        if attn.contains(&e.kind) { x.1 += ms; }
        if e.kind == Kind::AttnCore { x.2 += ms; x.3 += e.bytes; }
    }
    for (l, (g, a, c, b)) in per.into_iter().filter(|p| p.1.1 > 0.0) {
        s.push_str(&format!("\n  {l:>10} attention {a:.1} of {g:.1} GPU ms ({:.0}%) · attn.core reads KV at {:.0} GB/s", 100.0 * a / g.max(1e-9), b as f64 / (c * 1e6).max(1e-9)));
    }
    Some(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(i: u64) -> Event { Event::new(Kind::Gpu, i, i + 1) }

    #[test]
    fn utc_dates() {
        assert_eq!(utc(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(utc(951_782_400), (2000, 2, 29, 0, 0, 0));          // leap day
        assert_eq!(utc(1_793_026_799), (2026, 10, 26, 14, 59, 59));   // python: datetime.fromtimestamp(1793026799, utc)
    }

    #[test]
    fn json_line() {
        let mut e = Event::new(Kind::AttnCore, 10, 25);
        e.layer = 3; e.bytes = 2048;
        assert_eq!(e.json(), r#"{"k":"attn.core","l":"","th":0,"ly":3,"t0":10,"t1":25,"pos":0,"n":0,"b":0,"by":2048}"#);
    }

    #[test]
    fn ring_keeps_order_and_counts_losses() {
        let mut r = Ring::new(4);
        for i in 0..3 { r.push(ev(i)); }
        let (v, next, lost) = r.since(0);
        assert_eq!(v.iter().map(|e| e.t0).collect::<Vec<_>>(), vec![0, 1, 2]);
        assert_eq!((next, lost), (3, 0));
        for i in 3..10 { r.push(ev(i)); }
        // capacity 4: only 6..9 are left; a reader at 3 lost 3, 4, 5
        let (v, next, lost) = r.since(3);
        assert_eq!(v.iter().map(|e| e.t0).collect::<Vec<_>>(), vec![6, 7, 8, 9]);
        assert_eq!((next, lost), (10, 3));
        let (v, next, lost) = r.since(8);
        assert_eq!(v.iter().map(|e| e.t0).collect::<Vec<_>>(), vec![8, 9]);
        assert_eq!((next, lost), (10, 0));
        let (v, _, _) = r.since(10);
        assert!(v.is_empty());
    }

    /// Constant offset, notice latency between 20 and 500 us: the estimate is at most the smallest latency late.
    #[test]
    fn clock_offset_constant() {
        let mut c = ClockAlign::new(64);
        assert_eq!(c.to_cpu(5), None);
        let true_off: i128 = 1_000_000_000;
        let mut state = 12345u64;
        let mut min_lat = u64::MAX;
        for k in 0..200u64 {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let lat = 20_000 + (state >> 33) % 480_000;
            if k >= 136 { min_lat = min_lat.min(lat); }
            let gpu_end = 5_000_000 + k * 60_000_000;
            c.update((gpu_end as i128 + true_off) as u64 + lat, gpu_end);
        }
        let err = c.offset().unwrap() - true_off;
        assert!(err >= 0 && err as u64 == min_lat, "error {err} ns, smallest latency {min_lat} ns");
    }

    /// The GPU clock runs 50 ppm slow: with a window the offset follows; the error stays below the latency + drift of a window.
    #[test]
    fn clock_offset_follows_drift() {
        let mut c = ClockAlign::new(64);
        let lat = 30_000u64;
        let mut last_true = 0i128;
        for k in 0..2000u64 {
            let cpu_end = 1_000_000_000 + k * 60_000_000;
            let gpu_end = (cpu_end as f64 * (1.0 - 50e-6)) as u64;
            last_true = cpu_end as i128 - gpu_end as i128;
            c.update(cpu_end + lat, gpu_end);
        }
        let err = c.offset().unwrap() - last_true;
        // 64 submits x 60 ms x 50 ppm = 192 us of drift inside the window: the minimum is the oldest offset of the
        // window, so the estimate lags behind by up to that drift (negative error), or is late by the latency (positive)
        assert!(err.abs() <= lat as i128 + 200_000, "error {err} ns");
        // without the window (all 2000 submits) the minimum would stay at the first offset, ~6 ms away
        let mut all = ClockAlign::new(4000);
        for k in 0..2000u64 {
            let cpu_end = 1_000_000_000 + k * 60_000_000;
            all.update(cpu_end + lat, (cpu_end as f64 * (1.0 - 50e-6)) as u64);
        }
        assert!(all.offset().unwrap() - last_true < -5_000_000);
    }
}
