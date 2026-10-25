//! mechta-chat: a terminal chat with the model of the mechta server (127.0.0.1:4131), without opencode.
//!
//! Keys: Enter send · Alt+Enter new line · Esc stop the answer · Ctrl+T show / fold the thinking · wheel, ↑/↓, PgUp/PgDn,
//! Home/End scroll · Ctrl+Y copy the last answer · Ctrl+C quit. The two copy icons on the input box copy the last answer
//! / question (the mouse is captured for the wheel and the icons: Shift+drag selects text in the terminal).
//! Commands: /new · /system <text> · /think on|off · /copy [q] · /help · /quit.
//! Every chat is saved after each answer in ~/.local/share/mechta/chats/chat-<start time>.json.
//! The system prompt stays short (under 2048 tokens), so the server keeps opencode's saved root untouched; a long
//! one (/system) gets a saved root of its own on the server (one per system prompt).
mod client;
mod md;
mod locales { pub mod en; pub mod it; }

use client::Event;
use crossterm::event::{self, Event as TermEvent, KeyCode, KeyEventKind, KeyModifiers, MouseButton, MouseEventKind};
use std::cell::{Cell, RefCell};
use ratatui::layout::{Constraint, Layout, Position};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::{DefaultTerminal, Frame};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::time::Duration;
use unicode_width::UnicodeWidthChar;

#[derive(Clone, Copy, PartialEq)]
enum Role { User, Assistant, Note }

struct Msg { role: Role, content: String, reasoning: String }

#[derive(PartialEq)]
enum Phase { Idle, Waiting, Generating }

struct App {
    t: fn(&str) -> &'static str,
    system: String,
    msgs: Vec<Msg>,
    input: String,
    thinking: bool,
    show_thinking: bool,
    /// context length of the server (from /v1/models) and tokens of the conversation after the last answer
    ctx_max: u64,
    ctx_used: u64,
    phase: Phase,
    stream: Option<(Receiver<Event>, Arc<AtomicBool>)>,
    status: String,
    /// scroll: None = follow the end of the chat; Some(top) = first line shown, fixed while the answer grows
    anchor: Option<usize>,
    /// set while drawing: lines of the chat and height of its area; the copy icons (row, from x, to x)
    view: Cell<(usize, usize)>,
    icons: Cell<[(u16, u16, u16); 2]>,
    /// set while drawing: chat area (x, y, w, h), first line shown, plain text of every chat line (for the selection)
    chat_rect: Cell<(u16, u16, u16, u16)>,
    top_line: Cell<usize>,
    plain: RefCell<Vec<String>>,
    /// selection made with the mouse inside the chat: (anchor, point) as (line, column); copied when the button is released
    sel: Option<((usize, usize), (usize, usize))>,
    dragging: bool,
    /// while dragging past the top (-1) or bottom (+1) edge of the chat: scroll on every tick
    edge: isize,
    started: u64,
    quit: bool,
}

fn now_secs() -> u64 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) }

impl App {
    fn new() -> Self {
        let lang = ["LC_ALL", "LC_MESSAGES", "LANG"].iter().filter_map(|k| std::env::var(k).ok()).find(|v| !v.is_empty()).unwrap_or_default();
        let t: fn(&str) -> &'static str = if lang.starts_with("it") { locales::it::text } else { locales::en::text };
        // thinking: the server default, unless --think on|off is given; /think changes it
        let info = client::server_info();
        let status = if info.is_some() { t("ready").to_string() } else { t("offline").to_string() };
        let arg = std::env::args().skip_while(|a| a != "--think").nth(1);
        let thinking = match arg.as_deref() { Some("off") => false, Some("on") => true, _ => info.map(|i| i.1).unwrap_or(true) };
        App { t, system: t("default_system").to_string(), msgs: vec![], input: String::new(), thinking, show_thinking: false,
              ctx_max: info.map(|i| i.0).unwrap_or(0), ctx_used: 0, phase: Phase::Idle, stream: None, status, anchor: None,
              view: Cell::new((0, 0)), icons: Cell::new([(u16::MAX, 0, 0); 2]),
              chat_rect: Cell::new((0, 0, 0, 0)), top_line: Cell::new(0), plain: RefCell::new(vec![]), sel: None, dragging: false, edge: 0,
              started: now_secs(), quit: false }
    }

    fn note(&mut self, s: String) { self.msgs.push(Msg { role: Role::Note, content: s, reasoning: String::new() }); self.anchor = None; }

    fn new_chat(&mut self) { self.cancel(); self.msgs.clear(); self.started = now_secs(); self.anchor = None; self.ctx_used = 0; }

    /// Moves the view by `delta` lines (negative = up). Back at the end, the view follows the chat again.
    fn scroll_by(&mut self, delta: isize) {
        let (total, h) = self.view.get();
        let max_top = total.saturating_sub(h);
        let top = self.anchor.unwrap_or(max_top) as isize + delta;
        let top = top.clamp(0, max_top as isize) as usize;
        self.anchor = if top >= max_top { None } else { Some(top) };
    }

    /// Copies the last answer (Role::Assistant) or question (Role::User) to the clipboard with wl-copy.
    fn copy_last(&mut self, role: Role) {
        let text = self.msgs.iter().rev().find(|m| m.role == role && !m.content.trim().is_empty()).map(|m| m.content.trim().to_string());
        let msg = match text {
            None => (self.t)("nothing_to_copy").to_string(),
            Some(t) => match copy_to_clipboard(&t) {
                Ok(()) => (self.t)(if role == Role::Assistant { "copied_answer" } else { "copied_question" }).to_string(),
                Err(e) => format!("{}: {e}", (self.t)("copy_failed")),
            },
        };
        self.status = msg;
    }

    fn mouse(&mut self, kind: MouseEventKind, col: u16, row: u16) {
        let (cx, cy, cw, ch) = self.chat_rect.get();
        let inside = col >= cx && col < cx + cw && row >= cy && row < cy + ch;
        match kind {
            MouseEventKind::ScrollUp => self.scroll_by(-3),
            MouseEventKind::ScrollDown => self.scroll_by(3),
            MouseEventKind::Down(MouseButton::Left) => {
                let [a, q] = self.icons.get();
                if row == a.0 && (a.1..a.2).contains(&col) { self.copy_last(Role::Assistant); return; }
                if row == q.0 && (q.1..q.2).contains(&col) { self.copy_last(Role::User); return; }
                self.sel = None;
                if inside { let p = self.point(col, row); self.sel = Some((p, p)); self.dragging = true; self.edge = 0; }
            }
            MouseEventKind::Drag(MouseButton::Left) if self.dragging => {
                // past the edges of the chat: keep scrolling while the button is held there (see `tick`)
                self.edge = if row <= cy { -1 } else if row >= cy + ch.saturating_sub(1) { 1 } else { 0 };
                let p = self.point(col, row.clamp(cy, cy + ch.saturating_sub(1)));
                if let Some((a, _)) = self.sel { self.sel = Some((a, p)); }
            }
            MouseEventKind::Up(MouseButton::Left) if self.dragging => {
                self.dragging = false; self.edge = 0;
                match self.sel { Some((a, b)) if a != b => self.copy_selection(), _ => self.sel = None }
            }
            _ => {}
        }
    }

    /// Screen cell → (line of the whole chat, column)
    fn point(&self, col: u16, row: u16) -> (usize, usize) {
        let (cx, cy, _, _) = self.chat_rect.get();
        let total = self.plain.borrow().len();
        let line = (self.top_line.get() + row.saturating_sub(cy) as usize).min(total.saturating_sub(1));
        (line, col.saturating_sub(cx) as usize)
    }

    /// Called on every loop turn: while dragging at an edge, scroll one line and extend the selection to it.
    fn tick(&mut self) {
        if !self.dragging || self.edge == 0 { return; }
        self.scroll_by(self.edge);
        let (_, cy, _, ch) = self.chat_rect.get();
        let (total, h) = self.view.get();
        let max_top = total.saturating_sub(h);
        let top = self.anchor.map(|a| a.min(max_top)).unwrap_or(max_top);
        self.top_line.set(top);
        let row = if self.edge < 0 { cy } else { cy + ch.saturating_sub(1) };
        if let Some((a, b)) = self.sel { let p = self.point(0, row); self.sel = Some((a, (p.0, if self.edge < 0 { 0 } else { b.1.max(usize::MAX / 2) }))); }
    }

    /// The selected text (as drawn: Markdown already formatted), one line per chat line.
    fn selected_text(&self) -> String {
        let Some((a, b)) = self.sel else { return String::new() };
        let (s, e) = if a <= b { (a, b) } else { (b, a) };
        let plain = self.plain.borrow();
        let mut out = vec![];
        for li in s.0..=e.0.min(plain.len().saturating_sub(1)) {
            let line = &plain[li];
            let from = if li == s.0 { s.1 } else { 0 };
            let to = if li == e.0 { e.1 } else { usize::MAX };
            out.push(slice_cols(line, from, to).trim_end().to_string());
        }
        out.join("\n")
    }

    fn copy_selection(&mut self) {
        let text = self.selected_text();
        self.status = match copy_to_clipboard(&text) {
            Ok(()) => (self.t)("copied_selection").replace("{n}", &text.chars().count().to_string()),
            Err(e) => format!("{}: {e}", (self.t)("copy_failed")),
        };
    }

    fn cancel(&mut self) { if let Some((_, c)) = &self.stream { c.store(true, Ordering::Relaxed); } }

    fn submit(&mut self) {
        let text = self.input.trim().to_string();
        if text.is_empty() || self.phase != Phase::Idle { return; }
        self.input.clear();
        if let Some(cmd) = text.strip_prefix('/') {
            let (name, arg) = cmd.split_once(' ').map(|(a, b)| (a, b.trim())).unwrap_or((cmd, ""));
            match name {
                "new" => { self.new_chat(); let s = (self.t)("new_chat").to_string(); self.note(s); }
                "system" if !arg.is_empty() => { self.new_chat(); self.system = arg.to_string(); let s = format!("{}: {}", (self.t)("system_set"), arg); self.note(s); }
                "copy" => self.copy_last(if arg.starts_with('q') || arg.starts_with('d') { Role::User } else { Role::Assistant }),
                "think" => { self.thinking = arg != "off"; let s = (self.t)(if self.thinking { "think_on" } else { "think_off" }).to_string(); self.note(s); }
                "quit" | "exit" => self.quit = true,
                _ => { let s = (self.t)("commands").to_string(); self.note(s); }
            }
            return;
        }
        self.msgs.push(Msg { role: Role::User, content: text, reasoning: String::new() });
        self.msgs.push(Msg { role: Role::Assistant, content: String::new(), reasoning: String::new() });
        let mut history = vec![serde_json::json!({"role": "system", "content": self.system})];
        for m in &self.msgs[..self.msgs.len() - 1] {
            match m.role {
                Role::User => history.push(serde_json::json!({"role": "user", "content": m.content})),
                Role::Assistant => history.push(serde_json::json!({"role": "assistant", "content": m.content})),
                Role::Note => {}
            }
        }
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let (c2, thinking) = (cancel.clone(), self.thinking);
        std::thread::spawn(move || client::stream_chat(serde_json::Value::Array(history), thinking, tx, c2));
        self.stream = Some((rx, cancel));
        self.phase = Phase::Waiting;
        self.status = (self.t)("waiting").to_string();
        self.anchor = None;
    }

    // Pieces of the answer arrived from the client thread.
    fn pump(&mut self) {
        let Some((rx, _)) = &self.stream else { return };
        let mut done = None;
        while let Ok(ev) = rx.try_recv() {
            let last = self.msgs.last_mut().expect("assistant message");
            match ev {
                Event::Reasoning(s) => { last.reasoning.push_str(&s); self.phase = Phase::Generating; }
                Event::Content(s) => { last.content.push_str(&s); self.phase = Phase::Generating; }
                Event::Done { stopped } => done = Some(Ok(stopped)),
                Event::Error(e) => done = Some(Err(e)),
            }
        }
        if self.phase == Phase::Generating && done.is_none() { self.status = (self.t)("generating").to_string(); }
        match done {
            None => {}
            Some(r) => {
                self.stream = None;
                self.phase = Phase::Idle;
                match r {
                    Ok(stopped) => {
                        if let Some(n) = client::last_tokens() { self.ctx_used = n; }
                        self.status = match client::last_speeds() {
                            Some((p, n, g, k)) => (self.t)("speed").replace("{p}", &format!("{p:.0}")).replace("{n}", &n.to_string())
                                .replace("{g}", &format!("{g:.1}")).replace("{k}", &k.to_string()),
                            None => (self.t)("ready").to_string(),
                        };
                        if stopped { let s = (self.t)("stopped").to_string(); self.note(s); }
                    }
                    Err(e) => { let s = format!("{}: {e}", (self.t)("error")); self.status = s.clone(); self.note(s); }
                }
                if let Err(e) = self.save() { let s = format!("{}: {e}", (self.t)("error")); self.note(s); }
            }
        }
    }

    fn save(&self) -> std::io::Result<()> {
        let home = std::env::var_os("HOME").ok_or_else(|| std::io::Error::other("no HOME"))?;
        let dir = std::path::Path::new(&home).join(".local/share/mechta/chats");
        std::fs::create_dir_all(&dir)?;
        let msgs: Vec<serde_json::Value> = self.msgs.iter().filter(|m| m.role != Role::Note).map(|m| serde_json::json!({
            "role": if m.role == Role::User { "user" } else { "assistant" }, "content": m.content, "reasoning": m.reasoning })).collect();
        let v = serde_json::json!({ "started": self.started, "system": self.system, "messages": msgs });
        std::fs::write(dir.join(format!("chat-{}.json", self.started)), serde_json::to_string_pretty(&v).unwrap_or_default())
    }

    fn key(&mut self, code: KeyCode, mods: KeyModifiers) {
        let ctrl = mods.contains(KeyModifiers::CONTROL);
        match code {
            KeyCode::Char('c') if ctrl => { self.cancel(); self.quit = true; }
            KeyCode::Char('t') if ctrl => self.show_thinking = !self.show_thinking,
            KeyCode::Enter if mods.contains(KeyModifiers::ALT) => self.input.push('\n'),
            KeyCode::Enter => self.submit(),
            KeyCode::Esc if self.sel.is_some() => self.sel = None,
            KeyCode::Esc => self.cancel(),
            KeyCode::Backspace => { self.input.pop(); }
            KeyCode::Char('y') if ctrl => self.copy_last(Role::Assistant),
            KeyCode::Up => self.scroll_by(-1),
            KeyCode::Down => self.scroll_by(1),
            KeyCode::PageUp => { let h = self.view.get().1 as isize; self.scroll_by(-(h - 2).max(1)) }
            KeyCode::PageDown => { let h = self.view.get().1 as isize; self.scroll_by((h - 2).max(1)) }
            KeyCode::Home => self.anchor = Some(0),
            KeyCode::End => self.anchor = None,
            KeyCode::Char(c) if !ctrl => self.input.push(c),
            _ => {}
        }
    }
}

/// The text into the Wayland clipboard (wl-copy reads it from stdin).
fn copy_to_clipboard(text: &str) -> std::io::Result<()> {
    use std::io::Write;
    // tests: MECHTA_CHAT_CLIPBOARD_FILE gets the text instead of the system clipboard
    if let Ok(f) = std::env::var("MECHTA_CHAT_CLIPBOARD_FILE") { return std::fs::write(f, text); }
    let mut child = std::process::Command::new("wl-copy").stdin(std::process::Stdio::piped()).spawn()?;
    child.stdin.take().ok_or_else(|| std::io::Error::other("no stdin"))?.write_all(text.as_bytes())?;
    let st = child.wait()?;
    if st.success() { Ok(()) } else { Err(std::io::Error::other(format!("wl-copy: {st}"))) }
}

/// The part of `s` between display columns `from` (included) and `to` (included).
fn slice_cols(s: &str, from: usize, to: usize) -> String {
    let mut col = 0; let mut out = String::new();
    for c in s.chars() {
        let cw = c.width().unwrap_or(0);
        if col >= from && col <= to { out.push(c); }
        col += cw;
        if col > to { break; }
    }
    out
}

/// Wraps text at `width` columns, breaking at the last space when there is one.
fn wrap(text: &str, width: usize) -> Vec<String> { wrap_opts(text, width, true) }
/// The same for the input box: the spaces at the end of the last line stay, so the cursor moves when a space is typed.
fn wrap_input(text: &str, width: usize) -> Vec<String> { wrap_opts(text, width, false) }
fn wrap_opts(text: &str, width: usize, trim_last: bool) -> Vec<String> {
    let width = width.max(8);
    let mut out = vec![];
    for para in text.split('\n') {
        let mut line = String::new(); let mut w = 0;
        for word in para.split_inclusive(' ') {
            let ww: usize = word.chars().map(|c| c.width().unwrap_or(0)).sum();
            if w + ww > width && !line.is_empty() { out.push(std::mem::take(&mut line).trim_end().to_string()); w = 0; }
            if ww > width {   // a word longer than the line: cut it
                for c in word.chars() {
                    let cw = c.width().unwrap_or(0);
                    if w + cw > width { out.push(std::mem::take(&mut line)); w = 0; }
                    line.push(c); w += cw;
                }
            } else { line.push_str(word); w += ww; }
        }
        out.push(if trim_last { line.trim_end().to_string() } else { line });
    }
    out
}

/// "context 23% (15.1K / 65.5K)": tokens of the conversation after the last answer over the server context.
fn context_text(t: fn(&str) -> &'static str, used: u64, max: u64) -> String {
    let k = |n: u64| if n >= 1000 { format!("{:.1}K", n as f64 / 1000.0) } else { n.to_string() };
    if max == 0 { return format!("{} ?", t("context")); }
    format!("{} {}% ({} / {})", t("context"), (used * 100 + max / 2) / max, k(used), k(max))
}

/// Column of the cursor on the last line of the wrapped input.
fn input_cursor_col(input: &str, width: usize) -> usize {
    let last = wrap_input(input, width).pop().unwrap_or_default();
    last.chars().map(|c| c.width().unwrap_or(0)).sum::<usize>()
}

fn render(app: &App, f: &mut Frame) {
    let input_lines = wrap_input(&app.input, (f.area().width as usize).saturating_sub(4)).len().clamp(1, 6) as u16;
    let [chat, input, status] = Layout::vertical([Constraint::Min(3), Constraint::Length(input_lines + 2), Constraint::Length(2)]).areas(f.area());
    let width = chat.width as usize;
    let dim = Style::default().fg(Color::DarkGray);
    let mut lines: Vec<Line> = vec![];
    for m in &app.msgs {
        match m.role {
            Role::Note => { for l in wrap(&format!("· {}", m.content), width) { lines.push(Line::styled(l, dim)); } }
            Role::User | Role::Assistant => {
                let (who, color) = if m.role == Role::User { ((app.t)("you"), Color::Cyan) } else { ((app.t)("model"), Color::Magenta) };
                lines.push(Line::from(Span::styled(who, Style::default().fg(color).add_modifier(Modifier::BOLD))));
                if !m.reasoning.is_empty() {
                    if app.show_thinking {
                        lines.push(Line::styled(format!("▾ {}", (app.t)("thinking")), dim));
                        for l in wrap(m.reasoning.trim(), width.saturating_sub(2)) { lines.push(Line::styled(format!("  {l}"), dim.add_modifier(Modifier::ITALIC))); }
                    } else {
                        lines.push(Line::styled(format!("▸ {}", (app.t)("thinking_folded")), dim));
                    }
                }
                // answers: Markdown drawn for the terminal; questions: as typed
                if m.role == Role::Assistant { lines.extend(md::render(m.content.trim_start(), width)); }
                else { for l in wrap(m.content.trim_start(), width) { lines.push(Line::raw(l)); } }
                lines.push(Line::raw(""));
            }
        }
    }
    // the end of the conversation, or the lines from `anchor` when the view has been scrolled up
    let h = chat.height as usize;
    let total = lines.len();
    app.view.set((total, h));
    let max_top = total.saturating_sub(h);
    let top = app.anchor.map(|a| a.min(max_top)).unwrap_or(max_top);
    app.top_line.set(top);
    app.chat_rect.set((chat.x, chat.y, chat.width, chat.height));
    *app.plain.borrow_mut() = lines.iter().map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect::<String>()).collect();
    let shown: Vec<Line> = lines.into_iter().skip(top).take(h).collect();
    f.render_widget(Paragraph::new(shown), chat);
    // the selection, in reverse video
    if let Some((a, b)) = app.sel {
        let (s, e) = if a <= b { (a, b) } else { (b, a) };
        for row in 0..chat.height as usize {
            let li = top + row;
            if li < s.0 || li > e.0 { continue; }
            let from = if li == s.0 { s.1 } else { 0 };
            let to = if li == e.0 { e.1.min(chat.width as usize - 1) } else { chat.width as usize - 1 };
            for x in from..=to.min(chat.width as usize - 1) {
                let cell = &mut f.buffer_mut()[(chat.x + x as u16, chat.y + row as u16)];
                cell.set_style(Style::default().add_modifier(Modifier::REVERSED));
            }
        }
    }
    if app.anchor.is_some() && top < max_top {
        let mark = format!(" {} ", (app.t)("scrolled"));
        let mw = mark.chars().map(|c| c.width().unwrap_or(0)).sum::<usize>() as u16;
        let area = ratatui::layout::Rect { x: chat.x + chat.width.saturating_sub(mw), y: chat.y + chat.height.saturating_sub(1), width: mw.min(chat.width), height: 1 };
        f.render_widget(Paragraph::new(Line::styled(mark, Style::default().fg(Color::Black).bg(Color::Gray))), area);
    }

    let input_text: Vec<Line> = wrap_input(&app.input, (input.width as usize).saturating_sub(2)).into_iter().map(Line::raw).collect();
    let n = input_text.len();
    let visible: Vec<Line> = input_text.into_iter().skip(n.saturating_sub(input_lines as usize)).collect();
    let title = format!(" {} ", (app.t)("title"));
    // copy icons on the right of the top border: "󰆏 answer  󰆏 question", clickable
    let icon = '\u{F018F}';
    let (ia, iq) = (format!("{icon} {}", (app.t)("copy_answer")), format!("{icon} {}", (app.t)("copy_question")));
    let cw = |s: &str| s.chars().map(|c| c.width().unwrap_or(1)).sum::<usize>() as u16;
    let right_title = Line::from(vec![Span::raw(" "), Span::styled(ia.clone(), Style::default().fg(Color::Gray)), Span::raw("  "),
                                      Span::styled(iq.clone(), Style::default().fg(Color::Gray)), Span::raw(" ")]).right_aligned();
    let end = input.x + input.width.saturating_sub(1);           // the right corner of the border
    let q_from = end.saturating_sub(1 + cw(&iq));
    let a_from = q_from.saturating_sub(2 + cw(&ia));
    app.icons.set([(input.y, a_from, a_from + cw(&ia)), (input.y, q_from, q_from + cw(&iq))]);
    f.render_widget(Paragraph::new(visible).block(Block::default().borders(Borders::ALL).title(title).title_top(right_title)), input);
    // cursor at the end of the input
    let cx = input.x + 1 + input_cursor_col(&app.input, (input.width as usize).saturating_sub(2)) as u16;
    let cy = input.y + 1 + (n.min(input_lines as usize).saturating_sub(1)) as u16;
    f.set_cursor_position(Position::new(cx.min(input.x + input.width.saturating_sub(2)), cy));

    // first status row: the status on the left (cut if too long), thinking and context always whole on the right
    let mode = if app.thinking { (app.t)("think_on") } else { (app.t)("think_off") };
    let right = format!("{} · {}", mode, context_text(app.t, app.ctx_used, app.ctx_max));
    let rw = right.chars().map(|c| c.width().unwrap_or(0)).sum::<usize>() as u16;
    let [row1, row2] = Layout::vertical([Constraint::Length(1), Constraint::Length(1)]).areas(status);
    let [left_area, right_area] = Layout::horizontal([Constraint::Min(1), Constraint::Length(rw + 3)]).areas(row1);
    let gray = Style::default().fg(Color::Gray);
    f.render_widget(Paragraph::new(Line::styled(app.status.clone(), gray)), left_area);
    f.render_widget(Paragraph::new(Line::styled(format!(" · {right}"), gray)).alignment(ratatui::layout::Alignment::Right), right_area);
    // the legend: fainter than the chat (a terminal has one font size, so smaller is not possible)
    f.render_widget(Paragraph::new(Line::styled((app.t)("help"), dim.add_modifier(Modifier::ITALIC | Modifier::DIM))), row2);
}

fn run(terminal: &mut DefaultTerminal) -> std::io::Result<()> {
    let mut app = App::new();
    crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture)?;   // wheel and copy icons
    while !app.quit {
        app.pump();
        app.tick();
        terminal.draw(|f| render(&app, f))?;
        if event::poll(Duration::from_millis(50))? {
            match event::read()? {
                TermEvent::Key(k) if k.kind == KeyEventKind::Press => app.key(k.code, k.modifiers),
                TermEvent::Mouse(m) => app.mouse(m.kind, m.column, m.row),
                _ => {}
            }
        }
    }
    crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture)?;
    // wait for a stopped stream to end, so the server sees the client go away
    while app.stream.is_some() { app.pump(); std::thread::sleep(Duration::from_millis(50)); }
    Ok(())
}

fn main() -> std::io::Result<()> {
    // fixed window title: external tools (a launcher) can find the open chat by it
    crossterm::execute!(std::io::stdout(), crossterm::terminal::SetTitle("mechta-chat"))?;
    ratatui::run(run)
}

#[cfg(test)]
mod tests {
    use super::*;
    // a space typed at the end of the input must move the cursor at once
    #[test]
    fn cursor_moves_after_trailing_space() {
        assert_eq!(input_cursor_col("ciao ", 40), 5);
        assert_eq!(input_cursor_col("ciao", 40), 4);
        assert_eq!(input_cursor_col("ciao  ", 40), 6);
    }
    #[test]
    fn context_line() {
        assert_eq!(context_text(locales::en::text, 15100, 65536), "context 23% (15.1K / 65.5K)");
        assert_eq!(context_text(locales::it::text, 0, 65536), "contesto 0% (0 / 65.5K)");
        assert_eq!(context_text(locales::en::text, 10, 0), "context ?");
    }
    #[test]
    fn scroll_anchor() {
        let mut a = App::new();
        a.view.set((100, 20));                  // 100 lines, 20 visible: the end is top 80
        a.scroll_by(-5); assert_eq!(a.anchor, Some(75));
        a.scroll_by(-100); assert_eq!(a.anchor, Some(0));
        a.scroll_by(200); assert_eq!(a.anchor, None);   // back at the end: follow again
        a.view.set((10, 20)); a.scroll_by(-3); assert_eq!(a.anchor, None);   // shorter than the window: nothing to scroll
    }
    #[test]
    fn selection_text() {
        let mut a = App::new();
        *a.plain.borrow_mut() = vec!["alpha beta".into(), "gamma delta".into(), "epsilon".into()];
        a.sel = Some(((0, 6), (2, 2)));
        assert_eq!(a.selected_text(), "beta\ngamma delta\neps");
        a.sel = Some(((1, 9), (1, 2)));                       // dragged backwards on one line
        assert_eq!(a.selected_text(), "mma delt");
        assert_eq!(slice_cols("ab😀cd", 2, 3), "😀");          // a wide character takes two columns
    }
    #[test]
    fn wrap_keeps_words() {
        assert_eq!(wrap("aa bb cc dd", 8), vec!["aa bb", "cc dd"]);
        assert_eq!(wrap("", 10), vec![""]);
    }
}
