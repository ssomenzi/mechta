//! Markdown of the answers, drawn for the terminal. Input: the raw text of an answer (also while it is
//! still arriving: an unclosed ``` is a code block up to the end, an unclosed ** stays as text). Output: lines already
//! wrapped at `width`, so the chat can count them for scrolling. The raw text is what gets copied, not this.
//!
//! Drawn: headings (bold, coloured: a terminal has one font size), bold, italic, strikethrough, inline code, code
//! blocks (own background, no syntax colours), lists (nested, numbered, task boxes), quotes, links (text, then the
//! address in grey), rules, tables when they fit the width (otherwise one line per row, cells separated by │).
use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;

fn w(s: &str) -> usize { s.chars().map(|c| c.width().unwrap_or(0)).sum() }

const CODE_BG: Color = Color::Rgb(40, 44, 52);

/// The lines of an answer at `width` columns.
pub fn render(text: &str, width: usize) -> Vec<Line<'static>> {
    // the model's text goes to the terminal without control characters (a prompt could make the model
    // write ANSI sequences); newline and tab stay
    let clean: String = text.chars().filter(|&c| !c.is_control() || c == '\n' || c == '\t').collect();
    let text = clean.as_str();
    let mut r = Renderer { width: width.max(10), ..Default::default() };
    let opts = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    for ev in Parser::new_ext(text, opts) { r.event(ev); }
    r.flush();
    if r.in_code { r.end_code(); }   // still streaming: the code block is not closed yet
    while r.out.last().is_some_and(|l| l.width() == 0) { r.out.pop(); }
    r.out
}

#[derive(Default)]
struct Renderer {
    width: usize,
    out: Vec<Line<'static>>,
    /// the paragraph being built: pieces of text with their style
    cur: Vec<(String, Style)>,
    bold: u32, italic: u32, strike: u32, link: u32,
    heading: Option<HeadingLevel>,
    links: Vec<String>,
    /// lists: one entry per level, Some(next number) for numbered lists
    lists: Vec<Option<u64>>,
    /// prefix of the first line of the current item (marker) and of the next lines
    item_first: Option<String>,
    quote: u32,
    in_code: bool,
    code: String,
    table: Option<Table>,
}

#[derive(Default)]
struct Table { rows: Vec<Vec<String>>, head_rows: usize, cell: String, in_head: bool }

impl Renderer {
    fn style(&self) -> Style {
        let mut s = Style::default();
        if self.bold > 0 { s = s.add_modifier(Modifier::BOLD); }
        if self.italic > 0 { s = s.add_modifier(Modifier::ITALIC); }
        if self.strike > 0 { s = s.add_modifier(Modifier::CROSSED_OUT); }
        if self.link > 0 { s = s.fg(Color::Cyan).add_modifier(Modifier::UNDERLINED); }
        if let Some(h) = self.heading {
            s = s.add_modifier(Modifier::BOLD).fg(match h { HeadingLevel::H1 => Color::Magenta, HeadingLevel::H2 => Color::Yellow, _ => Color::Cyan });
            if h == HeadingLevel::H1 { s = s.add_modifier(Modifier::UNDERLINED); }
        }
        s
    }

    /// Indentation of lists and quotes: (prefix of the first line, prefix of the following lines)
    fn prefixes(&mut self) -> (Vec<(String, Style)>, Vec<(String, Style)>) {
        let dim = Style::default().fg(Color::DarkGray);
        let mut first = vec![]; let mut next = vec![];
        for _ in 0..self.quote { first.push(("│ ".to_string(), dim)); next.push(("│ ".to_string(), dim)); }
        let depth = self.lists.len();
        if depth > 0 {
            let pad = "  ".repeat(depth - 1);
            match self.item_first.take() {
                Some(m) => { let n = w(&m); first.push((format!("{pad}{m}"), Style::default().fg(Color::Yellow))); next.push((format!("{pad}{}", " ".repeat(n)), Style::default())); }
                None => { let p = format!("{pad}{}", " ".repeat(2)); first.push((p.clone(), Style::default())); next.push((p, Style::default())); }
            }
        }
        (first, next)
    }

    fn push(&mut self, text: &str, style: Style) {
        if let Some(t) = self.table.as_mut() { t.cell.push_str(text); return; }
        self.cur.push((text.to_string(), style));
    }

    /// Ends the paragraph being built: wraps it under its prefixes.
    fn flush(&mut self) {
        if self.cur.is_empty() { return; }
        let pieces = std::mem::take(&mut self.cur);
        let (first, next) = self.prefixes();
        let lines = wrap_styled(&pieces, self.width, &first, &next);
        self.out.extend(lines);
    }

    fn blank(&mut self) {
        if self.lists.is_empty() && self.out.last().is_some_and(|l| l.width() > 0) { self.out.push(Line::raw("")); }
    }

    fn end_code(&mut self) {
        self.in_code = false;
        let code = std::mem::take(&mut self.code);
        let st = Style::default().bg(CODE_BG).fg(Color::White);
        let (first, _) = self.prefixes();
        let pre_w = first.iter().map(|p| w(&p.0)).sum::<usize>();
        let inner = self.width.saturating_sub(pre_w + 2).max(4);
        for raw in code.trim_end_matches('\n').split('\n') {
            // long lines are cut, not reflowed: code keeps its shape
            let mut chunk = String::new(); let mut cw = 0;
            let mut chunks = vec![];
            for c in raw.replace('\t', "    ").chars() {
                let x = c.width().unwrap_or(0);
                if cw + x > inner { chunks.push(std::mem::take(&mut chunk)); cw = 0; }
                chunk.push(c); cw += x;
            }
            chunks.push(chunk);
            for ch in chunks {
                let pad = inner.saturating_sub(w(&ch));
                let mut spans: Vec<Span<'static>> = first.iter().map(|(t, s)| Span::styled(t.clone(), *s)).collect();
                spans.push(Span::styled(format!(" {ch}{} ", " ".repeat(pad)), st));
                self.out.push(Line::from(spans));
            }
        }
        self.blank();
    }

    fn end_table(&mut self) {
        let Some(t) = self.table.take() else { return };
        let cols = t.rows.iter().map(|r| r.len()).max().unwrap_or(0);
        if cols == 0 { return; }
        let mut widths = vec![0usize; cols];
        for r in &t.rows { for (i, c) in r.iter().enumerate() { widths[i] = widths[i].max(w(c)); } }
        let total: usize = widths.iter().sum::<usize>() + 3 * (cols - 1);
        let dim = Style::default().fg(Color::DarkGray);
        if total <= self.width {
            for (ri, r) in t.rows.iter().enumerate() {
                let mut spans = vec![];
                for i in 0..cols {
                    let c = r.get(i).map(String::as_str).unwrap_or("");
                    if i > 0 { spans.push(Span::styled(" │ ", dim)); }
                    let st = if ri < t.head_rows { Style::default().add_modifier(Modifier::BOLD) } else { Style::default() };
                    spans.push(Span::styled(format!("{c}{}", " ".repeat(widths[i] - w(c))), st));
                }
                self.out.push(Line::from(spans));
                if ri + 1 == t.head_rows {
                    let sep: Vec<String> = widths.iter().map(|&n| "─".repeat(n)).collect();
                    self.out.push(Line::styled(sep.join("─┼─"), dim));
                }
            }
        } else {
            // too wide: one row per line, wrapped, cells separated by │
            for (ri, r) in t.rows.iter().enumerate() {
                let st = if ri < t.head_rows { Style::default().add_modifier(Modifier::BOLD) } else { Style::default() };
                let pieces: Vec<(String, Style)> = r.iter().enumerate().flat_map(|(i, c)| {
                    let mut v = vec![]; if i > 0 { v.push((" │ ".to_string(), dim)); } v.push((c.clone(), st)); v }).collect();
                self.out.extend(wrap_styled(&pieces, self.width, &[], &[]));
            }
        }
        self.blank();
    }

    fn event(&mut self, ev: Event) {
        if self.in_code {
            match ev {
                Event::Text(t) => { self.code.push_str(&t); return; }
                Event::End(TagEnd::CodeBlock) => { self.end_code(); return; }
                _ => {}
            }
        }
        match ev {
            Event::Start(tag) => match tag {
                Tag::Paragraph => {}
                Tag::Heading { level, .. } => { self.flush(); self.heading = Some(level); }
                Tag::BlockQuote(_) => { self.flush(); self.quote += 1; }
                Tag::CodeBlock(kind) => {
                    self.flush();
                    if let CodeBlockKind::Fenced(lang) = &kind && !lang.is_empty() {
                        let (first, _) = self.prefixes();
                        let mut spans: Vec<Span<'static>> = first.iter().map(|(t, s)| Span::styled(t.clone(), *s)).collect();
                        spans.push(Span::styled(format!(" {lang} "), Style::default().fg(Color::DarkGray).bg(CODE_BG)));
                        self.out.push(Line::from(spans));
                    }
                    self.in_code = true;
                }
                Tag::List(start) => { self.flush(); self.lists.push(start); }
                Tag::Item => {
                    self.flush();
                    let marker = match self.lists.last_mut() {
                        Some(Some(n)) => { let m = format!("{n}. "); *n += 1; m }
                        _ => "• ".to_string(),
                    };
                    self.item_first = Some(marker);
                }
                Tag::Table(_) => { self.flush(); self.table = Some(Table::default()); }
                Tag::TableHead => { if let Some(t) = self.table.as_mut() { t.in_head = true; t.rows.push(vec![]); } }
                Tag::TableRow => { if let Some(t) = self.table.as_mut() { t.rows.push(vec![]); } }
                Tag::TableCell => { if let Some(t) = self.table.as_mut() { t.cell.clear(); } }
                Tag::Emphasis => self.italic += 1,
                Tag::Strong => self.bold += 1,
                Tag::Strikethrough => self.strike += 1,
                Tag::Link { dest_url, .. } => { self.link += 1; self.links.push(dest_url.to_string()); }
                Tag::Image { dest_url, .. } => { self.push(&format!("[image: {dest_url}]"), Style::default().fg(Color::DarkGray)); }
                _ => {}
            },
            Event::End(tag) => match tag {
                TagEnd::Paragraph => { self.flush(); self.blank(); }
                TagEnd::Heading(_) => { self.flush(); self.heading = None; self.blank(); }
                TagEnd::BlockQuote(_) => { self.flush(); self.quote = self.quote.saturating_sub(1); self.blank(); }
                TagEnd::List(_) => { self.flush(); self.lists.pop(); if self.lists.is_empty() { self.blank(); } }
                TagEnd::Item => self.flush(),
                TagEnd::Table => self.end_table(),
                TagEnd::TableHead => { if let Some(t) = self.table.as_mut() { t.in_head = false; t.head_rows = t.rows.len(); } }
                TagEnd::TableCell => { if let Some(t) = self.table.as_mut() { let c = std::mem::take(&mut t.cell); if let Some(r) = t.rows.last_mut() { r.push(c.trim().to_string()); } } }
                TagEnd::Emphasis => self.italic = self.italic.saturating_sub(1),
                TagEnd::Strong => self.bold = self.bold.saturating_sub(1),
                TagEnd::Strikethrough => self.strike = self.strike.saturating_sub(1),
                TagEnd::Link => {
                    self.link = self.link.saturating_sub(1);
                    if let Some(u) = self.links.pop() {
                        let shown: String = self.cur.iter().rev().take(1).map(|p| p.0.clone()).collect();
                        if !u.is_empty() && shown.trim() != u { self.push(&format!(" ({u})"), Style::default().fg(Color::DarkGray)); }
                    }
                }
                _ => {}
            },
            Event::Text(t) => { let s = self.style(); self.push(&t, s); }
            Event::Code(t) => { let s = self.style().fg(Color::Yellow).bg(CODE_BG); self.push(&t, s); }
            Event::InlineMath(t) | Event::DisplayMath(t) => { let s = self.style().fg(Color::Green); self.push(&t, s); }
            Event::Html(t) | Event::InlineHtml(t) => { self.push(&t, Style::default().fg(Color::DarkGray)); }
            Event::SoftBreak => { let s = self.style(); self.push(" ", s); }
            Event::HardBreak => self.flush(),
            Event::Rule => { self.flush(); self.out.push(Line::styled("─".repeat(self.width.min(60)), Style::default().fg(Color::DarkGray))); self.blank(); }
            Event::TaskListMarker(done) => { self.item_first = Some(if done { "☑ " } else { "☐ " }.to_string()); }
            Event::FootnoteReference(t) => { self.push(&format!("[{t}]"), Style::default().fg(Color::DarkGray)); }
        }
    }
}

/// Wraps styled pieces at `width` columns (breaking after spaces), with a prefix on the first line and another on
/// the following ones. A word longer than the line is cut.
fn wrap_styled(pieces: &[(String, Style)], width: usize, first: &[(String, Style)], next: &[(String, Style)]) -> Vec<Line<'static>> {
    let pw = |p: &[(String, Style)]| p.iter().map(|x| w(&x.0)).sum::<usize>();
    let mut lines = vec![];
    let mut spans: Vec<Span<'static>> = first.iter().map(|(t, s)| Span::styled(t.clone(), *s)).collect();
    let mut avail = width.saturating_sub(pw(first)).max(4);
    let mut used = 0usize;
    let new_line = |lines: &mut Vec<Line<'static>>, spans: &mut Vec<Span<'static>>, used: &mut usize, avail: &mut usize| {
        // drop trailing spaces of the line being closed
        if let Some(last) = spans.last_mut() { let t = last.content.trim_end().to_string(); last.content = t.into(); }
        lines.push(Line::from(std::mem::take(spans)));
        *spans = next.iter().map(|(t, s)| Span::styled(t.clone(), *s)).collect();
        *avail = width.saturating_sub(pw(next)).max(4);
        *used = 0;
    };
    for (text, style) in pieces {
        for word in text.split_inclusive(' ') {
            let ww = w(word);
            if used + w(word.trim_end()) > avail && used > 0 { new_line(&mut lines, &mut spans, &mut used, &mut avail); }
            if used == 0 && word.trim().is_empty() { continue; }   // no spaces at the start of a line
            if ww > avail {
                let mut chunk = String::new();
                for c in word.chars() {
                    let cw = c.width().unwrap_or(0);
                    if used + cw > avail { spans.push(Span::styled(std::mem::take(&mut chunk), *style)); new_line(&mut lines, &mut spans, &mut used, &mut avail); }
                    chunk.push(c); used += cw;
                }
                spans.push(Span::styled(chunk, *style));
            } else {
                spans.push(Span::styled(word.to_string(), *style));
                used += ww;
            }
        }
    }
    if used > 0 || lines.is_empty() {
        if let Some(last) = spans.last_mut() { let t = last.content.trim_end().to_string(); last.content = t.into(); }
        lines.push(Line::from(spans));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    fn text(l: &Line) -> String { l.spans.iter().map(|s| s.content.as_ref()).collect() }
    fn texts(v: &[Line]) -> Vec<String> { v.iter().map(text).collect() }

    /// control characters of the model's text never reach the terminal
    #[test]
    fn control_characters_are_dropped() {
        // ESC and BEL go; the newline stays (markdown joins the paragraph: "ciao[31mrosso fine")
        let out = texts(&render("ciao\x1b[31mrosso\x07\nfine", 40));
        assert_eq!(out, vec!["ciao[31mrosso fine"]);
        assert_eq!(texts(&render("a\x1bb", 40)), vec!["ab"]);
    }

    #[test]
    fn inline_styles() {
        let l = render("a **bold** and *it* and `code`", 80);
        assert_eq!(texts(&l), vec!["a bold and it and code"]);
        let bold = l[0].spans.iter().find(|s| s.content == "bold").unwrap();
        assert!(bold.style.add_modifier.contains(Modifier::BOLD));
        let code = l[0].spans.iter().find(|s| s.content == "code").unwrap();
        assert_eq!(code.style.bg, Some(CODE_BG));
    }
    #[test]
    fn lists_and_heading() {
        let l = texts(&render("# Title\n\n- one\n- two\n  1. a\n  2. b\n\nend", 80));
        assert_eq!(l, vec!["Title", "", "• one", "• two", "  1. a", "  2. b", "", "end"]);
    }
    #[test]
    fn wrapped_item_is_indented() {
        let l = texts(&render("- aaa bbb ccc ddd eee", 12));
        assert_eq!(l, vec!["• aaa bbb", "  ccc ddd", "  eee"]);
        assert!(l.iter().all(|x| w(x) <= 12));
    }
    #[test]
    fn code_block_and_unclosed_while_streaming() {
        let l = texts(&render("x\n\n```rust\nfn main() {}\n```", 30));
        assert_eq!(l[2], " rust ");
        assert!(l[3].starts_with(" fn main() {}") && w(&l[3]) == 30);
        let s = texts(&render("x\n\n```\nlet a = 1;", 30));   // the closing ``` has not arrived yet
        assert!(s.iter().any(|x| x.starts_with(" let a = 1;")));
        let b = texts(&render("an **unclosed bold", 30));        // the closing ** has not arrived yet
        assert_eq!(b, vec!["an **unclosed bold"]);
    }
    #[test]
    fn tables_fit_or_fall_back() {
        let t = texts(&render("| a | bb |\n|---|---|\n| 1 | 2 |", 40));
        assert_eq!(t, vec!["a │ bb", "──┼───", "1 │ 2 "]);
        let wide = texts(&render("| aaaaaaaaaa | bbbbbbbbbb |\n|---|---|\n| 1 | 2 |", 15));
        assert!(wide.iter().all(|x| w(x) <= 15), "{wide:?}");
    }
    #[test]
    fn link_shows_address() {
        let l = texts(&render("see [docs](https://x.org) now", 80));
        assert_eq!(l, vec!["see docs (https://x.org) now"]);
    }
    #[test]
    fn every_line_fits() {
        let md = "# A long heading that will need wrapping\n\n> a quote that is long enough to wrap twice at least\n\n1. item with `inline code` inside\n\n---\n\nplain text with a verylongwordthatcannotbebrokennicely end";
        for width in [12usize, 20, 33] {
            for l in render(md, width) { assert!(w(&text(&l)) <= width, "width {width}: {:?}", text(&l)); }
        }
    }
}
