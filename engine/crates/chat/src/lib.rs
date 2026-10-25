//! Qwen3.6 chat template and output parser.
//! Also Flint (`Family::Flint`): its official template and its tool-call format.
//! The public signatures are the CONTRACT with the server: they are not changed.
#![forbid(unsafe_code)]
use std::sync::OnceLock;

use minijinja::{Environment, Error, ErrorKind, Value as JValue};
use serde_json::{Map, Value};

/// The model family: which chat template and which tool-call format.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Family {
    /// Qwen3.6 (and its fine-tunes): `<|im_start|>`, `<function=...><parameter=...>`
    #[default]
    Qwen,
    /// Flint (RecursiveSI): `<|user|>`/`<|assistant|>`, `<tool_call>name<arg_key>..</arg_key><arg_value>..</arg_value>`.
    /// The model decides by itself whether to reason: no switch, the prompt never opens `<think>`.
    Flint,
}

#[derive(Clone, Debug)]
pub struct Options {
    /// thinking on (default): the prompt ends with "<|im_start|>assistant\n<think>\n"
    pub thinking: bool,
    /// carries over the reasoning of ALL the assistant turns, not only of those after the last user question
    /// (default). This way the prompt of a turn is the exact prefix of the next turn and the prefix cache reuses it
    /// entirely: without it, at every new question the old reasonings disappear and the prefill restarts from the first message.
    pub preserve_thinking: bool,
    /// template and format of the model (default Qwen). With Flint `thinking` and `preserve_thinking` have no effect.
    pub family: Family,
}
impl Default for Options { fn default() -> Self { Options { thinking: true, preserve_thinking: true, family: Family::Qwen } } }

// ---------------------------------------------------------------------------------------------------------------
// Template
// ---------------------------------------------------------------------------------------------------------------

/// The official template, extracted from the GGUF.
const TEMPLATE: &str = include_str!("../template.jinja");
/// Flint's official template (`chat_template.jinja` of RecursiveSI/flint, identical to the one in the GGUF).
const TEMPLATE_FLINT: &str = include_str!("../template-flint.jinja");

/// JSON formatter like Python's `json.dumps` (the one transformers uses for the `tojson` filter):
/// separators ", " and ": ", no escaping of non-ASCII characters nor of < > & (minijinja instead escapes them for HTML).
struct PythonFormat;
impl serde_json::ser::Formatter for PythonFormat {
    fn begin_array_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + std::io::Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + std::io::Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

/// Serializes like Python's `json.dumps(x, ensure_ascii=False)`.
fn json_python(v: &Value) -> String {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PythonFormat);
    serde::Serialize::serialize(v, &mut ser).expect("serializzazione JSON in memoria");
    String::from_utf8(out).expect("serde_json produce UTF-8")
}

/// `tojson` filter the transformers way (replaces the minijinja one, which escapes < > & ').
/// Flint's template passes `ensure_ascii=False`: the only value supported (it is what the output already is).
fn filter_tojson(v: JValue, kwargs: minijinja::value::Kwargs) -> Result<JValue, Error> {
    if kwargs.get::<Option<bool>>("ensure_ascii")? == Some(true) {
        return Err(Error::new(ErrorKind::InvalidOperation, "tojson: ensure_ascii=True is not supported"));
    }
    kwargs.assert_all_used()?;
    let j = serde_json::to_value(&v)
        .map_err(|e| Error::new(ErrorKind::InvalidOperation, format!("tojson: {e}")))?;
    Ok(JValue::from_safe_string(json_python(&j)))
}

fn jinja_env() -> &'static Environment<'static> {
    static AMB: OnceLock<Environment<'static>> = OnceLock::new();
    AMB.get_or_init(|| {
        let mut env = Environment::new();
        // like transformers (ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)); with this template,
        // which uses `{%-`/`{{-` everywhere, the result does not change, but we keep the same configuration.
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        // Python methods on strings: .startswith .endswith .split .rstrip .lstrip
        env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
        env.add_filter("tojson", filter_tojson);
        env.add_function("raise_exception", |msg: String| -> Result<JValue, Error> {
            Err(Error::new(ErrorKind::InvalidOperation, msg))
        });
        env.add_template("chat", TEMPLATE).expect("template.jinja does not compile");
        env.add_template("chat_flint", TEMPLATE_FLINT).expect("template-flint.jinja does not compile");
        env
    })
}

/// OpenAI content → string: the parts `[{type:"text", text}]` are joined; null stays null.
fn normalize_content(c: &Value) -> Value {
    match c {
        Value::Array(parts) => {
            let mut s = String::new();
            for p in parts {
                if let Some(t) = p.get("text").and_then(Value::as_str) { s.push_str(t); }
                else if let Some(t) = p.as_str() { s.push_str(t); }
            }
            Value::String(s)
        }
        other => other.clone(),
    }
}

/// An OpenAI message → the shape the template expects.
fn normalize_message(m: &Value) -> Value {
    let Some(obj) = m.as_object() else { return m.clone() };
    let mut out = obj.clone();
    if let Some(c) = obj.get("content") { out.insert("content".into(), normalize_content(c)); }
    // some clients send the reasoning back as `reasoning` instead of `reasoning_content`
    if !out.contains_key("reasoning_content") {
        if let Some(r) = obj.get("reasoning").filter(|r| r.is_string()) { out.insert("reasoning_content".into(), r.clone()); }
    }
    // tool_calls: `function.arguments` arrives as a JSON string, the template wants an object
    if let Some(Value::Array(calls)) = obj.get("tool_calls") {
        let converted: Vec<Value> = calls.iter().map(|tc| {
            let mut tc = tc.clone();
            if let Some(f) = tc.get_mut("function").and_then(Value::as_object_mut) {
                if let Some(Value::String(a)) = f.get("arguments") {
                    let a = a.trim();
                    let v = if a.is_empty() { Value::Object(Map::new()) }
                        else { serde_json::from_str::<Value>(a).unwrap_or_else(|_| Value::String(a.to_string())) };
                    f.insert("arguments".into(), v);
                }
            }
            tc
        }).collect();
        out.insert("tool_calls".into(), Value::Array(converted));
    }
    Value::Object(out)
}

/// OpenAI messages (JSON array of chat/completions) + OpenAI tools (array, optional) → prompt text, with the model's
/// official template (`template.jinja`), add_generation_prompt = true.
pub fn prompt(msgs: &Value, tools: Option<&Value>, opts: &Options) -> Result<String, String> {
    let list = msgs.as_array().ok_or("messages is not an array")?;
    let msgs: Vec<Value> = list.iter().map(normalize_message).collect();
    // OpenAI tools ({type:"function", function:{...}}) pass through as they are: the template only does `tool | tojson`
    let tools = tools.filter(|t| t.as_array().is_some_and(|a| !a.is_empty()));
    let ctx = minijinja::context! {
        messages => JValue::from_serialize(&msgs),
        tools => tools.map(JValue::from_serialize),
        add_generation_prompt => true,
        enable_thinking => opts.thinking,
        preserve_thinking => opts.preserve_thinking,
    };
    let name = match opts.family { Family::Qwen => "chat", Family::Flint => "chat_flint" };
    let t = jinja_env().get_template(name).map_err(|e| e.to_string())?;
    t.render(ctx).map_err(|e| format!("chat template: {e:#}"))
}

// ---------------------------------------------------------------------------------------------------------------
// Output parser
// ---------------------------------------------------------------------------------------------------------------

/// A piece of the model output, already classified.
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// text inside <think>...</think> → `reasoning_content`
    Reasoning(String),
    /// text of the answer → `content`
    Text(String),
    /// a complete tool call: arguments already in JSON (serialized object), types taken from the tools schema
    ToolCall { id: String, name: String, arguments_json: String },
}

const OPEN_THINK: &str = "<think>";
const CLOSE_THINK: &str = "</think>";
const OPEN_TOOL: &str = "<tool_call>";
const CLOSE_TOOL: &str = "</tool_call>";

#[derive(Clone, Copy, Debug, PartialEq)]
enum State { Reasoning, Text, Tool }

/// Streaming output parser: text is pushed as it arrives, events come out. It does not hold back more than
/// necessary (only the piece that could be the start of a tag).
pub struct Parser {
    state: State,
    /// text arrived and not yet classified (at most a tail that may be the start of a tag; in the Tool state
    /// the whole block from `<tool_call>` on)
    buf: String,
    /// at the start of the Text the newlines (after </think>) or all whitespace (after </tool_call>) are removed
    trim: Trim,
    tools: Option<Value>,
    rng: u64,
    family: Family,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Trim { Nothing, Newlines, Spaces }

impl Parser {
    /// `thinking`: if true, the output STARTS inside the reasoning (the "<think>\n" is in the prompt).
    /// `tools`: the OpenAI schema of the tools (to convert parameter values into the right types).
    pub fn new(thinking: bool, tools: Option<&Value>) -> Self { Self::per_family(Family::Qwen, thinking, tools) }

    /// As `new`, for a model family. Flint: the output never starts inside the reasoning (the model writes `<think>`
    /// itself), and no whitespace is trimmed, so the text given back to the template is the text the model wrote.
    pub fn per_family(family: Family, thinking: bool, tools: Option<&Value>) -> Self {
        let thinking = thinking && family == Family::Qwen;
        let ns = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0);
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let rng = (ns ^ n.wrapping_mul(0x9E37_79B9_7F4A_7C15)) | 1;
        Parser {
            state: if thinking { State::Reasoning } else { State::Text },
            buf: String::new(),
            trim: Trim::Nothing,
            tools: tools.cloned(),
            rng,
            family,
        }
    }

    pub fn push(&mut self, text: &str) -> Vec<Event> {
        self.buf.push_str(text);
        let mut ev = Vec::new();
        self.advance(&mut ev, false);
        ev
    }

    /// end of the output: flushes what is left
    pub fn end(&mut self) -> Vec<Event> {
        let mut ev = Vec::new();
        self.advance(&mut ev, true);
        let resto = std::mem::take(&mut self.buf);
        match self.state {
            State::Reasoning => push_reasoning(&mut ev, resto),
            // unclosed tool_call: it is not lost, it comes out as text as it is
            State::Tool => push_text(&mut ev, resto),
            State::Text => self.emit_text(&mut ev, resto),
        }
        ev
    }

    /// Consumes `buf` as far as possible. With `end` true it does not hold back suspicious tails (the caller flushes them).
    fn advance(&mut self, ev: &mut Vec<Event>, end: bool) {
        loop {
            match self.state {
                State::Reasoning => {
                    if let Some(i) = self.buf.find(CLOSE_THINK) {
                        let before = self.buf[..i].to_string();
                        self.buf.drain(..i + CLOSE_THINK.len());
                        push_reasoning(ev, before);
                        self.state = State::Text;
                        if self.family == Family::Qwen { self.trim = Trim::Newlines; }
                        continue;
                    }
                    if end { return; }
                    let k = suspect_tail(&self.buf, &[CLOSE_THINK]);
                    let ready: String = self.buf.drain(..self.buf.len() - k).collect();
                    push_reasoning(ev, ready);
                    return;
                }
                State::Text => {
                    let t = self.buf.find(OPEN_TOOL);
                    let r = self.buf.find(OPEN_THINK);
                    let first = match (t, r) {
                        (Some(a), Some(b)) => Some(if a < b { (a, State::Tool) } else { (b, State::Reasoning) }),
                        (Some(a), None) => Some((a, State::Tool)),
                        (None, Some(b)) => Some((b, State::Reasoning)),
                        (None, None) => None,
                    };
                    if let Some((i, new)) = first {
                        let before: String = self.buf.drain(..i).collect();
                        self.emit_text(ev, before);
                        if new == State::Reasoning { self.buf.drain(..OPEN_THINK.len()); }
                        // in the Tool state the buffer also holds the "<tool_call>" (needed if the block stays open)
                        self.state = new;
                        continue;
                    }
                    if end { return; }
                    let k = suspect_tail(&self.buf, &[OPEN_TOOL, OPEN_THINK]);
                    let ready: String = self.buf.drain(..self.buf.len() - k).collect();
                    self.emit_text(ev, ready);
                    return;
                }
                State::Tool => {
                    let Some(i) = self.buf.find(CLOSE_TOOL) else { return };
                    let block: String = self.buf.drain(..i + CLOSE_TOOL.len()).collect();
                    let inner = &block[OPEN_TOOL.len()..block.len() - CLOSE_TOOL.len()];
                    match self.read_call(inner) {
                        Some(e) => ev.push(e),
                        None => push_text(ev, block),
                    }
                    self.state = State::Text;
                    if self.family == Family::Qwen { self.trim = Trim::Spaces; }
                }
            }
        }
    }

    /// Emits normal text, removing the leading newlines/spaces if requested.
    fn emit_text(&mut self, ev: &mut Vec<Event>, s: String) {
        let s = match self.trim {
            Trim::Nothing => s,
            Trim::Newlines => s.trim_start_matches(['\n', '\r']).to_string(),
            Trim::Spaces => s.trim_start().to_string(),
        };
        if !s.is_empty() {
            self.trim = Trim::Nothing;
            push_text(ev, s);
        }
    }

    /// Inside of a <tool_call>...</tool_call> → event. None if there is no <function=...>.
    fn read_call(&mut self, inner: &str) -> Option<Event> {
        if self.family == Family::Flint { return self.read_call_flint(inner); }
        let a = inner.find("<function=")? + "<function=".len();
        let name_end = inner[a..].find('>')? + a;
        let name = inner[a..name_end].trim().to_string();
        if name.is_empty() { return None; }
        let mut body = &inner[name_end + 1..];
        if let Some(f) = body.rfind("</function>") { body = &body[..f]; }
        let mut arguments = Map::new();
        let mut resto = body;
        while let Some(p) = resto.find("<parameter=") {
            let after = &resto[p + "<parameter=".len()..];
            let Some(g) = after.find('>') else { break };
            let pname = after[..g].trim().to_string();
            let value_and_rest = &after[g + 1..];
            // end of the value: </parameter>; if missing, the next <parameter= or the end
            let (value, beyond) = match value_and_rest.find("</parameter>") {
                Some(e) => (&value_and_rest[..e], &value_and_rest[e + "</parameter>".len()..]),
                None => match value_and_rest.find("<parameter=") {
                    Some(e) => (&value_and_rest[..e], &value_and_rest[e..]),
                    None => (value_and_rest, ""),
                },
            };
            let value = value.strip_prefix('\n').unwrap_or(value);
            let value = value.strip_suffix('\n').unwrap_or(value);
            let v = self.convert(&name, &pname, value);
            arguments.insert(pname, v);
            resto = beyond;
        }
        Some(Event::ToolCall { id: self.new_id(), name, arguments_json: Value::Object(arguments).to_string() })
    }

    /// Flint: inside of `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>` → event. The template
    /// writes string values as they are and the others as JSON, so a value goes through the schema like Qwen's.
    /// None if there is no name.
    fn read_call_flint(&mut self, inner: &str) -> Option<Event> {
        const K: &str = "<arg_key>"; const KF: &str = "</arg_key>"; const V: &str = "<arg_value>"; const VF: &str = "</arg_value>";
        let name_end = inner.find(K).unwrap_or(inner.len());
        let name = inner[..name_end].trim().to_string();
        if name.is_empty() || name.contains('<') { return None; }
        let mut arguments = Map::new();
        let mut resto = &inner[name_end..];
        while let Some(p) = resto.find(K) {
            let after = &resto[p + K.len()..];
            let Some(fk) = after.find(KF) else { break };
            let pname = after[..fk].trim().to_string();
            let after = &after[fk + KF.len()..];
            let Some(pv) = after.find(V) else { break };
            let value_and_rest = &after[pv + V.len()..];
            // end of the value: </arg_value>; if missing, the next <arg_key> or the end
            let (value, beyond) = match value_and_rest.find(VF) {
                Some(e) => (&value_and_rest[..e], &value_and_rest[e + VF.len()..]),
                None => match value_and_rest.find(K) {
                    Some(e) => (&value_and_rest[..e], &value_and_rest[e..]),
                    None => (value_and_rest, ""),
                },
            };
            let v = self.convert(&name, &pname, value);
            arguments.insert(pname, v);
            resto = beyond;
        }
        Some(Event::ToolCall { id: self.new_id(), name, arguments_json: Value::Object(arguments).to_string() })
    }

    /// Converts the value of a parameter to the schema type; if that fails, it stays a string.
    fn convert(&self, function: &str, param: &str, value: &str) -> Value {
        let string = || Value::String(value.to_string());
        let Some(types) = self.param_types(function, param) else { return string() };
        if types.iter().any(|t| t == "string") || types.is_empty() { return string(); }
        let v = value.trim();
        if types.iter().any(|t| t == "boolean") {
            match v { "True" => return Value::Bool(true), "False" => return Value::Bool(false), _ => {} }
        }
        if types.iter().any(|t| matches!(t.as_str(), "integer" | "number" | "boolean" | "object" | "array" | "null")) {
            if let Ok(j) = serde_json::from_str::<Value>(v) { return j; }
        }
        string()
    }

    /// The JSON-schema types declared for `param` of the function `function` (`type` string or list).
    fn param_types(&self, function: &str, param: &str) -> Option<Vec<String>> {
        let tools = self.tools.as_ref()?.as_array()?;
        let f = tools.iter().find_map(|t| {
            let f = t.get("function").unwrap_or(t);
            (f.get("name")?.as_str()? == function).then_some(f)
        })?;
        let kind = f.get("parameters")?.get("properties")?.get(param)?.get("type")?;
        match kind {
            Value::String(s) => Some(vec![s.clone()]),
            Value::Array(a) => Some(a.iter().filter_map(|x| x.as_str().map(String::from)).collect()),
            _ => None,
        }
    }

    /// "call_" + 24 alphanumerics (xorshift64*).
    fn new_id(&mut self) -> String {
        const ALFA: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        let mut s = String::from("call_");
        for _ in 0..24 {
            self.rng ^= self.rng >> 12;
            self.rng ^= self.rng << 25;
            self.rng ^= self.rng >> 27;
            let x = self.rng.wrapping_mul(0x2545_F491_4F6C_DD1D);
            s.push(ALFA[(x >> 32) as usize % ALFA.len()] as char);
        }
        s
    }
}

fn push_text(ev: &mut Vec<Event>, s: String) { if !s.is_empty() { ev.push(Event::Text(s)); } }
fn push_reasoning(ev: &mut Vec<Event>, s: String) { if !s.is_empty() { ev.push(Event::Reasoning(s)); } }

/// Length of the longest tail of `buf` that is a (proper) prefix of one of the tags: that must be held back.
fn suspect_tail(buf: &str, tag: &[&str]) -> usize {
    let max = tag.iter().map(|t| t.len() - 1).max().unwrap_or(0).min(buf.len());
    (1..=max).rev()
        .find(|&k| buf.is_char_boundary(buf.len() - k) && tag.iter().any(|t| t.starts_with(&buf[buf.len() - k..])))
        .unwrap_or(0)
}

#[cfg(test)]
mod test;
#[cfg(test)]
mod test_flint;
