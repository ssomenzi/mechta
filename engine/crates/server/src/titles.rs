//! Session titles: opencode asks the same model for the title
//! with a prompt of its own ("You are a title generator..."). Here the answer is a DETERMINISTIC title, without the engine.
//! It is the point where a small model (local or cloud) can be plugged in later: it is enough for `fake_title` to become a call.
//! MECHTA_TITLES=engine: no fake title, the request goes to the model like the others.
use serde_json::Value;

/// If the request is the title one, returns the title; otherwise None.
pub fn fake_title(body: &[u8]) -> Option<String> {
    if std::env::var("MECHTA_TITLES").is_ok_and(|v| v == "engine") { return None; }
    let v: Value = serde_json::from_slice(body).ok()?;
    let m = v["messages"].as_array()?;
    let system = m.first().filter(|x| x["role"] == "system").and_then(|x| text(&x["content"]))?;
    if !system.starts_with("You are a title generator") { return None; }
    // the question: the last user message (opencode sends it in quotes)
    let question = m.iter().rev().filter(|x| x["role"] == "user").find_map(|x| text(&x["content"])).unwrap_or_default();
    let clean: String = question.trim().trim_matches('"').split_whitespace().collect::<Vec<_>>().join(" ");
    let mut t: String = clean.chars().take(50).collect();
    if clean.chars().count() > 50 { t = t.trim_end().to_string(); t.push('…'); }
    if t.is_empty() { t = "Nuova sessione".into(); }
    Some(t)
}

fn text(c: &Value) -> Option<String> {
    match c {
        Value::String(s) => Some(s.clone()),
        Value::Array(parts) => Some(parts.iter().filter_map(|p| p["text"].as_str()).collect::<Vec<_>>().join("")),
        _ => None,
    }
}
