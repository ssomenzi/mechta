//! Flint: the template against jinja2 (expected outputs rendered with jinja2 in `flint/atteso/`) and the parser of its
//! tool calls (CPU only, no model).
use super::*;
use serde_json::json;

fn folder() -> std::path::PathBuf { std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("flint") }

fn flint() -> Options { Options { family: Family::Flint, ..Default::default() } }

#[test]
fn template_identical_to_jinja2() {
    let cases: Value = serde_json::from_str(&std::fs::read_to_string(folder().join("casi.json")).unwrap()).unwrap();
    let tool_sets: Value = serde_json::from_str(&std::fs::read_to_string(folder().join("strumenti.json")).unwrap()).unwrap();
    let mut different = vec![];
    for c in cases.as_array().unwrap() {
        let name = c["nome"].as_str().unwrap();
        let tools = c.get("tools").and_then(Value::as_str).map(|k| &tool_sets[k]);
        let p = prompt(&c["messages"], tools, &flint()).unwrap();
        let expected = std::fs::read_to_string(folder().join("atteso").join(format!("{name}.txt"))).unwrap();
        if p != expected { println!("--- {name}\nmechta:  {p:?}\njinja2:  {expected:?}"); different.push(name.to_string()); }
    }
    assert!(different.is_empty(), "diversi da jinja2: {different:?}");
}

#[test]
fn qwen_resta_il_default() {
    let m = json!([{"role": "user", "content": "Ciao"}]);
    assert!(prompt(&m, None, &Options::default()).unwrap().starts_with("<|im_start|>user"));
}

fn chunked(tools: Option<&Value>, output: &str, n: usize) -> Vec<Event> {
    let mut p = Parser::per_family(Family::Flint, true, tools);
    let mut ev = Vec::new();
    let chars: Vec<char> = output.chars().collect();
    for piece in chars.chunks(n) { ev.extend(p.push(&piece.iter().collect::<String>())); }
    ev.extend(p.end());
    // consecutive pieces of the same kind joined; call ids dropped (random)
    let mut out: Vec<Event> = vec![];
    for e in ev {
        let e = match e { Event::ToolCall { name, arguments_json, .. } => Event::ToolCall { id: String::new(), name, arguments_json }, x => x };
        match (out.last_mut(), &e) {
            (Some(Event::Text(a)), Event::Text(b)) => a.push_str(b),
            (Some(Event::Reasoning(a)), Event::Reasoning(b)) => a.push_str(b),
            _ => out.push(e),
        }
    }
    out
}

fn all_measures(tools: Option<&Value>, output: &str) -> Vec<Event> {
    let whole = chunked(tools, output, usize::MAX);
    for n in [1, 2, 3, 5, 7, 11] { assert_eq!(chunked(tools, output, n), whole, "pezzi da {n}"); }
    whole
}

fn call(name: &str, args: &str) -> Event { Event::ToolCall { id: String::new(), name: name.into(), arguments_json: args.into() } }

#[test]
fn parser_reasoning_and_typed_calls() {
    let tool_sets: Value = serde_json::from_str(&std::fs::read_to_string(folder().join("strumenti.json")).unwrap()).unwrap();
    let tools = &tool_sets["STRUMENTI"];
    // the same text the template writes for case 08: the parser must give back what the template takes
    let output = "<think>Servono due chiamate.</think><tool_call>read<arg_key>filePath</arg_key><arg_value>/x/main.rs</arg_value><arg_key>offset</arg_key><arg_value>10</arg_value><arg_key>limit</arg_key><arg_value>200</arg_value></tool_call><tool_call>grep<arg_key>pattern</arg_key><arg_value>TODO</arg_value><arg_key>include</arg_key><arg_value>[\"*.rs\", \"*.toml\"]</arg_value><arg_key>opzioni</arg_key><arg_value>{\"case\": false}</arg_value></tool_call>";
    assert_eq!(all_measures(Some(tools), output), vec![
        Event::Reasoning("Servono due chiamate.".into()),
        call("read", r#"{"filePath":"/x/main.rs","offset":10,"limit":200}"#),
        call("grep", r#"{"pattern":"TODO","include":["*.rs","*.toml"],"opzioni":{"case":false}}"#),
    ]);
}

#[test]
fn parser_without_reasoning_no_trimming() {
    // no <think>: everything is text from the first character, spaces and newlines kept
    assert_eq!(all_measures(None, "\n\nCiao <b>mondo</b> < 3\n"), vec![Event::Text("\n\nCiao <b>mondo</b> < 3\n".into())]);
    // text after </think> kept as written
    assert_eq!(all_measures(None, "<think>x</think>\n\nRisposta."),
        vec![Event::Reasoning("x".into()), Event::Text("\n\nRisposta.".into())]);
}

#[test]
fn parser_edge_cases() {
    // no schema: values stay strings
    assert_eq!(all_measures(None, "<tool_call>f<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>"), vec![call("f", r#"{"n":"7"}"#)]);
    // a call without arguments
    assert_eq!(all_measures(None, "<tool_call>ora</tool_call>"), vec![call("ora", "{}")]);
    // a value with newlines and tags of other formats inside
    assert_eq!(all_measures(None, "<tool_call>w<arg_key>c</arg_key><arg_value>a\n<b>\n</arg_value></tool_call>"), vec![call("w", r#"{"c":"a\n<b>\n"}"#)]);
    // a call cut in the middle: text, whole
    assert_eq!(all_measures(None, "Ecco:<tool_call>f<arg_key>n"), vec![Event::Text("Ecco:<tool_call>f<arg_key>n".into())]);
    // a block without a name: text
    assert_eq!(all_measures(None, "<tool_call><arg_key>n</arg_key></tool_call>"), vec![Event::Text("<tool_call><arg_key>n</arg_key></tool_call>".into())]);
}
