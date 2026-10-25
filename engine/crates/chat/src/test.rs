//! Tests of the template and of the parser (CPU only, no model).
use super::*;
use serde_json::json;

fn example_tools() -> Value {
    json!([
        {"type": "function", "function": {
            "name": "leggi_file",
            "description": "Legge un file <di testo> & ne restituisce il contenuto",
            "parameters": {"type": "object", "properties": {
                "percorso": {"type": "string", "description": "percorso del file"},
                "riga_iniziale": {"type": "integer"},
                "intero": {"type": "boolean"}
            }, "required": ["percorso"]}
        }},
        {"type": "function", "function": {
            "name": "cerca",
            "description": "Cerca nel codice",
            "parameters": {"type": "object", "properties": {
                "schema": {"type": "string"},
                "opzioni": {"type": "object"},
                "limite": {"type": ["integer", "null"]},
                "file": {"type": "array"}
            }}
        }}
    ])
}

#[test]
fn simple_thinking_on_and_off() {
    let m = json!([
        {"role": "system", "content": "Sei un assistente."},
        {"role": "user", "content": [{"type": "text", "text": "Ciao, "}, {"type": "text", "text": "come va?"}]}
    ]);
    let p = prompt(&m, None, &Options { thinking: true, ..Default::default() }).unwrap();
    println!("--- thinking acceso ---\n{p}---");
    assert_eq!(p, "<|im_start|>system\nSei un assistente.<|im_end|>\n<|im_start|>user\nCiao, come va?<|im_end|>\n<|im_start|>assistant\n<think>\n");
    let p = prompt(&m, None, &Options { thinking: false, ..Default::default() }).unwrap();
    assert!(p.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{p:?}");
    // empty tools = no tool
    let p2 = prompt(&m, Some(&json!([])), &Options { thinking: false, ..Default::default() }).unwrap();
    assert_eq!(p, p2);
}

#[test]
fn error_without_messages() {
    assert!(prompt(&json!([]), None, &Options::default()).is_err());
}

#[test]
fn conversation_with_tools() {
    let m = json!([
        {"role": "system", "content": "Sei un agente di programmazione."},
        {"role": "user", "content": "Leggi main.rs"},
        {"role": "assistant", "content": null, "reasoning_content": "Devo leggere il file.",
         "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "leggi_file", "arguments": "{\"percorso\": \"src/main.rs\", \"riga_iniziale\": 3, \"intero\": true}"}},
            {"id": "call_2", "type": "function", "function": {"name": "cerca", "arguments": "{\"schema\": \"fn main\", \"opzioni\": {\"maiuscole\": false, \"tipi\": [\"rs\"]}}"}}
         ]},
        {"role": "tool", "tool_call_id": "call_1", "content": "fn main() {}"},
        {"role": "tool", "tool_call_id": "call_2", "content": [{"type": "text", "text": "src/main.rs:1"}]},
        {"role": "assistant", "content": "Il file contiene solo `main`.", "reasoning_content": "Ho letto."},
        {"role": "user", "content": "Grazie"}
    ]);
    let p = prompt(&m, Some(&example_tools()), &Options { preserve_thinking: false, ..Default::default() }).unwrap();
    println!("--- conversazione con tools ---\n{p}---");
    // booleans become "True" as in Python/transformers (`true|string`), and the parser recognizes them
    // tools like Python's json.dumps: separators ", " and ": ", no escaping of < > &
    assert!(p.starts_with("<|im_start|>system\n# Tools\n\nYou have access to the following functions:\n\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"leggi_file\", \"description\": \"Legge un file <di testo> & ne restituisce"));
    assert!(p.contains("</IMPORTANT>\n\nSei un agente di programmazione.<|im_end|>\n"));
    // the reasoning of the turns before the last user question is not carried over
    assert!(p.contains("<|im_start|>user\nLeggi main.rs<|im_end|>\n<|im_start|>assistant\n<tool_call>\n<function=leggi_file>\n<parameter=percorso>\nsrc/main.rs\n</parameter>\n<parameter=riga_iniziale>\n3\n</parameter>\n<parameter=intero>\nTrue\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=cerca>\n<parameter=schema>\nfn main\n</parameter>\n<parameter=opzioni>\n{\"maiuscole\": false, \"tipi\": [\"rs\"]}\n</parameter>\n</function>\n</tool_call><|im_end|>\n"));
    assert!(p.contains("<|im_start|>user\n<tool_response>\nfn main() {}\n</tool_response>\n<tool_response>\nsrc/main.rs:1\n</tool_response><|im_end|>\n"));
    assert!(p.contains("<|im_start|>assistant\nIl file contiene solo `main`.<|im_end|>\n<|im_start|>user\nGrazie<|im_end|>\n<|im_start|>assistant\n<think>\n"));
    assert!(p.ends_with("<|im_start|>user\nGrazie<|im_end|>\n<|im_start|>assistant\n<think>\n"));
}

#[test]
fn reasoning_carried_after_the_last_question() {
    // in the middle of a tool round (after the last user question) the reasoning is carried over in the prompt
    let m = json!([
        {"role": "user", "content": "Leggi main.rs"},
        {"role": "assistant", "content": "Lo leggo.", "reasoning_content": "Devo leggere il file.\n",
         "tool_calls": [{"id": "c", "type": "function", "function": {"name": "leggi_file", "arguments": "{\"percorso\": \"main.rs\"}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "fn main() {}"}
    ]);
    let p = prompt(&m, Some(&example_tools()), &Options::default()).unwrap();
    println!("--- giro di tools in corso ---\n{p}---");
    assert!(p.contains("<|im_start|>assistant\n<think>\nDevo leggere il file.\n</think>\n\nLo leggo.\n\n<tool_call>\n<function=leggi_file>\n<parameter=percorso>\nmain.rs\n</parameter>\n</function>\n</tool_call><|im_end|>\n<|im_start|>user\n<tool_response>\nfn main() {}\n</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"));
    // reasoning inside the content (<think>...</think>) without reasoning_content: the template separates it
    let m = json!([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "<think>\npenso\n</think>\n\nrisposta"},
        {"role": "user", "content": "b"}
    ]);
    let p = prompt(&m, None, &Options { preserve_thinking: false, ..Default::default() }).unwrap();
    assert!(p.contains("<|im_start|>assistant\nrisposta<|im_end|>\n"), "{p:?}");
}

#[test]
fn preserve_thinking_keeps_the_prefix_between_two_questions() {
    // A short "thanks" once cost 173 s of prefill because the new question removed the old reasonings.
    // With the default (preserve_thinking) the previous prompt + the generated answer is the exact prefix of the next prompt.
    let before = json!([
        {"role": "user", "content": "Leggi main.rs"},
        {"role": "assistant", "content": "", "reasoning_content": "Devo leggere il file.",
         "tool_calls": [{"id": "c", "type": "function", "function": {"name": "leggi_file", "arguments": "{\"percorso\": \"main.rs\"}"}}]},
        {"role": "tool", "tool_call_id": "c", "content": "fn main() {}"}
    ]);
    let mut after = before.as_array().unwrap().clone();
    after.push(json!({"role": "assistant", "content": "Contiene solo main.", "reasoning_content": "Ho letto."}));
    after.push(json!({"role": "user", "content": "ottimo grazie"}));
    let after = Value::Array(after);
    let p1 = prompt(&before, Some(&example_tools()), &Options::default()).unwrap();
    let p2 = prompt(&after, Some(&example_tools()), &Options::default()).unwrap();
    assert!(p2.starts_with(&format!("{p1}Ho letto.\n</think>\n\nContiene solo main.<|im_end|>\n")), "{p1:?}\n{p2:?}");
    // with the behaviour of the official template the prefix breaks
    let off = Options { preserve_thinking: false, ..Default::default() };
    assert!(!prompt(&after, Some(&example_tools()), &off).unwrap().starts_with(&prompt(&before, Some(&example_tools()), &off).unwrap()));
}

// ------------------------------------------------------------------- parser

/// Normalized events for the comparison: consecutive texts joined, call ids removed.
fn normalize(ev: Vec<Event>) -> Vec<(char, String)> {
    let mut out: Vec<(char, String)> = Vec::new();
    for e in ev {
        let (k, s) = match e {
            Event::Reasoning(s) => ('R', s),
            Event::Text(s) => ('T', s),
            Event::ToolCall { id, name, arguments_json } => {
                assert!(id.starts_with("call_") && id.len() == 29 && id[5..].chars().all(|c| c.is_ascii_alphanumeric()), "{id}");
                out.push(('C', format!("{name} {arguments_json}")));
                continue;
            }
        };
        assert!(!s.is_empty(), "evento vuoto");
        match out.last_mut() {
            Some((k2, s2)) if *k2 == k => s2.push_str(&s),
            _ => out.push((k, s)),
        }
    }
    out
}

fn chunked(thinking: bool, tools: Option<&Value>, output: &str, n: usize) -> Vec<(char, String)> {
    let mut p = Parser::new(thinking, tools);
    let mut ev = Vec::new();
    let chars: Vec<char> = output.chars().collect();
    for piece in chars.chunks(n) {
        ev.extend(p.push(&piece.iter().collect::<String>()));
    }
    ev.extend(p.end());
    normalize(ev)
}

fn all_measures(thinking: bool, tools: Option<&Value>, output: &str) -> Vec<(char, String)> {
    let whole = chunked(thinking, tools, output, usize::MAX);
    for n in [1, 2, 3, 5, 7, 11] {
        assert_eq!(chunked(thinking, tools, output, n), whole, "pezzi da {n}");
    }
    whole
}

#[test]
fn parser_two_typed_calls() {
    let tools = example_tools();
    let output = "L'utente vuole leggere un file. Uso </think di prova e <tool poi.\n</think>\n\nLeggo il file e cerco: è facile.\n\n<tool_call>\n<function=leggi_file>\n<parameter=percorso>\nsrc/main.rs\n</parameter>\n<parameter=riga_iniziale>\n12\n</parameter>\n<parameter=intero>\nfalse\n</parameter>\n</function>\n</tool_call>\n<tool_call>\n<function=cerca>\n<parameter=schema>\nfn main\nfn altro\n</parameter>\n<parameter=opzioni>\n{\"maiuscole\": true, \"tipi\": [\"rs\"]}\n</parameter>\n<parameter=limite>\n5\n</parameter>\n<parameter=file>\nnon json\n</parameter>\n<parameter=sconosciuto>\n42\n</parameter>\n</function>\n</tool_call>\n";
    let ev = all_measures(true, Some(&tools), output);
    println!("{ev:#?}");
    assert_eq!(ev, vec![
        ('R', "L'utente vuole leggere un file. Uso </think di prova e <tool poi.\n".to_string()),
        ('T', "Leggo il file e cerco: è facile.\n\n".to_string()),
        ('C', r#"leggi_file {"percorso":"src/main.rs","riga_iniziale":12,"intero":false}"#.to_string()),
        ('C', r#"cerca {"schema":"fn main\nfn altro","opzioni":{"maiuscole":true,"tipi":["rs"]},"limite":5,"file":"non json","sconosciuto":"42"}"#.to_string()),
    ]);
}

#[test]
fn parser_thinking_off_and_think_unexpected() {
    let ev = all_measures(false, None, "Ciao <b>mondo</b> < 3");
    assert_eq!(ev, vec![('T', "Ciao <b>mondo</b> < 3".to_string())]);
    let ev = all_measures(false, None, "<think>\nboh\n</think>\n\nRisposta.");
    assert_eq!(ev, vec![('R', "\nboh\n".to_string()), ('T', "Risposta.".to_string())]);
    // thinking on, no </think>: all reasoning
    let ev = all_measures(true, None, "penso ancora <tool_call> no");
    assert_eq!(ev, vec![('R', "penso ancora <tool_call> no".to_string())]);
}

#[test]
fn parser_tool_call_not_closed_and_without_schema() {
    // without a schema the values stay strings
    let ev = all_measures(false, None, "<tool_call>\n<function=f>\n<parameter=n>\n7\n</parameter>\n</function>\n</tool_call>");
    assert_eq!(ev, vec![('C', r#"f {"n":"7"}"#.to_string())]);
    // call cut in the middle: comes out as text, whole
    let ev = all_measures(false, None, "Ecco:\n<tool_call>\n<function=f>\n<parameter=n>\n7");
    assert_eq!(ev, vec![('T', "Ecco:\n<tool_call>\n<function=f>\n<parameter=n>\n7".to_string())]);
    // tail that looks like a tag but is not, at the end
    let ev = all_measures(false, None, "fine <tool_ca");
    assert_eq!(ev, vec![('T', "fine <tool_ca".to_string())]);
    // block without <function=...>: text
    let ev = all_measures(false, None, "<tool_call>boh</tool_call>");
    assert_eq!(ev, vec![('T', "<tool_call>boh</tool_call>".to_string())]);
}

#[test]
fn parser_holds_only_the_necessary() {
    let mut p = Parser::new(true, None);
    assert_eq!(p.push("abc</th"), vec![Event::Reasoning("abc".into())]);
    assert_eq!(p.push("ink>\n\nCiao <"), vec![Event::Text("Ciao ".into())]);
    assert_eq!(p.push("b>"), vec![Event::Text("<b>".into())]);
    assert_eq!(p.end(), vec![]);
}
