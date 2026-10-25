//! FAKE engine to try the server without a GPU: canned answers, in small pieces, with ~15 ms between one piece and
//! the next. Listens on 127.0.0.1:4132 (4131 belongs to the real server). Usage: `fake_engine [address]`.
#![forbid(unsafe_code)]

use std::time::{Duration, Instant};

use mechta_api::{Outcome, Finish, Engine, Params};
use mechta_server::{Config, start};

struct Fake;

const REASONING: &str = "The user has written me a message. This is a fake engine: \
I do not really think, but I pretend to, to test the reasoning channel.\n";
const ANSWER: &str = "Hello! I am the fake engine of mechta. This answer arrives in small pieces, \
with accents (café, naïve, déjà vu) to test UTF-8.";
const REASONING_TOOL: &str = "I need information from the project: I use the first available tool.\n";
const SENTENCE_TOOL: &str = "Checking with a tool.";

/// First tool declared in the prompt (JSON line after "<tools>"): name and one parameter with a plausible value.
fn first_tool(prompt: &str) -> Option<(String, String, String)> {
    let after = &prompt[prompt.find("<tools>")? + "<tools>".len()..];
    let row = after.lines().map(str::trim).find(|r| r.starts_with('{'))?;
    let v: serde_json::Value = serde_json::from_str(row).ok()?;
    let f = if v.get("function").is_some() { &v["function"] } else { &v };
    let name = f["name"].as_str()?.to_string();
    let param = f["parameters"]["required"][0]
        .as_str()
        .map(str::to_string)
        .or_else(|| f["parameters"]["properties"].as_object().and_then(|o| o.keys().next().cloned()))
        .unwrap_or_else(|| "input".to_string());
    let value = match param.as_str() {
        "command" => "ls",
        "filePath" | "path" | "file_path" => ".",
        "pattern" => "**/*.rs",
        "url" => "https://example.com",
        "description" => "List the files",
        _ => "test",
    };
    Some((name, param, value.to_string()))
}

/// Splits into small pieces of ~4 characters (never in the middle of a UTF-8 character).
fn bits(s: &str) -> Vec<String> {
    let c: Vec<char> = s.chars().collect();
    c.chunks(4).map(|p| p.iter().collect()).collect()
}

impl Engine for Fake {
    fn model_name(&self) -> &str { "qwen3.6-35b-a3b" }
    fn max_context(&self) -> usize { 65536 }
    fn count_tokens(&self, text: &str) -> Result<usize, String> { Ok(text.chars().count().div_ceil(4)) }

    fn generate(&mut self, prompt: &str, p: &Params, piece: &mut dyn FnMut(&str) -> bool) -> Result<Outcome, String> {
        // to test the server watchdog: an engine that panics must make the process exit
        if prompt.contains("PANICO_FINTO") {
            panic!("test panic requested by the prompt");
        }
        let t0 = Instant::now();
        // fake prefill; MECHTA_FAKE_WAIT_S=N simulates a long prefill (test of the client timeout)
        let wait = std::env::var("MECHTA_FAKE_WAIT_S").ok().and_then(|v| v.parse::<u64>().ok()).map(|s| s * 1000).unwrap_or(30);
        std::thread::sleep(Duration::from_millis(wait));
        let ms_prefill = t0.elapsed().as_secs_f64() * 1000.0;
        let think = prompt.ends_with("<think>\n");
        // after a tool response it answers in words: otherwise opencode would call tools forever
        let tool = if prompt.contains("<tool_response>") { None } else { first_tool(prompt) };
        let mut text = String::new();
        match &tool {
            Some((name, par, val)) => {
                // MECHTA_FAKE_TOOL_PIECES=N: the tool argument made of N pieces, to test a long tool call that
                // the parser holds back (no bytes to the client while it is generated)
                let val = match std::env::var("MECHTA_FAKE_TOOL_PIECES").ok().and_then(|v| v.parse::<usize>().ok()) {
                    Some(n) => "line of code\n".repeat(n / 3 + 1),
                    None => val.to_string(),
                };
                if think {
                    text.push_str(REASONING_TOOL);
                    text.push_str("</think>\n\n");
                }
                text.push_str(SENTENCE_TOOL);
                text.push_str(&format!(
                    "\n\n<tool_call>\n<function={name}>\n<parameter={par}>\n{val}\n</parameter>\n</function>\n</tool_call>"
                ));
            }
            None => {
                if think {
                    text.push_str(REASONING);
                    text.push_str("</think>\n\n");
                }
                text.push_str(ANSWER);
            }
        }
        let t1 = Instant::now();
        let mut generated = 0;
        let mut end = Finish::Stop;
        for pz in bits(&text) {
            if generated >= p.max_token {
                end = Finish::Length;
                break;
            }
            std::thread::sleep(Duration::from_millis(15));
            generated += 1;
            if !piece(&pz) {
                end = Finish::Cancelled;
                break;
            }
        }
        Ok(Outcome {
            token_prompt: self.count_tokens(prompt)?,
            reused_tokens: 0,
            generated_tokens: generated,
            end,
            ms_prefill,
            ms_generation: t1.elapsed().as_secs_f64() * 1000.0,
        })
    }
}

fn main() {
    let address = std::env::args().nth(1).unwrap_or_else(|| "127.0.0.1:4132".to_string());
    // MECHTA_API_KEY as the real server, to try the key with the fake engine
    let api_key = std::env::var("MECHTA_API_KEY").ok().filter(|k| !k.is_empty());
    start(Box::new(|| Box::new(Fake) as Box<dyn Engine>), Config { address, api_key, ..Config::default() });
}
