//! CONTRACT between the HTTP server and the engine: changing it means changing both sides.
//! The server uses only this trait; the real engine (crates/mechta) and the fake one (crates/server) implement it.
#![forbid(unsafe_code)]

/// Parameters of a generation. The engine may ignore temperature/top_p/top_k (greedy choice).
#[derive(Clone, Debug)]
pub struct Params {
    /// maximum number of generated tokens (end = Finish::Length if reached)
    pub max_token: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub seed: u64,
    /// false = generate without MTP drafts (one token per step). Request field `mtp` (mechta only).
    pub mtp: bool,
    /// thinking budget in tokens (0 = no limit). When the reasoning reaches it, the engine closes it with the
    /// sentence of the official Qwen method and `</think>`, and the model goes on with the answer.
    pub reasoning_budget: usize,
}

impl Default for Params {
    fn default() -> Self { Params { max_token: 8192, temperature: 0.6, top_p: 0.95, top_k: 20, seed: 0, mtp: true, reasoning_budget: 0 } }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Finish {
    /// the model emitted an end token (<|im_end|> or <|endoftext|>); the end token is NOT in the text
    Stop,
    /// reached `max_token` or the maximum context
    Length,
    /// the `piece` function returned false (client disconnected)
    Cancelled,
}

#[derive(Clone, Debug)]
pub struct Outcome {
    pub token_prompt: usize,
    /// prompt tokens NOT reprocessed because already in cache from the previous turn (prefix reuse)
    pub reused_tokens: usize,
    pub generated_tokens: usize,
    pub end: Finish,
    pub ms_prefill: f64,
    pub ms_generation: f64,
}

/// A generation engine. It lives on ONE thread (created in there): Send is not needed.
pub trait Engine {
    /// id of the model exposed in /v1/models
    fn model_name(&self) -> &str;
    /// maximum tokens (prompt + generated)
    fn max_context(&self) -> usize;
    /// number of tokens of the text (for checks and usage). A tokenizer error is an error
    /// of the request (the server answers 400), not a panic on the engine thread.
    fn count_tokens(&self, text: &str) -> Result<usize, String>;
    /// Generates from `prompt`, text ALREADY formatted by the chat template (ends with
    /// "<|im_start|>assistant\n<think>\n" if thinking is on). The engine tokenizes, reuses the prefix shared with
    /// the previous request, and calls `piece` with the NEW decoded text (complete UTF-8, never half a character)
    /// as it generates. If `piece` returns false it stops (Finish::Cancelled). The end token is not passed.
    fn generate(&mut self, prompt: &str, p: &Params, piece: &mut dyn FnMut(&str) -> bool) -> Result<Outcome, String>;
}
