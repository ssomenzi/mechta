# engine

The Cargo workspace of mechta. Safe Rust on top of `wgpu` (Vulkan): every crate is `#![forbid(unsafe_code)]`
except `crates/shader`, which holds the only two `unsafe` calls of the project, each with its safety comment.

| crate | what |
|---|---|
| `crates/gguf` | GGUF reader: header, metadata, tensor table, chunked reads (no mmap); errors, never panics, on a malformed file |
| `crates/shader` | builds the WGSL shader modules (loop guard off, buffer bounds checks on) and enables the experimental matrix units |
| `crates/gpu` | safe `wgpu` layer: buffers, pipelines, dispatch, readback, GPU timestamps |
| `crates/api` | the contract between the HTTP server and an engine: the `Engine` trait, `Params`, `Outcome` |
| `crates/chat` | the official chat templates (Qwen3.6, Flint) rendered with minijinja, and the streaming parser of the model output (reasoning, text, tool calls) |
| `crates/server` | OpenAI-compatible HTTP server on `std::net`: `/v1/chat/completions` (streaming and not), `/v1/models`, `/health`; plus `fake_engine`, a fake model for testing clients |
| `crates/mechta` | the `mechta` binary: the engine (`engine.rs`, `session.rs`, `flint.rs`), the WGSL kernels (`src/wgsl/`) and the verification and benchmark tools |
| `crates/tui` | `mechta-chat`, a terminal chat client |

Other files: `SPEC.md` (the operations of one token, in llama.cpp order), `USAGE.md` (every option and environment
variable), `PARAMETERS.md` and `CALIBRATION.md` (fixed values and how they were chosen), `AUDIT.md` (security review),
`DEVLOG.md` (development log), `reference/` (llama.cpp dumps used to check the kernels, not in git), `tools/`.

## Tests

```
cargo test --release --workspace
```

They run on the CPU (template rendering, output parser, sampler, memory estimate, and the main attention and
operation shaders parsed and validated with `naga` for every KV format). The numerical checks against llama.cpp need the GPU, the model and the reference dumps: see `DEVLOG.md`.
