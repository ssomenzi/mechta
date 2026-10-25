# mechta

An LLM inference engine written from scratch in Rust, for one model and one class of GPU: **Qwen3.6-35B-A3B on
the AMD Radeon 780M**, the integrated GPU of Ryzen mini PCs. It serves an OpenAI-compatible API, so coding agents
such as opencode can use it as a local provider.

It is a personal project and a challenge: how far can a small engine go when it is built for exactly one model on
exactly one machine, instead of being a general runner like llama.cpp.

## What it does

- **OpenAI-compatible server**: `/v1/chat/completions` with streaming, tool calls, reasoning (`reasoning_content`)
  on or off per request, and an optional reasoning budget.
- **Multi-token prediction (MTP)**: the model's own MTP layer drafts the next token and the engine verifies two
  tokens per step. The output has the same distribution as without MTP.
- **Prefix cache across sessions**: the state after a long system prompt is saved to disk and reloaded, so a new
  agent session does not recompute it.
- **Long context**: up to 262,144 tokens, KV cache in f16 (default), f32 or q8. A memory check before the start
  says whether the chosen context fits.
- **GPU kernels in WGSL** on `wgpu` (Vulkan), including flash attention on the matrix units of RDNA 3 for the prefill.
- **Safe Rust**: every crate forbids `unsafe` except one small crate with two documented calls.
- **Checked against llama.cpp**: tokenizer, dequantization and generated tokens were compared with llama.cpp step
  by step while the engine was built.

## Speed

Measured on the real server with a real agent conversation (Radeon 780M, 16 GB VRAM, Qwen3.6-35B-A3B UD-Q4_K_S,
f16 KV cache, MTP on):

| context | 12.6K | 28K | 44K | 60K | 78K | 93K |
|---|---|---|---|---|---|---|
| prefill (tokens/s) | 224 | 193 | 168 | 148 | 132 | 120 |
| generation (tokens/s) | 32.8 | 28.4 | 26.4 | 23.7 | 22.5 | 21.1 |

For reference, on the same machine `llama-bench` generates 23.4 tokens/s without MTP at empty context, and LM
Studio logs 29–30 tokens/s with MTP. Method, more numbers and the correctness checks: [engine/BENCHMARKS.md](engine/BENCHMARKS.md).

## Supported models and hardware

- **Models**: Qwen3.6-35B-A3B and fine-tunes with the same architecture (tested: Tiel-Coder-35B-A3B,
  Ornith-1.5-35B-A3B). Experimental: Flint 40B-A2.5B. Quantizations Q8_0, Q3_K, Q4_K, Q6_K, Q4_1, F32, BF16.
- **GPU**: developed and tested only on the Radeon 780M (RDNA 3, gfx1103) with 16 GB of VRAM set in the BIOS and
  64 GB of system RAM. With a 64K context the default model takes about 23 GB of GPU memory (VRAM plus GTT).
  Other Vulkan GPUs may work but are untested.
- **OS**: Linux. On the 780M, Linux 7.1 and 7.2 kernels without a recent TTM fix can crash the system when GPU memory is
  swapped out; the fix and how it was tested are in `kernel/`.

## Build

```
cd engine
cargo build --release
```

This builds `engine/target/release/mechta` (engine and server) and `engine/target/release/mechta-chat`
(terminal chat client).

## Run

Download the GGUF (tested: `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` from
[unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF)) and point the engine
at it, at its tokenizer (a copy is in `tokenizers/`) and at a folder for the prefix cache:

```
export MECHTA_MODEL=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
export MECHTA_TOKENIZER=tokenizers/qwen3.6-tokenizer.json
export MECHTA_CACHE=prefix-cache
engine/target/release/mechta serve --ctx 65536
```

The model loads in about 10 seconds, then the server listens on `http://127.0.0.1:4131/v1`:

```
curl http://127.0.0.1:4131/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Hello"}],"stream":false}'
```

The server listens on localhost only. To open it to the network set `MECHTA_ADDRESS=0.0.0.0:4131`, and set
`MECHTA_API_KEY` too: every request to `/v1/*` must then carry `Authorization: Bearer <key>`. See
[engine/SECURITY.md](engine/SECURITY.md).

`engine/mechta-serve.sh` starts the same server as a systemd user unit, after a memory check. All options and
environment variables are in [engine/USAGE.md](engine/USAGE.md).

### opencode

Add the `provider.mechta` block of `engine/opencode-mechta.json` to your `opencode.json` and choose the model
`mechta/qwen3.6-35b-a3b`.

## Repository

| folder | what |
|---|---|
| `engine/` | the Cargo workspace: crates, WGSL kernels, start script, documentation ([engine/README.md](engine/README.md)) |
| `tokenizers/` | the tokenizers of the supported models (third-party, see `THIRD-PARTY.md`) |
| `kernel/` | the Linux kernel fix for the TTM crash on the 780M |
| `model.md` | a study of the model: architecture, sizes, memory per token |

How it was built, step by step: [engine/DEVLOG.md](engine/DEVLOG.md).

## License

MIT, see `LICENSE`. Tokenizers, chat templates and documents copied from other projects (Qwen, Unsloth,
RecursiveSI, DwarfStar) keep their own license: see `THIRD-PARTY.md`.
