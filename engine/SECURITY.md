# Security

## Threat model

mechta is a local inference server for one machine.

- It listens on `127.0.0.1:4131` by default (`MECHTA_ADDRESS` changes it). Only processes on the same machine
  can reach it.
- There is no authentication unless `MECHTA_API_KEY` is set. With a key set, every `/v1/*` endpoint requires
  `Authorization: Bearer <key>` (compared in constant time) and answers 401 otherwise; `/health` stays open.
  `mechta-chat` sends the key if it finds it in the same variable.
- To open the server to a network (for example `MECHTA_ADDRESS=0.0.0.0:4131`), put it behind a firewall that
  only admits the hosts you trust **and** set `MECHTA_API_KEY`. Traffic is plain HTTP: the key and the
  conversation are not encrypted, so do not expose the port to an untrusted network or to the internet.
- The model file (GGUF) and the tokenizer are chosen by the operator, but a downloaded file can be broken or
  hostile, so the parser treats it as untrusted input.

## Protections in place

HTTP server (`crates/server`):

- **Connection limit**: at most 32 concurrent connections (`MAX_CONNECTIONS`); beyond that the server answers
  `503` immediately.
- **Request body**: at most 16 MiB (`MAX_BODY`, `413` beyond that), headers at most 64 KiB. The body is read
  in chunks, so memory grows with the bytes actually received, not with the declared `Content-Length`.
  Chunked request bodies are refused with `411`.
- **Slow clients**: the whole request (headers and body) must arrive within 30 s (`READ_DEADLINE`); otherwise
  the connection is closed with `408 Request Timeout`. The socket timeout is recomputed before every read,
  so a client sending one byte at a time cannot hold a thread.
- **Fixed routes**: only three fixed paths; no files are served, so there is no path traversal. One request
  per connection with `Connection: close`.
- **Tokenizer errors** on client input are returned as `400` instead of stopping the engine thread.
  Chat template errors are also returned as `400`.
- **Control characters**: text from the client that ends up in the log or in an error response (for example
  an unknown path) has control characters replaced and is cut to 200 characters, so it cannot inject ANSI
  sequences or fake log lines.

GGUF parser (`crates/gguf`):

- Returns an error on malformed files instead of panicking: checked arithmetic on tensor sizes and offsets,
  `general.alignment` must be non-zero and a power of two, nested arrays at most 8 levels deep, strings at
  most 64 MiB and read with a bounded reader, at most 8 dimensions per tensor, bounded preallocation.
  `mechta inspect` prints the error and exits with status 1.

Terminal client (`crates/tui`):

- Control characters in the model's output are removed before rendering (newlines and tabs are kept), so a
  prompt cannot make the model drive the terminal with escape sequences.

### `unsafe` code

All crates except `crates/shader` contain no `unsafe` code; most declare `#![forbid(unsafe_code)]`.
`crates/shader` has two documented `unsafe` blocks:

1. `create_shader_module_trusted`: shader modules are created with all wgpu runtime checks **except** the
   guard against infinite loops (`force_loop_bounding = false`), which costs ×1.25 to ×3 on the kernels.
   Buffer bounds checks stay on.
2. `wgpu::ExperimentalFeatures::enabled()`: needed to request cooperative-matrix support for one attention
   kernel, which is checked against a CPU reference.

## Known limitations

- `cargo audit` reports the `paste` crate as unmaintained. It comes in through `tokenizers` and can only be
  removed by an upstream release; it is a build-time macro crate, not a known vulnerability.
- The shader infinite-loop guard is disabled by design (see above). The contract is that every loop in the
  shaders has a constant bound, or a bound read from a finite uniform or push constant, and a step that always
  advances. A shader that broke this contract could hang the GPU.

## Reporting a vulnerability

Open an issue on GitHub that describes the problem and the affected component, **without exploit details**,
or, preferably, open a private security advisory from the repository's *Security* tab on GitHub. Details can
then be shared privately.
