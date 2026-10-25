# Benchmarks

All numbers were measured on one machine: a mini PC with an AMD Ryzen 7 255 and its integrated Radeon 780M
(RDNA 3, gfx1103), 16 GB of VRAM set in the BIOS, 64 GB of DDR5 shared with the GPU, Linux with the TTM fix of
`kernel/`. Model: Qwen3.6-35B-A3B, `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` from
[unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF), the file with the
MTP layer.

## mechta, real server

Method: the real server (`mechta serve`), KV cache f16, MTP on with exact acceptance, GPU clock set to `high`.
A real coding conversation grows by about 16K tokens per stage; at each stage the client sends the whole
conversation and the server reuses the prefix already in cache, so the prefill covers only the new tokens.
Prefill and generation speeds come from the server's own log of each request.

| context | 12.6K | 28K | 44K | 60K | 78K | 93K |
|---|---|---|---|---|---|---|
| prefill (tokens/s) | 224 | 193 | 168 | 148 | 132 | 120 |
| generation (tokens/s) | 32.8 | 28.4 | 26.4 | 23.7 | 22.5 | 21.1 |

The prefill uses flash attention on the matrix units of RDNA 3 (`attn_fa_mx`); compared with the same kernel
without matrix units, the prefill is 6% faster at 12.6K and 45% faster at 93K, with the same generation speed.

With the same method, an earlier run up to 114K tokens (GPU clock `auto`) generated 17.6–17.9 tokens/s at 114K;
the MTP drafts were accepted 70–89% of the time across the stages.

## llama.cpp on the same machine

`llama-bench` from upstream llama.cpp, Vulkan backend, the same GGUF, everything on the GPU, flash attention on,
3 repetitions (`-ngl 99 -fa on -p 512 -n 128 -r 3`):

| context already filled | prefill pp512 (tokens/s) | generation tg128 (tokens/s) |
|---|---|---|
| 0 | 449.5 ± 23.5 | 23.37 ± 0.19 |
| 4,096 | 458.9 ± 16.3 | 22.93 ± 0.04 |
| 16,384 | 254.5 ± 16.9 | 21.80 ± 0.08 |
| 65,536 | 74.0 ± 0.9 | 16.46 ± 0.03 |

`llama-bench` does not use the MTP layer, and its prefill is a synthetic 512-token batch, so the two tables do not
measure the same thing: they are a reference, not a race. LM Studio on the same machine logged 29–30 tokens/s of
generation with MTP at short context.

## Memory bandwidth

Generation reads about 2.59 GB of weights per token with this file (dense parts in Q8_0, experts in Q4_K/Q6_K).
Measured read bandwidth: 83 GB/s from the GPU, 59 GB/s from the CPU, about 80 GB/s when both read at once.
The ceiling for one token per step is therefore about 32 tokens/s; MTP goes past it by producing about 1.8 tokens
per step.

## Correctness checks

While the engine was built, every stage was compared with llama.cpp:

- tokenizer: 14,710 tokens out of 14,710 identical on a real session;
- dequantization of Q8_0, Q4_K and Q6_K on the GPU: bit for bit identical to ggml's `to_float` on 8.4 M values;
- generation: 32 greedy tokens out of 32 identical to llama.cpp, with the GPU argmax equal to the CPU argmax.

The day-by-day log of these checks is in `DEVLOG.md`.
