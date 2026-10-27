# Benchmarks

All numbers were measured on one machine: a mini PC with an AMD Ryzen 7 255 and its integrated Radeon 780M
(RDNA 3, gfx1103), 64 GB of DDR5 shared with the GPU, 2 GB of it set as VRAM in the BIOS and the rest mapped as GTT, Linux with the TTM fix of
`kernel/`. Model: Qwen3.6-35B-A3B, `Qwen3.6-35B-A3B-UD-Q4_K_S.gguf` from
[unsloth/Qwen3.6-35B-A3B-MTP-GGUF](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF), the file with the
MTP layer.

## mechta, real server

Method: the real server (`mechta serve`, context 131072), KV cache f16, MTP on with exact acceptance, GPU clock set
to `high`. A real coding conversation grows by about 16K tokens per stage; at each stage the client sends the whole
conversation and the server reuses the prefix already in cache, so the prefill covers only the new tokens. Each
stage is answered twice (400 tokens, temperature 0.6). The server was started four times, in the order double step,
4-token step, 4-token step, double step; generation is the generated tokens divided by the generation time of the
four answers of each kind of step. Prefill and generation speeds come from the server's own log of each request. GPU memory
taken: 24.6 GB.

| context | 12.6K | 28K | 44K | 60K | 78K | 93K | 109K |
|---|---|---|---|---|---|---|---|
| prefill (tokens/s) | 219 | 192 | 168 | 148 | 133 | 119 | 110 |
| generation, 4-token step (tokens/s) | 34.6 | 30.9 | 29.2 | 27.6 | 26.3 | 24.5 | 24.7 |
| generation, double step (tokens/s) | 32.2 | 29.0 | 27.1 | 25.5 | 24.4 | 22.5 | 22.0 |
| change | +7.4% | +6.6% | +7.8% | +8.0% | +8.1% | +9.1% | +12.2% |
| MTP drafts accepted, 4-token step | 52% | 48% | 47% | 48% | 49% | 47% | 51% |
| MTP drafts accepted, double step | 77% | 72% | 72% | 72% | 77% | 71% | 75% |

The 4-token step (the default) checks the sure token and 3 MTP drafts in one pass: the MTP layer drafts the first
from the rows just accepted, then 2 more in a chain. The double step checks the sure token and 1 draft
(`MECHTA_STEP4=0`). Generation changes from answer to answer with the share of accepted drafts: at 78K two starts
of the 4-token step gave 28.0 and 24.9 tokens/s.

With greedy sampling (same series, two starts each) the gain is larger, because the 2nd and 3rd drafts are accepted
more often:

| context | 12.6K | 28K | 44K | 60K | 78K | 93K | 109K |
|---|---|---|---|---|---|---|---|
| generation, 4-token step (tokens/s) | 36.4 | 31.0 | 31.3 | 29.4 | 25.8 | 26.3 | 26.0 |
| generation, double step (tokens/s) | 31.6 | 28.8 | 27.6 | 25.4 | 22.5 | 22.7 | 21.5 |
| change | +15.1% | +7.6% | +13.5% | +15.7% | +14.5% | +16.0% | +20.8% |

The prefill uses flash attention on the matrix units of RDNA 3 (`attn_fa_mx`); compared with the same kernel
without matrix units, the prefill is 6% faster at 12.6K and 45% faster at 93K, with the same generation speed.

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
The ceiling for one token per step is therefore about 32 tokens/s. With MTP one step checks several tokens: the
double step produces about 1.74 tokens per step (74% of the drafts accepted at temperature 0.6), the 4-token step
about 2.47 (49% of the drafts accepted). A step that checks more tokens costs more than a single one (94 ms for the
4-token step at 109K, greedy), so generation stays under that ceiling times the tokens per step.

## Correctness checks

While the engine was built, every stage was compared with llama.cpp:

- tokenizer: 14,710 tokens out of 14,710 identical on a real session;
- dequantization of Q8_0, Q4_K and Q6_K on the GPU: bit for bit identical to ggml's `to_float` on 8.4 M values;
- generation: 32 greedy tokens out of 32 identical to llama.cpp, with the GPU argmax equal to the CPU argmax.

The day-by-day log of these checks is in `DEVLOG.md`.
