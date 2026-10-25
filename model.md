# Qwen3.6-35B-A3B Q4_K_M — what we know about the model

Sources: the official `config.json` on Hugging Face and the GGUF header. Numbers marked "calculated" are derived, not measured.

## 1. The model, verified facts

### General shape
| | |
|---|---|
| architecture | `qwen3_5_moe` (GGUF: `qwen35moe`) |
| hidden | 2048 |
| layers | 40 = 10 × (3 DeltaNet + 1 full attention) |
| vocabulary | 248,320 (BPE gpt2, pre `qwen35`) |
| native context | 262,144 |
| active parameters | ~2.95 B per token (1.13 B experts + ~1.8 B dense, calculated) |
| MTP | 1 MTP layer in the original model; **absent from the GGUF Q4_K_M** (733 tensors, no `nextn`/`mtp`) |
| vision | ViT 27 layers, hidden 1152, patch 16 — separate `mmproj` file, 0.9 GB BF16 |
| default sampling | temp 1.0, top_k 20, top_p 0.95 |

### DeltaNet layers (30 of 40) — "linear attention"
- 16 QK heads of 128, 32 V heads of 128 (V heads are twice the QK heads).
- Projections: `attn_qkv` 2048→8192 (q 2048 + k 2048 + v 4096), `attn_gate` (z) 2048→4096,
  `ssm_alpha`/`ssm_beta` 2048→32, `ssm_out` 4096→2048.
- Causal 1D convolution, kernel 4, over 8192 channels.
- Recurrent state: 32 heads × 128×128 **in f32** (`mamba_ssm_dtype`).
- **Fixed state: 63 MB + 3 MB of conv, whatever the context length.**

### Full attention layers (10 of 40)
- 16 Q heads, **2 KV heads**, head_dim 256 (GQA 8:1).
- `attn_q` 2048→8192 = Q + **output gate** (`attn_output_gate: true`).
- q_norm and k_norm (per-head RMSNorm).
- **Partial** RoPE: only 64 dimensions out of 256, theta 10,000,000, interleaved mRoPE [11,11,10]
  (for text only the three sections have the same position → equivalent to 1D RoPE).
- **KV cache: 20 KB/token in f16** → 2.7 GB at 128K, 5.4 GB at 262K.

### MoE (all 40 layers)
- 256 experts, 8 routed + 1 shared, intermediate 512, SiLU (gate/up/down).
- F32 router 2048×256 per layer; the shared expert also has its own scalar F32 gate.

### Quantization of the Q4_K_M (from the header)
| tensors | type | GB |
|---|---|---|
| experts gate + up | Q4_K | 12.08 |
| experts down | half the layers Q6_K, half Q4_K | 7.42 |
| output head | Q6_K | 0.42 |
| embedding | Q4_K | 0.29 |
| DeltaNet/attn projections | mixed Q4_K + Q6_K | ~0.8 |
| router, norms, conv, ssm_a | F32 | ~0.09 |
| **total** | | **21.16** (matches the file: 21.17) |

---

## 2. What follows for decode and prefill (calculated, to be measured)

### Decode: bytes read per token = **1.97 GB**
| part | MB | % |
|---|---|---|
| DeltaNet projections (30 layers) | 630 | 32 |
| routed experts (8 × 40) | 610 | 31 |
| output head Q6_K | 417 | 21 |
| full attention projections | 154 | 8 |
| F32 router | 84 | 4 |
| shared experts | 76 | 4 |

- **Surprise: the experts are only 1/3 of the bytes.** Two thirds are dense parts, identical
  for every token: always the same, 1.3 GB. These can always stay in the GPU's 16 GB.
- The output head alone is 1/5: 248K rows for every token.
- Theoretical ceiling: 30 tok/s at 60 GB/s, 46 tok/s at 90 GB/s. The real bandwidth must be **measured**.
- KV cache and DeltaNet state are small: decode stays weight-bound even at 128K.

### Prefill
- With a block of N tokens, experts touched per layer (uniform routing, hypothesis):
  N=32 → 163/256; N=128 → 252/256; N=512 → all of them, 16 tokens each.
- So from N≈128 the prefill reads **all 21 GB per block**: it becomes a series of
  small matmuls (4–64 rows per expert). The bottleneck moves from bandwidth to compute.
- DeltaNet in prefill: **chunkwise** form; in decode: **recurrent** form.
  Two different kernels for the same layer.
- Quadratic attention only in 10 layers out of 40, with 2 KV heads.

---

## 3. The questions to ask ourselves before the architecture

**About the model**
1. Is expert routing really uniform? How "hot" are some experts on real texts (code, Italian)? → decides whether an expert cache makes sense.
2. Does the routing of nearby tokens repeat? → decides whether prefetching is possible.
3. Do we need MTP? It is not in the GGUF: using it means converting it ourselves from the original safetensors.
4. How much precision does the DeltaNet state need (f32 mandatory? does bf16 hold up?).
5. Do we need vision in the first round, or text only?

**About the machine** (all to be measured, no data today)
6. Real RAM bandwidth from the CPU, and from the GPU (GTT) on the 780M.
7. Real compute throughput: AVX-512 on 8 cores versus the 780M, for small Q4_K matmuls.
8. Cost of moving data CPU↔GPU: on the same RAM is it nearly zero, or not?
9. Which GPU allocation does **not** trigger the TTM bug (all at once, no eviction)?

**About usage**
10. Typical context: 8K, 32K, 128K? How much does prefill matter (agent with long prompts) versus decode (chat)?
11. A single user and a single session, or several sessions? KV/state on disk like ds4?

**About the language**
12. With safe Rust: `read` into `Vec` instead of `mmap`; SIMD via autovectorization; GPU with `wgpu` (no int8 dot product). What does each cost? → to be measured with micro-benchmarks.

