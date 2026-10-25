# Step 3 of M1: the kernels of one layer against llama.cpp

Command: `mechta layer $(seq 0 55)` (19 s). Output in `risultato.txt`; `err.txt` has the attention per position.
Layer 0 (DeltaNet + MoE) and layer 3 (attention + MoE), all 56 positions of the dump.
Every operation takes its inputs **from the dump**, not from the previous operation.
Error = max|a-b| / max|b|, the worst over all positions.

## Outcome
- **All the kernels are correct.** Against an exact f64 computation on the CPU: error 1e-7 .. 8e-7
  (f32 rounding). This holds for the Q8_0, Q4_K, F32 matrices, for the attention, for everything.
- Element-wise operations against the dump: 0 .. 7e-7. Routing: **the same 8 experts** as
  llama.cpp in all positions, in both layers.
- **Matrices against the dump: 3e-3 .. 4e-2.** This is not an error in mechta: the ggml CPU
  **quantizes x to 8 bits** before the product (Q8_0 for Q8_0 weights, Q8_K for K weights). Redone on the CPU
  with x quantized that way: error against the dump 1e-7 .. 2e-6 (column `ggml-dump`). Explained.
  (Two rows of layer 0, shared gate/up, stay at 6e-5 / 2e-4: not investigated.)
- **Attention against the dump: 2.6e-3.** The dump was made with flash attention: f16 cache, q in
  f16 and **sum of the V in f16** (`VKQ16` in `ggml_compute_forward_flash_attn_ext_f16_one_chunk`).
  Redone that way on the CPU: 2e-5 .. 5.7e-4 depending on the position (without flash attention: 2.4e-3,
  so flash attention was active). The rest is about one f16 step; cause not isolated
  (FMA as in AVX-512: no change).

## What this means for step 4
mechta computes in full f32: it is **more precise** than llama.cpp CPU, not equal to it. The tokens can
differ only where the two best candidates are close. In the 32 generated tokens of the dump the minimum
gap between the first and second logit is **0.58** (positions 23..54).
