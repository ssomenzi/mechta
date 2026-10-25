# Numerical reference for M1 (llama.cpp `42d9581`, CPU only)

Produced by a llama.cpp build with the `mechta-dump` example (`examples/mechta-dump/`), in 8 s.
- Model: UD-Q4_K_S (unsloth, with MTP). `-ngl 0 -t 8 -c 1024`.
- `prompt.txt`: a coding question in Qwen chat format, without reasoning (`<think>\n\n</think>`).
  **24 tokens**, then **32 greedy tokens** (argmax). The prompt is processed **one token at a time**:
  the same recurrent DeltaNet path that mechta uses in M1.
- `token.tsv`: for each position 0..55 the token, its text and the 5 best candidates with their logit.
  Generated text: «```python def is_prime(n: int) -> bool: """ Check if a number is prime. Args:».
- `tensors.bin` (545 MB): 11,536 records, format described in the header of `mechta-dump.cpp`
  (name, position, ne[4], float32). Contains, for **every position**:
  - `attn_residual-<il>` and `l_out-<il>` for the 40 layers (x after attention, x after MoE);
  - `result_norm` and `result_output` (logits, 248,320);
  - **all** the F32 tensors of layers **0** (DeltaNet) and **3** (attention): 50 + 37 names.
- Missing: `inp_embd` (it does not go through the callback; it is checked via `attn_norm-0`) and
  `qkv_mixed_transposed-0` (not contiguous; `linear_attn_qkv_mixed-0` is there instead).
