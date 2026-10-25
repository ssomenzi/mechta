# Step 4 of M1: the whole model against llama.cpp

Command: `mechta generate` (21 s including loading). Output in `risultato.txt`.
Prompt: the 24 tokens of the dump. Then mechta picks the most probable token and feeds it back in,
for 32 tokens. Each step uses mechta's outputs (not the dump's): errors accumulate as in real use.
Everything in f32, KV cache included.

## Outcome
- **Most probable token equal to llama.cpp in all 56 positions.** The 32 generated tokens are
  **identical**: «```python def is_prime(n: int) -> bool: """ Check if a number is prime. Args: n».
- Even where mechta's first two candidates are close (gap 0.025 at position 2, 0.012 at
  position 10, both in the prompt) the choice is the same.
- Logit error against the dump (max|a-b| / max|b|): 7e-3 .. 1e-1. x at the end of a layer: up to 1.5e-1.
  Step 3 showed that the kernels are exact (<= 8e-7) and that the differences come from llama.cpp
  (x in 8 bits, attention in f16); here they accumulate over 40 layers. Whether the sum is **entirely**
  explained by this has not been verified.
- Weights on the GPU 2.01 GB (everything except experts and embedding). Experts: the 8 chosen ones read from the file
  at every layer. About 330 ms per token: no optimization, speed is M2.
