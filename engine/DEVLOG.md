# mechta — development log

The day-by-day log of how the engine was built, from the first token (M1) to the working prototype. Day 1 is the first
day of the project; numbers and decisions are those of the moment. For what the engine is today see `README.md` (this folder) and the top-level README.

## Day 3
- M1: formats of the UD-Q4_K_S file read from the header:
  dense parts Q8_0, experts Q4_K/Q6_K, output Q6_K, router and DeltaNet F32, 41 blocks (40 = MTP).
  **2.59 GB per token**, ceiling ~32 tokens/s (M0 corrected accordingly).
- M1: `SPEC.md` written from the llama.cpp graph `42d9581`.
- M1: `mechta inspect`: 753 tensors, end of the last one = file length
  (21,388,319,008 bytes) ✅. `mechta token`: **14,710 tokens out of 14,710 identical to llama.cpp** on the
  "pomodoro" session ✅.
- M1: **step 1 done**: reference dump in `reference/` (see its README):
  24 prompt tokens + 32 greedy, 40 layers, logits at every step, everything of layers 0 and 3.
- M1: **step 2 done**: Q8_0, Q4_K, Q6_K decoding in WGSL
  (`crates/mechta/src/wgsl/dequant.wgsl`), compared with ggml's `to_float` on 6 chunks of real tensors
  (8.4 M values: token_embd, blk.0.attn_qkv, blk.0.ffn_gate/down_exps, output, blk.34.ffn_down_exps).
  **Bit for bit identical: 100%, max ulp 0** ✅. Result in `reference/deq/risultato.txt`.
- M1: **step 3 done**: one-layer kernels in `crates/mechta/src/wgsl/` (`matvec`,
  `ops`, `routing`), test `mechta layer` on layer 0 (DeltaNet) and 3 (attention), 56 positions.
  All correct against an exact f64 computation (error <= 8e-7); same experts as llama.cpp everywhere.
  The differences from the dump (up to 4e-2 in the matrices, 2.6e-3 in the attention) come from llama.cpp:
  x quantized to 8 bits and flash attention with f16 accumulation, verified by redoing its computations.
  Details: `reference/strato/README.md`.
- **step 4 done, M1 closed**: `crates/mechta/src/engine.rs` (40 layers, f32, experts
  read from the file at every token), test `mechta generate`: **56 positions out of 56 with the same most likely
  token as llama.cpp, 32 generated tokens identical**. Details: `reference/genera/README.md`.
- M2, measurement 1, where the time goes (`mechta profile`):
  **301 ms per token** (3.3 t/s). Experts copied from the file 68 ms, waits on the 81 submits 52 ms,
  K-quant kernel with 8 threads out of 64 working 91 ms, single-thread routing 18 ms, dense Q8_0 31 ms.
- M2, fixes 1: matvec in chunks of 32 values (all threads working), routing on
  256 threads. Exact as before. With the GPU clock fixed at 2600 MHz: **241 → 160-166 ms per token**
  (GPU kernels 122 → 59 ms). In `auto` the GPU stays at 0.8-1.1 GHz under this load and costs ~90 ms.
- M2: experts on the GPU, one submit per token, commands built only once:
  **57.4-57.7 ms per token = 17.4 tokens/s** (auto clock). 20.5 GB of GPU memory. The first run had
  taken 47 GB of GTT: 256 MiB blocks of the default allocator and temporary copies never freed;
  fixed with `MemoryHints::MemoryUsage` and a wait after every tensor (`mechta memtest`).

## Day 4
- M2: page cache corruption during the GPU runs:
  the cause found is WHERE the buffers the GPU writes to live. The engine now creates them first, in VRAM
  (VRAM at 16 GB from the BIOS): **55 runs out of 55 clean**. The old order (`MECHTA_ORDER=weights-first`) corrupts
  the page cache and kernel memory and gives wrong tokens: for tests only.
- M2: the Q4_K kernel moves to `matvec3.wgsl` (x read as vec4, scales as words,
  `d1 (q . x) - m1 (sum x)`): experts 46 → 62 GB/s, **-1.8 ms per token** (53.1 → 51.2-51.6 at clock 2600),
  32/32 tokens equal. The GPU clock fixed at 2600 (`power_dpm_force_performance_level=high`) does not change the
  token speed compared to `auto`.
- M2: `delta` kernel rewritten (state `o[h][i][j]` instead of `o[h][j][i]`, column in registers:
  one read and one write per value): 4.1 → 1.24 ms per token, **-2.5 ms** (51.2 → 48.3 at clock 2600),
  32/32 tokens equal. The old one remains as `delta1` (`MECHTA_DELTA=v1`). Benchmark: `mechta bench-delta <1|2|3>`.
- M2: Q8_0 in `matvec4.wgsl` (9 words per chunk realigned with a funnel shift, `unpack4xI8`,
  x as vec4): the dense matrices go from 60-66 to 77-82 GB/s, the measured read-only ceiling is 81-83.
  **-3.3 ms** (48.3 → 44.9 at clock 2600), 32/32 tokens equal. matvec4 is the default for Q8_0 and Q4_K.
  Since the start of the day: 53.1 → 44.9 ms (22.3 t/s); llama.cpp 42.7.
- M2: Q6_K in `matvec4.wgsl` with the same method as Q8_0 (realigned words, `unpack4xU8`):
  output head 7.5 → 5.0 ms (56 → 83 GB/s, the ceiling). **-2.5 ms** (44.9 → 42.5). matvec4 is the default for
  Q8_0, Q4_K and Q6_K. **Since the start of the day 53.1 → 42.5 ms (23.5 t/s), 32/32 tokens: the speed of M0 llama.cpp (42.7)** —
  but M0 was measured with 2 GB VRAM and auto clock: to be re-measured under equal conditions. Outside the kernels ~4.8 ms remain.
- M2, CPU side: measured with two timestamps that the ~1100 dispatches cost only 0.6 ms of gaps.
  Then: argmax on the GPU with the 4-byte readback in the same submit (`Engine::step`, -0.8 ms) and busy waiting
  after the submit (default, `MECHTA_POLL=sleeping` to turn it off, -0.6 ms). **40.8-41.0 ms (24.4 t/s)**, 32/32.
  Outside the GPU 2.6 ms remain, of which 1.9 is driver latency around the submit.
- **M2 closed: 40.9 ms per token (24.4 t/s), 32/32.** From 53.1 at the start of the day (-23%), from 301 on Day 3.
  The big kernels are at the read ceiling (81-83 GB/s); what is missing from the 83 average is work without weights, small
  expert dispatches and driver latency. Moving on to M4 (MTP).
- **M4: the MTP layer (blk.40) runs in mechta** and gives the same argmax as llama.cpp in 55 drafts out of 55
  (reference `reference-mtp/`, produced by `examples/mechta-dump-mtp`). A single block function for the 41 layers;
  `mechta mtp` for the check. n-column benchmark (`matvec_n4.wgsl`): n = 2 almost free, n ≥ 3 pays for the x cache.
  Next: the 2-token step (draft + verification in one read of the weights).
- **M4: the double step runs** (2 tokens in one pass over the weights: `programma2`, kernels `conv2`, `delta2`,
  `rope2`, `attn2`, `routing_b`): l_out and argmax equal to the M1 dump in 4 pairs of positions (`mechta step2`).
- M4: double step correct (27/27 argmax after fixing the conv parity, now an internal
  counter) but **slow: 58.5 ms against 40.4 (x1.45)**: experts twice (x1.97), matvec k = 4096 +60%
  (2-column x out of the cache), delta2 x1.8. With this ratio MTP is worth ×1.06-1.22; it needs to drop to ~1.15.
- M4: experts merged in the double step (`merge.wgsl`, `matvec_u.wgsl`): 58.9 → 56.4 ms (x1.40),
  27/27 argmax. Floor of the double step ≈ 48 ms (it reads ~1.19x the weights of a single step); on top there are ~8 ms of
  double columns on the dense parts (k = 4096: +60%, x out of the cache) and delta2.
- **M4: the MTP loop generates.** `mechta generate-mtp`: **30.4 t/s against 24.4 (x1.25)**, 32/32 tokens
  identical to llama.cpp and to plain generation, 16/16 drafts accepted on this prompt. Rejection branch to be tested.
- M4: rejection branch verified (drafts deliberately broken every 1/2/3 steps: 32/32 tokens
  identical). Speed as a function of acceptance: 100% 30.4 t/s, 68% 25.4, 52% 22.4, 0% 15.0; break-even at α ≈ 0.61.
  **M4 v1 complete.** Next: double step towards the floor (48 ms) and a lighter draft.
- M4: double draft with the one-column output head: **30.7-31.5 t/s (x1.26-1.31)**, 32/32.
- M4: double draft with MoE only on the second token, correct (bug found with `mechta mtp2`:
  `mv_x` on F32 used matvec.wgsl, without x_off). **31.7-32.7 ms/token = 30.6-31.5 t/s**, 16/16 drafts, 32/32 tokens.
- M4: x in f16 in the double step (f16 producers, matvec_nh/matvec_uh): double step 56.4 → 50.9 ms
  (x1.25 of the single one, at the floor). **MTP: 28.6-29.4 ms/token = 34.1-34.9 t/s (x1.40)**, 32/32 tokens, 16/16 drafts.
- M4: reduced head for the drafts (`MECHTA_DRAFT_HEAD=frequenze.tsv`, 32.6K rows = the tokens that opencode
  sessions use in output): on the code prompt (828 tokens) MTP 35.1 → 33.3 ms/token (30 t/s), 180/180 identical.
  ⚠️ Context costs: single step 40.9 ms at ~30 tokens, 48.8 at ~900. To be profiled (attention sequential over positions).

## Day 5
- M4: attention `attn_v3` (parallel over positions, K as vec4): at context 900 the attention goes from
  7.8 to 2.7 ms, the token from 47.9 to 43.0; MTP on the code prompt 30.7 t/s. Context stays limited to 1024 (CTX, `punt`).
- M4: flash-decoding attention `attn_fd` (K/V read once for the 8 Q heads, blocks of 128 positions +
  merge) and **CTX 1024 → 8192**. At 900 the attention 2.7 → 2.1 ms; at 4300 of context 5.4 ms (token 45.9 ms = 21.8 t/s, first
  step beyond 1K). Still 2.4x below the bandwidth at 4300: next attempt K in shared memory in contiguous chunks.
- M4: measured which phase of `attn_fd` weighs (it was V, not K): V as vec4 with 4 lanes, merge over
  the valid blocks only. Attention at 900: 2.06 → 1.09 ms (token 41.2 ms); at 4300: 5.5 → 5.0 ms. MTP on code 31.6 t/s.
  New tools: `MECHTA_FAKE_POS=N` (profile at context N without prefill), `MECHTA_FD_PHASE=1|2`.
- M4: `attn_fd` K phase with 2 threads per position: attention at 900 0.86 ms, at 4300 3.3 ms (~57 GB/s).
  MTP on code 31.9 t/s. The context cost is closed for decode: at 10K ~7.5 ms per token.
- **M3 started: prefill in blocks of 128 tokens**, first correct version (`generate` 32/32 with the
  prompt in one block, per-layer error like the single step; 828 and 4301 tokens: same generated tokens). New GEMM
  (`gemm.wgsl`, 1.3-1.5 TFLOP/s, 17% of the ceiling), DeltaNet/routing kernels for N tokens, grouped experts. Prompt:
  41 → **17 ms/token** (4301 tokens: 182 → 73 s). Expected ~6: to be profiled per kernel (`profile-prefill`).
- M3: blocks of 512 (+128 for the tails), experts with "per expert" GEMM (tile 64x8 → 128x16 → weights
  in registers `gemm_r`): prompt **42 → 6.9 ms/token** (4301 tokens: 182 → 30 s), still with equal generated tokens.
  Remaining: experts 2.7 ms/token (floor 0.45), dense GEMM 2.2 (1.45 TFLOP/s, 17% of the ceiling).
- M3: benchmark of the experts kernel (`bench-experts`): bandwidth, decoding, latency and x reads ruled
  out as causes; adopted `gemm_r5` (2 rows per thread). Prompt **6.5 ms/token** (4301 tokens: 27.8 s). Experts 2.4 ms/token, dense 2.2.
- M3: dense GEMM 2 (tile 128 x 64, weights in shared memory per row): 1.5 → 2.2 TFLOP/s. Prompt
  **5.8 ms/token** (4301 tokens in 24.8 s, was 182 s at the start of the day), generated tokens still equal.
- `Session` (trait `mechta_api::Engine`), CTX 65536, prefix reuse with
  a snapshot at the last `<|im_start|>`, MTP cache only on the tail of the prompt, prefill one submit per layer (stability),
  prefill attention `attn_fd4` (K/V once per 4 tokens): 30K prompt tokens in 287 s (was 453).
- **Prototype complete** (`PROTOTYPE.md`, instructions in `USAGE.md`): OpenAI-compatible
  server (`mechta serve`, port 4131, systemd unit via `mechta-serve.sh`) used by opencode as a provider. Official chat
  template, thinking and tool calls (Qwen XML → `tool_calls`), fake titles, prefix reuse between turns and ROOT between
  sessions (on disk), prefill blocks of 512/128/32/8 at minimum cost, SSE keep-alive (opencode was closing at 300 s),
  exact sampling with MTP. Stability proven up to 56K of context. Built in ~3 hours.
