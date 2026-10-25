# Kernel variants

mechta keeps the older and the alternative versions of its GPU kernels next to the current ones. This file lists
every kernel variant that is not the default, and every environment variable or command that selects one.

- The **default** is what `mechta serve` runs when no variable is set. All the speed numbers in `BENCHMARKS.md`
  are defaults.
- Every other variant is kept on purpose. Each one is a step of the path that led to the current kernels, or an
  alternative that lost a measurement. Keeping them means that step can be read in the code and run again.
- Variables are read once, at start. To change variant, restart the process.
- All the kernels of `ops.wgsl`, and the expert and dense GEMM variants of `gemm.wgsl` that the server can select,
  are compiled at every server start, including the ones kept only for comparisons. `matvec2.wgsl`,
  `matvec3.wgsl`, `wgsl/m1/` and `fa_mx.wgsl` are compiled only when an option selects them.

> **Warning.** The group 3 options (`MECHTA_FD_PHASE`, `MECHTA_FA_PHASE`, `MECHTA_GEMM_PHASE`,
> `MECHTA_ATTN2=read|read2`) give **wrong output on purpose**: they skip or fake a part of a kernel to measure what
> that part costs. Use them only with the benchmark and profile commands. A server started with one of them
> produces garbage text.

## Groups

| group | what it is | where it is documented |
|---|---|---|
| 1 | reachable only from a developer benchmark command (`bench`, `bench-n`, `bench-delta`, `bench-experts`) | here |
| 2 | reachable in the server with an option kept for comparisons with older versions | here |
| 3 | measurement-only option: the output is wrong on purpose | here |
| 4 | alternative selectable in the server, with correct output | `CALIBRATION.md`, `USAGE.md`, here |

File paths below are relative to `engine/crates/mechta/src/wgsl/` unless they say otherwise. Numbers are from
`DEVLOG.md`, `BENCHMARKS.md`, the code comments, or from measurement notes that are not published (these are
marked "unpublished notes"). Where no number was found, the text says "not measured in the published notes".

## Summary

| variant / kernel | file | how to select it | group | status |
|---|---|---|---|---|
| **Single-step attention** | | | | |
| `attn_fd` + `attn_fd_combine` | `ops.wgsl` | default | — | default |
| `attn`, `attn2` | `ops.wgsl` | `MECHTA_ATTN=v1` with `MECHTA_KV=f32` | 2 | comparison |
| `attn_v2`, `attn2_v2` | `ops.wgsl` | `MECHTA_ATTN=v2` with `MECHTA_KV=f32` | 2 | comparison |
| `attn_v3`, `attn2_v3` | `ops.wgsl` | `MECHTA_ATTN=v3` with `MECHTA_KV=f32` | 2 | comparison |
| phase switch of the decode attention | `ops.wgsl` | `MECHTA_FD_PHASE=1\|2` (1-7 with `attn_fd2cm`) | 3 | measurement-only |
| **Double-step (MTP) attention** | | | | |
| `attn_fd2h` | `ops.wgsl` | default | — | default |
| `attn_fd2i` | `ops.wgsl` | `MECHTA_ATTN2=i` | 4 | alternative |
| `attn_fd2c` | `ops.wgsl` | `MECHTA_ATTN2=c` | 4 | alternative |
| `attn_fd2g` | `ops.wgsl` | `MECHTA_ATTN2=g` | 4 | alternative |
| `attn_fd2f` | `ops.wgsl` | `MECHTA_ATTN2=f` | 4 | alternative |
| `attn_fd2e` | `ops.wgsl` | `MECHTA_ATTN2=e` | 4 | alternative |
| `attn_fd2d` | `ops.wgsl` | `MECHTA_ATTN2=d` | 4 | alternative |
| `attn_fd2b` | `ops.wgsl` | `MECHTA_ATTN2=b256` | 4 | alternative |
| `attn_fd2` | `ops.wgsl` | `MECHTA_ATTN2=fd2` | 4 | alternative |
| `attn_fd`, one workgroup per token | `ops.wgsl` | `MECHTA_ATTN2=old` | 4 | alternative |
| `attn_fd2cm` (measuring copy of `attn_fd2c`) | `ops.wgsl` | `MECHTA_ATTN2=cm` | 4 | alternative |
| `attn_fd2_read` | `ops.wgsl` | `MECHTA_ATTN2=read` | 3 | measurement-only |
| `attn_fd2_read2` | `ops.wgsl` | `MECHTA_ATTN2=read2` | 3 | measurement-only |
| **Prefill attention** | | | | |
| `attn_fa_mx` + `q_f16` | `fa_mx.wgsl` | default with the f16 KV cache (`MECHTA_PREFILL_ATTN` unset or `mx`) | — | default |
| `attn_fa` | `ops.wgsl` | `MECHTA_PREFILL_ATTN=fa`; default with `MECHTA_KV=f32` or `q8` | 4 | alternative |
| `attn_fa_mx2` | `fa_mx.wgsl` | `MECHTA_PREFILL_ATTN=mx2` (f16 KV only) | 4 | alternative |
| `attn_fd4` | `ops.wgsl` | `MECHTA_PREFILL_ATTN=fd4`; always used for blocks of 32 and 8 tokens | 4 | alternative |
| `attn_fd4` with longer blocks | `ops.wgsl` | `MECHTA_PREFILL_BLOCK=128\|256\|512` | 4 | alternative |
| `attn_fd`, one workgroup per token | `ops.wgsl` | `MECHTA_ATTN_BATCH=0` | 4 | alternative |
| phase switch of `attn_fa`, `attn_fa_mx`, `attn_fa_mx2` | `ops.wgsl`, `fa_mx.wgsl` | `MECHTA_FA_PHASE=1..4` | 3 | measurement-only |
| **Matvec (single step)** | | | | |
| `mv_q8_0`, `mv_q4_k`, `mv_q6_k`, `mv_q3_k` / `mv_f32` | `matvec4.wgsl` / `matvec.wgsl` | default | — | default |
| `mv_q4_k64` | `matvec4.wgsl` | `MECHTA_Q4K=64` | 4 | alternative |
| everything from `matvec.wgsl` | `matvec.wgsl` | `MECHTA_MATVEC=base` | 2 | comparison |
| Q4_K from `matvec3.wgsl`, rest from `matvec.wgsl` | `matvec3.wgsl` | `MECHTA_MATVEC=q4k3` | 2 | comparison |
| Q8_0 and Q4_K from `matvec4.wgsl`, Q6_K from `matvec.wgsl` | `matvec4.wgsl` | `MECHTA_MATVEC=q8` | 2 | comparison |
| Q8_0 and Q4_K from `matvec2.wgsl` | `matvec2.wgsl` | `MECHTA_MATVEC=mixed` | 2 | comparison |
| M1 matvec and routing | `m1/matvec.wgsl`, `m1/routing.wgsl` | `MECHTA_KERNEL=m1` | 2 | comparison |
| `mv_q8_0b` | `matvec4.wgsl` | `mechta bench <tensor> 5` | 1 | bench-only |
| `mv_read`, `mv_read6` | `matvec4.wgsl` | `mechta bench <tensor> 0` | 1 | bench-only |
| **Multi-column matvec (double step)** | | | | |
| `matvec_nh.wgsl`, `matvec_uh.wgsl` (x in f16) | | default | — | default |
| `matvec_n4.wgsl`, `matvec_u.wgsl` (x in f32) | | `MECHTA_X16=no` | 4 | alternative |
| `matvec_n.wgsl` | | `mechta bench-n <tensor> <n> [reps] a` | 1 | bench-only |
| **Dense GEMM (prefill)** | | | | |
| `gemm2_*` | `gemm.wgsl` | default | — | default |
| `gemm_*` | `gemm.wgsl` | `MECHTA_GEMM_DENSE=old` | 4 | alternative |
| **Expert GEMM (prefill)** | | | | |
| `gemmr5_*` | `gemm.wgsl` | default | — | default |
| `gemmr_*` | `gemm.wgsl` | `MECHTA_GEMM_EXPERTS=r` | 4 | alternative |
| `gemm16_*` | `gemm.wgsl` | `MECHTA_GEMM_EXPERTS=tile16` | 4 | alternative |
| `gemm8_*` | `gemm.wgsl` | `MECHTA_GEMM_EXPERTS=narrow` | 4 | alternative |
| `gemm_*` | `gemm.wgsl` | `MECHTA_GEMM_EXPERTS=wide` | 4 | alternative |
| `gemmr2_*` | `gemm.wgsl` | `mechta bench-experts <tensor> <n> [reps] r2` | 1 | bench-only |
| `gemmr3_*`, `gemmr4_*` | `gemm.wgsl` | `mechta bench-experts <tensor> <n> [reps] r3\|r4` | 1 | bench-only |
| phase switch of the expert GEMM | `gemm.wgsl` | `MECHTA_GEMM_PHASE=1..6` | 3 | measurement-only |
| **DeltaNet** | | | | |
| `delta` | `ops.wgsl` | default | — | default |
| `delta1` | `ops.wgsl` | `MECHTA_DELTA=v1` | 2 | comparison |
| `delta1`, `delta2`, `delta3` | `delta.wgsl` | `mechta bench-delta 1\|2\|3` | 1 | bench-only |
| **Routing and experts of the double step** | | | | |
| experts in union (`merge` + `matvec_uh`) | `merge.wgsl`, `matvec_uh.wgsl` | default | — | default |
| one expert pass per token | `matvec4.wgsl` | `MECHTA_EXPERTS=token` with `MECHTA_X16=no` | 2 | comparison |
| **MTP draft** | | | | |
| MoE of the draft only for the second token | — | default | — | default |
| MoE of the draft for both tokens | — | `MECHTA_MTP2=full` | 2 | comparison |
| **Flint model** | | | | |
| `attn_sink` (generation attention of Flint) | `flint_attn.wgsl` | `MECHTA_FLINT_ATTN=sink` | 2 | comparison |
| `attn_pf` with packed f16 in shared memory | `flint_attn.wgsl` | `MECHTA_FLINT_PF=f16` | 2 | comparison |

Other switches that change how kernels run, without being kernel variants, are in [Miscellaneous](#miscellaneous).

## Trying a variant

All commands are run from the repository root, after `cargo build --release` in `engine/`, with the three
required variables of `USAGE.md`:

```
export MECHTA_MODEL=/path/to/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf
export MECHTA_TOKENIZER=tokenizers/qwen3.6-tokenizer.json
export MECHTA_CACHE=/path/to/prefix-cache
```

Server with a variant (the startup log names the `MECHTA_MATVEC` and `MECHTA_DELTA` choices, and the prefill
attention when it runs on the matrix units):

```
MECHTA_ATTN2=c engine/target/release/mechta serve --ctx 65536
```

Numerical check of a variant against the default (from `USAGE.md`; `double` runs the double step of MTP
generation, so a change of the generation kernels is covered):

```
engine/target/release/mechta logit-dump text.txt 16384 256 ref.bin double
MECHTA_ATTN2=c engine/target/release/mechta logit-dump text.txt 16384 256 var.bin double
engine/target/release/mechta kl ref.bin var.bin 'median<=0.001' 'top1>=85'
```

The limits used to accept a variant are `median<=0.001 top1>=85` on code and `mean<=0.01` on English prose.

Speed tools used for the numbers below:

| command | what it measures |
|---|---|
| `mechta bench-attn <ctx,...>` | GPU time per kernel of the double step at the given contexts (fake positions, cache of zeros: times valid). Set `MECHTA_CTX` above the largest context |
| `mechta profile-prefill-at <text> <n> <pos,...>` | prefill of the first `<n>` tokens of a text at each position (cache of zeros), time per kernel |
| `mechta bench-step <ctx>` | every kernel of the double step and of the MTP draft, with bytes read and GB/s |
| `mechta bench-kv <ctx,...>` | single steps and MTP cycles at the given contexts, with the KV format of `MECHTA_KV` |

The four benchmark commands of group 1, with the syntax of `main.rs`:

```
mechta bench <tensor> <variant> [repetitions=50]                  # variant 1 2 3 4 5 6 0
mechta bench-n <tensor> <n> [repetitions=50] [a|b|c|g|g2]         # default a; n 1..4 (a, b, c), up to 512 (g, g2)
mechta bench-delta <1|2|3> [repetitions=50]                       # no model needed
mechta bench-experts <tensor> <n> [repetitions=20] [r|r2|r3|r4|r5|16|8|64]   # default r; n multiple of 8, at most 512
```

Each prints the median GPU time, the GB/s (and GFLOP/s for `bench-n` and `bench-experts`) and the error against
an f64 computation on the CPU. Tensor names are GGUF names, for example `blk.0.attn_qkv.weight`,
`blk.0.ffn_gate_exps.weight`, `blk.0.ffn_down_exps.weight`, `output.weight`. For `bench` on an expert tensor,
`MECHTA_BENCH_EXPERTS=N` (1 to 256) sets how many experts one dispatch reads (default 8, as in the engine).

## Decode attention, single step

The attention layers (10 of 40) read the whole KV cache at every generated token. The single step (one token)
uses `attn_fd` followed by `attn_fd_combine`.

**`attn_fd` (default).** Flash-decoding with GQA: one workgroup per (KV head, block of 128 positions, token) reads
K and V of its block once and computes the 8 query heads that share that KV head; `attn_fd_combine` merges the
blocks. History in `DEVLOG.md`, Day 5: at a context of ~900 the attention went from 2.7 ms (`attn_v3`) to 2.1 ms,
and the context grew from 1024 to 8192. The same day the V phase was read as vec4 with 4 lanes (2.06 → 1.09 ms at
~900) and the K phase got 2 threads per position (0.86 ms at ~900, 3.3 ms at 4300).

**`attn` / `attn2` (`MECHTA_ATTN=v1`).** The first attention: one workgroup per query head, a sequential loop over
the positions with a barrier per step, scores kept in a workgroup array of 1024 entries. It is only valid up to
1024 positions. `DEVLOG.md`, Day 4: a single step took 40.9 ms at ~30 tokens of context and 48.8 at ~900.
Unpublished notes: at ~900 this kernel alone took 7.84 ms per token. Not the default because it is sequential over
the positions.

**`attn_v2` / `attn2_v2` (`MECHTA_ATTN=v2`).** Parallel over the positions: one thread per position computes the
score, two reductions, then one thread per dimension accumulates V. Unpublished notes: 7.84 → 6.68 ms at ~900
(-15%). Kept as the step between v1 and v3.

**`attn_v3` / `attn2_v3` (`MECHTA_ATTN=v3`).** Like v2, with each K row read as vec4 (64 reads instead of 256)
and q in vec4. `DEVLOG.md`, Day 5: at ~900 the attention went from 7.8 to 2.7 ms and the token from 47.9 to 43.0 ms.
It still reads K and V once per query head (8 times per KV head), which `attn_fd` removed.

The three `MECHTA_ATTN` values read the cache as f32: the engine stops at start unless `MECHTA_KV=f32`. They change
both the single step and the double step. The prefill attention is not affected.

```
MECHTA_KV=f32 MECHTA_ATTN=v3 engine/target/release/mechta serve --ctx 65536
```

**`MECHTA_FD_PHASE` (group 3, wrong output).** Written into the uniform of every decode attention dispatch.
`attn_fd`, `attn_fd2` and the `attn_fd2b`..`attn_fd2i` kernels read it: 1 skips the V phase, 2 skips the K phase.
`attn_fd2cm` (below) has 7 switches. `DEVLOG.md`, Day 5: this showed that the V phase, not K, was the heavy one.
Unpublished notes, at a fake context of 4300: whole kernel 5.17 ms, K phase alone 2.04, V phase alone 2.90.

```
MECHTA_CTX=131072 MECHTA_FD_PHASE=1 engine/target/release/mechta bench-attn 45056
```

## Double-step (MTP) attention

With MTP on, each cycle runs a double step: the next token and the draft go through the model together. The double
step's attention is chosen with `MECHTA_ATTN2`. All variants except `old` launch one workgroup per (KV head, block)
for both tokens, so K and V are read once for the two tokens, and write the same partials, merged by
`attn_fd_combine`. They work with every KV format.

Unpublished notes, `mechta bench-attn` (median of 8 double steps, f16 KV), time of the attention kernel alone:

| variant | 12K | 45K | 64K | 124K |
|---|---|---|---|---|
| `fd2` (`attn_fd2`) | 6.56 ms | 22.72 ms | | |
| `b256` (`attn_fd2b`) | 5.27 ms | 18.79 ms | | |
| `c` (`attn_fd2c`) | 4.75 ms | 16.80-17.3 ms | 24.9 ms | 47.4 ms |
| `d` (`attn_fd2d`) | 6.94 ms | 25.01 ms | | |
| `e` (`attn_fd2e`) | 6.12 ms | 21.31 ms | | |
| `f` (`attn_fd2f`) | | 27.5 ms | 39.0 ms | 73.4 ms |
| `g` (`attn_fd2g`) | | 19.2 ms | 28.1 ms | 53.9 ms |
| `h` (`attn_fd2h`, default) | | 14.1 ms | 20.4 ms | 38.9 ms |
| `i` (`attn_fd2i`) | | 14.0-14.7 ms | 20.3 ms | 38.4 ms |

The rows come from several sessions of the same tool; the `c` row is the reference repeated in each session.
`f`, `g`, `h` and `i` were run alternated with `c` (3 rounds, spread under 1%).

**`attn_fd2h` (default).** `attn_fd2c` with 4 rows of V requested at once in phase 2. Code comment and `USAGE.md`:
-18% against `attn_fd2c` from 45K to 124K, numeric check PASS. It was written after the piece-by-piece measurement
of `attn_fd2cm` showed that the V read was the most expensive piece.

**`old`: `attn_fd` per token.** The double step runs `attn_fd` with one workgroup per token (grid z = 2), so every
block of the cache is read twice per cycle. Unpublished notes, real server, f16 KV, MTP on: `attn_fd2` raised
generation from 26.7 to 28.1 t/s at 12.5K, 22.9 to 25.0 at 28K, 20.0 to 21.7 at 44K.

**`fd2`: `attn_fd2`.** The first kernel that reads K and V once for both tokens: a workgroup (KV head, block of 128
positions) holds the q of 2 tokens x 8 heads (16 rows). Numbers in the table.

**`b256`: `attn_fd2b`.** `attn_fd2` with blocks of 256 positions (half the q loads, partials and merge work per
position) and max / sum computed by all 256 threads. At 45K: 22.72 → 18.79 ms.

**`c`: `attn_fd2c`.** `attn_fd2b` with less on-chip traffic: a thread does 2 positions in phase 1, so each q vec4
read from shared memory serves two K rows, and scores are stored per position so phase 2 reads the 16 weights of a
position with 4 vec4 reads. It was the default before `attn_fd2h`. Unpublished notes: logits within 0.008 of
`attn_fd2`, 8/8 argmax; real server with MTP, against `attn_fd2`: 12.7K 26.5 → 28.75 t/s, 28K 23.6 → 25.6,
44K 21.8 → 23.6.

**`d`: `attn_fd2d`** (phase 1 on 4 positions per thread, quarter rows) and **`e`: `attn_fd2e`** (phase 2 on half
the rows per thread, fewer registers). Both slower than `c` (table); unpublished notes say "more registers".

**`f`: `attn_fd2f`.** `attn_fd2c` with K read by consecutive lanes on consecutive addresses, through shared memory
in chunks of 16 positions. Numeric check PASS, but 55-59% slower than `c`: the new way of computing phase 1 costs
more than the read saves.

**`g`: `attn_fd2g`.** `attn_fd2c` with the K read of phase 1 spread over 4 interleaved lanes per position, without
shared memory. Numeric check PASS, 12-15% slower than `c`.

**`i`: `attn_fd2i`.** `attn_fd2h` plus 4 vec4 of each K row requested before use. Numeric check PASS. The
difference from `attn_fd2h` is under 1% and inside the spread between rounds: no gain, so `h` stays the default.

**`cm`: `attn_fd2cm`.** A copy of `attn_fd2c` with switches in `MECHTA_FD_PHASE`: 1 = no V phase, 2 = no K phase,
3 = K phase with fake K, 4 = K read without the dot products, 5 = no softmax, 6 = V phase with fake V, 7 = V read
without the weighted sum. With 0 it computes the same as `attn_fd2c` (about 10% slower, unpublished notes). Unpublished
notes, savings when a piece is removed (base 19.2 ms at 45K, 52.5 ms at 124K): the V read alone 6.3 / 18.2 ms, the
K read alone 3.6 / 11.1 ms, the softmax 0.3 / 1.1 ms.

**`read` and `read2` (group 3, wrong output).** The grid of `attn_fd2c` reading the same K and V bytes with almost
no math. `read`: K and V both read with consecutive lanes on consecutive addresses. `read2`: K read as in phase 1 of
`attn_fd2c`. Unpublished notes: `read` 11.8 ms at 45K and 33.0 ms at 124K (78-79 GB/s), `read2` 18.2 and 40.9 ms,
against 16.9 and 46.9 ms for the real `attn_fd2c`. The grid can reach the memory ceiling; the K access pattern of
phase 1 alone costs more than all of `attn_fd2c` at 45K.

Any other value of `MECHTA_ATTN2` selects the default.

```
MECHTA_CTX=131072 MECHTA_ATTN2=g engine/target/release/mechta bench-attn 45056,65536,126976
MECHTA_ATTN2=i engine/target/release/mechta serve --ctx 131072
```

## Prefill attention

The prefill runs the prompt in blocks of 512, 128, 32 and 8 tokens (`BLOCKS` in `engine.rs`). Blocks of 64 tokens
or more use the kernel chosen with `MECHTA_PREFILL_ATTN`. Blocks of 32 and 8 tokens always use `attn_fd4` (or
`attn_fd` with `MECHTA_ATTN_BATCH=0`), so `MECHTA_PREFILL_BLOCK` and `MECHTA_ATTN_BATCH` also act on those small
blocks with the default settings.

**`mx`: `attn_fa_mx` (default with the f16 KV cache).** `attn_fa` with S = Q K^T computed on the matrix units of
RDNA 3 (cooperative matrix, 16x16x16, f16 in, f32 sum), reading Q (a f16 copy written by `q_f16`) and K straight
from memory. It needs the f16 KV cache and an experimental `wgpu` feature, requested only when this kernel is in use.
`BENCHMARKS.md`: prefill 6% faster at 12.6K and 45% faster at 93K than `attn_fa` on the real server, same generation
speed. Unpublished notes, kernel time at 100K: `attn_fa` 8.27-8.32, `attn_fa_mx` 4.66-4.68 ms per token.

**`fa`: `attn_fa`.** Flash attention without matrix units: a workgroup = (KV head, chunk of 2048 positions, group of
8 tokens), 64 rows, tiles of 64 keys, online softmax; Q, K and V in packed f16 in shared memory, products and sums
in f32. Default when `MECHTA_KV=f32` or `q8`. Unpublished notes, 1024 tokens of prefill at a given position against
`attn_fd4` (first version of `attn_fa`): 12K 129 → 160 t/s, 44K 62 → 88, 100K 33 → 50. After the later changes
(Q, K, V as f16 in shared memory, vector reads of the softmax weights) 76 t/s at 100K. On the real server against `attn_fd4` (which was
measured with the GPU clock on `auto`): 146 → 210 t/s at 12.6K, 31 → 73 t/s at 114K.

**`mx2`: `attn_fa_mx2`.** `attn_fa_mx` with O += P V on the matrix units too: P goes to shared memory as f16, V is
read straight from the f16 cache, and the softmax uses a lazy reference max (a row's reference moves only when its
max grows by more than a threshold) because the matrix units cannot scale rows. Unpublished notes: numeric check PASS
against `attn_fa`; kernel time at 100K 4.60-4.61 ms per token against 4.66-4.68 for `attn_fa_mx` (-1.3%). The gain is
too small to make it the default; why its QK part costs more than in `attn_fa_mx` was not found.

**`fd4`: `attn_fd4`.** The previous prefill attention: a workgroup = (KV head, block of 64 positions), two passes of
4 tokens of a batch of 8, so K and V are read once per 4 tokens. `DEVLOG.md`, Day 5: a 30K prompt in 287 s instead of
453 s. Unpublished notes: at 100K, `attn_fd4` 20.6 ms per token plus 4.3 ms of merge, the reason for `attn_fa`.

**`MECHTA_PREFILL_BLOCK=128|256|512`.** Positions per workgroup of `attn_fd4` (default 64), in tiles of 64 with a
running max and sum. Code comment in `engine.rs`: at 100K, 256 cuts the merge from 4.3 to 1.2 ms per token, but
`attn_fd4` grows by 3.9: no gain. Unpublished notes: KL 0.00008 against 64 and the same t/s at 12K, 44K and 100K.
Any other value stops the engine.

**`MECHTA_ATTN_BATCH=0`.** The prefill runs `attn_fd` with one workgroup per token instead of `attn_fd4`. Only 0 is
special: any other number selects `attn_fd4` (default 8). This is the "453 s" side of the `DEVLOG.md` Day 5 number above.

**`MECHTA_FA_PHASE` (group 3, wrong output).** Skips one phase of the flash attention kernels: for `attn_fa`
1 = no QK, 2 = no softmax, 3 = no PV, 4 = no Q/K loads; for `attn_fa_mx` and `attn_fa_mx2` 1 to 3. Unpublished notes:
in `attn_fa` QK was 69% of the kernel, PV 25%, softmax 6%. At 100K, `attn_fa_mx` 4.66 ms whole, 3.20 without QK,
2.97 without PV; `attn_fa_mx2` 4.61, 2.10, 3.09.

```
MECHTA_PREFILL_ATTN=fa engine/target/release/mechta serve --ctx 131072
MECHTA_CTX=131072 MECHTA_PREFILL_ATTN=mx2 engine/target/release/mechta profile-prefill-at text.txt 1024 12288,45056,100352
MECHTA_PREFILL_ATTN=fd4 MECHTA_PREFILL_BLOCK=256 engine/target/release/mechta logit-dump text.txt 16384 16 var.bin
```

## Matvec, single step

One generated token reads about 2.59 GB of weights with matrix-vector products. The default takes Q8_0, Q4_K,
Q6_K and Q3_K from `matvec4.wgsl` and F32 from `matvec.wgsl`. The history of these kernels is milestone M2 in
`DEVLOG.md` (Day 3 and Day 4).

**`matvec.wgsl` (`MECHTA_MATVEC=base`).** One thread per piece of 32 values, reduction in shared memory, bytes read
one at a time. `DEVLOG.md`, Day 3: this kernel, replacing the M1 one (one thread per block), together with the
routing on 256 threads took the token from 241 to 160-166 ms with the GPU clock fixed (experts were still copied
from the file at every token then).

**`matvec3.wgsl` (`MECHTA_MATVEC=q4k3`).** Q4_K with x read as vec4, scales as words, and
`d1 (q . x) - m1 (sum x)`. `DEVLOG.md`, Day 4: experts 46 → 62 GB/s, -1.8 ms per token (53.1 → 51.2-51.6).

**`matvec4.wgsl` for Q8_0 and Q4_K only (`MECHTA_MATVEC=q8`).** Q8_0 read as 9 realigned words per piece and
unpacked with `unpack4xI8`. `DEVLOG.md`, Day 4: dense matrices 60-66 → 77-82 GB/s, -3.3 ms (48.3 → 44.9). With
`q8` the Q6_K stays on `matvec.wgsl`; the default also moved Q6_K to `matvec4.wgsl`: output head 7.5 → 5.0 ms,
-2.5 ms (44.9 → 42.5).

**`matvec2.wgsl` (`MECHTA_MATVEC=mixed`).** Bytes read as 4-byte words, without vec4. Unpublished notes: on the
bench -18% on `attn_qkv` and -12% on the Q4_K expert down matrix, but +9% on the output head and +59% on the Q6_K
expert down matrix; in the engine it was not faster (three rounds: 53.1-56.2 ms base, 54.1-56.1 ms mixed).

**M1 kernels (`MECHTA_KERNEL=m1`).** `m1/matvec.wgsl` (one thread per block: with Q4_K, 8 threads of 64 work) and
`m1/routing.wgsl` (routing on a single thread). `DEVLOG.md`, Day 3: K-quant kernel 91 ms and routing 18 ms per token
before they were replaced. Note: the current engine also asks this file for a `routing_b` entry point (the second
token of the double step), which `m1/routing.wgsl` does not have, so pipeline creation is expected to fail with
this option. Not run.

**`mv_q4_k64` (`MECHTA_Q4K=64`).** Q4_K in pieces of 64 values: one thread reads the 32 bytes of qs and the scales
once for the low and high nibbles, where the default uses two threads that read them twice. Unpublished notes: on the
bench -11% on the gate matrix with 256 experts per dispatch, but in the engine (8 experts per dispatch) no change
(±0.2 ms, inside the noise), so it stays opt-in. It applies to the single-step matvec only; the double step's
expert kernels are not affected.

**`mv_q8_0b` (bench variant 5)** is Q8_0 with the int8 unpacked by hand (shifts and masks) instead of `unpack4xI8`:
same speed on the bench, so the builtin was kept. **`mv_read` and `mv_read6` (bench variant 0)** only read the bytes
of a Q8_0 or Q6_K tensor with the same geometry: they measure the read ceiling, and the CPU check fails on purpose.
`DEVLOG.md`, Day 4: the measured ceiling is 81-83 GB/s.

The `bench` variants: 1 = `matvec.wgsl`, 2 = `matvec2.wgsl`, 3 = `matvec3.wgsl`, 4 = `matvec4.wgsl`, 5 = `mv_q8_0b`
(Q8_0 only), 6 = `mv_q4_k64` (Q4_K only), 0 = read only (Q8_0 or Q6_K).

```
engine/target/release/mechta bench blk.0.attn_qkv.weight 4
engine/target/release/mechta bench output.weight 0
MECHTA_BENCH_EXPERTS=256 engine/target/release/mechta bench blk.0.ffn_gate_exps.weight 6
MECHTA_MTP=no MECHTA_MATVEC=q4k3 engine/target/release/mechta serve --ctx 65536
```

`matvec.wgsl`, `matvec2.wgsl`, `matvec3.wgsl` and `m1/matvec.wgsl` have no `x_off` / `y_off` in their uniform. The MTP
draft (MoE for the second token only, and the draft head) reads its inputs at an offset that only `matvec4.wgsl`
supports. Compare `MECHTA_MATVEC` and `MECHTA_KERNEL=m1` with `MECHTA_MTP=no`, as these kernels were measured before
MTP existed.

### Multi-column matvec, double step

The double step multiplies each dense matrix by 2 columns (the two tokens), reading the weights once.

**`matvec_nh.wgsl` and `matvec_uh.wgsl` (default).** x in f16 (4 values in one `vec2<u32>`), vec4 accumulators with
one component per column. `DEVLOG.md`, Day 4: double step 56.4 → 50.9 ms, MTP 34.1-34.9 t/s.

**`matvec_n4.wgsl` and `matvec_u.wgsl` (`MECHTA_X16=no`).** The same with x in f32, and the producers of x
(`rmsnorm`, `swiglu`, `attn_gate`) writing f32. With 2 columns of k = 4096, x takes 32 KB and falls out of the cache;
that is what f16 fixed. `DEVLOG.md`, Day 4: n = 2 almost free, n ≥ 3 pays for the x cache.

**`matvec_n.wgsl` (`bench-n` variant `a`).** The first n-column kernel: an array of accumulators per column and a
branch per column inside the loops. Unpublished notes: already at n = 1 it cost 13-50% more than the one-column
kernel; the vec4 accumulators of `matvec_n4.wgsl` (variant `b`) replaced it.

`bench-n` variants: `a` = `matvec_n.wgsl`, `b` = `matvec_n4.wgsl`, `c` = `matvec_nh.wgsl`, `g` = `gemm_*`, `g2` = `gemm2_*`.

```
engine/target/release/mechta bench-n blk.0.attn_qkv.weight 2 50 c
MECHTA_X16=no engine/target/release/mechta serve --ctx 65536
```

## Dense GEMM, prefill

**`gemm2_*` (default).** Tile of 128 rows x 64 tokens, each thread 8 rows x 4 tokens, weights stored in shared memory
by row as whole vec4. `DEVLOG.md`, Day 5: 1.5 → 2.2 TFLOP/s, prompt 6.5 → 5.8 ms per token.

**`gemm_*` (`MECHTA_GEMM_DENSE=old`).** The first GEMM: tile of 64 x 64, weights decoded into shared memory
transposed, one scalar at a time. `DEVLOG.md`, Day 5: 1.3-1.5 TFLOP/s, 17% of the ceiling. Unpublished notes,
`bench-n` at 512 tokens: `attn_qkv` 11.5 ms (`gemm`) against 7.6 ms (`gemm2`); `ssm_out` (k = 4096) 31.5 against 3.9 ms.

```
engine/target/release/mechta bench-n blk.0.ssm_out.weight 512 10 g
MECHTA_GEMM_DENSE=old engine/target/release/mechta serve --ctx 65536
```

## Expert GEMM, prefill

In a prefill block each of the 256 experts multiplies only the tokens routed to it (about 16 per expert in a block
of 512). The variants differ in the tile and in where the weights live. History in `DEVLOG.md`, Day 5 ("tile 64x8 →
128x16 → weights in registers `gemm_r`", then `gemm_r5`).

Unpublished notes, time of the three expert GEMMs on one block of 512 tokens: `narrow` 1,872 ms, `tile16` 1,764,
`r` 1,399, `r5` 1,206. `wide` was measured on blocks of 128 only: about 500 ms per GEMM against 194-222 for `narrow`.

**`gemmr5_*` (default).** Weights decoded into registers, 2 rows per thread, x read once for both rows, accumulators
written out explicitly. `DEVLOG.md`, Day 5: adopted, prompt 6.5 ms per token. It is the only expert GEMM with a Q3_K
version: a model with Q3_K experts stops at start with any other `MECHTA_GEMM_EXPERTS`.

**`gemmr_*` (`MECHTA_GEMM_EXPERTS=r`).** Weights in registers, one row per thread (64 rows x 4 lanes of k), the 16
tokens of x in shared memory. `DEVLOG.md`, Day 5: prompt 6.9 ms per token.

**`gemm16_*` (`tile16`).** Tile of 128 rows x 16 tokens, weights through shared memory. Unpublished notes: -6% on
`narrow`; the weight phase got slower (16 scalar writes with branches per thread), the compute faster.

**`gemm8_*` (`narrow`).** Tile of 64 rows x 8 tokens, written when the 64 x 64 tile was computing 64 columns to use 4.
Unpublished notes: on blocks of 128 the prompt went from 16.3 to 8.6 ms per token.

**`gemm_*` (`wide`).** The dense 64 x 64 GEMM used for the experts: most of the tile is wasted with few tokens per
expert. Unpublished notes: the three expert GEMMs were 80% of the prefill.

**Bench only: `gemmr2_*`, `gemmr3_*`, `gemmr4_*`.** `r2` gives each lane 2 pieces per step (twice the bytes in
flight, half the barriers); `r3` / `r4` give each thread 2 / 4 rows but reread x per row. Unpublished notes,
`bench-experts` at 512 tokens, gate / down of block 0 (Q4_K): `r` 10.0 / 13.4 ms, `r2` 10.5 / 14.9 (so latency is not
the limit), `r3` 9.9 / 12.0, `r4` 9.9 / 11.1, `r5` 8.7 / 11.4. `bench-experts` variants: `r`, `r2`, `r3`, `r4`, `r5`,
`16` (`gemm16`), `8` (`gemm8`), `64` (`gemm`).

**`MECHTA_GEMM_PHASE` (group 3, wrong output).** Passed to the expert GEMM, in the server and in `bench-experts`.
`gemm8`, `gemm16` and `gemmr` read 1 = only the weight load and decoding, 2 = only the compute loop. `gemmr3` /
`gemmr4` read 3 = Q4_K without the scale, 4 = raw words without decoding, 5 = no reads of x, 6 = no reads of weights.
`gemmr5` and `gemm` ignore it. Unpublished notes: `gemm8` on a block of 512, whole 1,872 ms, weights only 645 ms,
compute only 1,400 ms; with `r4`, modes 3-6 changed the time little (9.0-9.6 ms on the gate), which ruled out
decoding and memory reads as the cause.

```
engine/target/release/mechta bench-experts blk.0.ffn_gate_exps.weight 512 20 r5
MECHTA_GEMM_PHASE=4 engine/target/release/mechta bench-experts blk.0.ffn_gate_exps.weight 512 20 r4
MECHTA_GEMM_EXPERTS=narrow engine/target/release/mechta serve --ctx 65536
```

## DeltaNet

30 of the 40 layers are Gated DeltaNet layers with a 128 x 128 f32 state per head (2 MiB per layer), read and
written at every token.

**`delta` (default).** State stored as `o[h][i][j]` (the 128 threads read 512 contiguous bytes per step), column kept
in registers between the two passes: one read and one write per value. `DEVLOG.md`, Day 4: 4.1 → 1.24 ms per token,
-2.5 ms (51.2 → 48.3).

**`delta1` (`MECHTA_DELTA=v1`).** The original layout, as in ggml (`o[h][j][i]`), two passes over the state in memory.
It replaces the single-step `delta` only.

**`delta.wgsl` (`bench-delta`).** Three versions of the same step on a random state: 1 = `delta1`, 2 = transposed
state with two passes, 3 = transposed plus registers (what `delta` is). Unpublished notes: 0.137-0.155, 0.101-0.103
and 0.041 ms per dispatch (27-31, 41 and 102 GB/s on 2 x 2 MiB).

```
engine/target/release/mechta bench-delta 2
MECHTA_MTP=no MECHTA_DELTA=v1 engine/target/release/mechta serve --ctx 65536
```

## Routing and experts of the double step

**Experts in union (default).** In the double step the two tokens choose 8 experts each. `merge.wgsl` builds the
union, and `matvec_uh.wgsl` (or `matvec_u.wgsl` with `MECHTA_X16=no`) reads each expert of the union once for the
tokens that use it. `DEVLOG.md`, Day 4: double step 58.9 → 56.4 ms.

**`MECHTA_EXPERTS=token`.** One expert pass per token (`matvec4.wgsl` with offsets), 16 matrices per layer. The engine
requires `MECHTA_X16=no` with it.

The routing itself (`routing.wgsl`, 256 threads) has no alternative other than the M1 single-thread kernel of
`MECHTA_KERNEL=m1` (see Matvec).

```
MECHTA_X16=no MECHTA_EXPERTS=token engine/target/release/mechta serve --ctx 65536
```

## MTP draft

**MoE only for the second token (default).** The draft of two tokens needs only K and V of the first token, so the
MoE runs for the second token only. `DEVLOG.md`, Day 4: correct after a fix, 31.7-32.7 ms per token, 16/16 drafts.

**`MECHTA_MTP2=full`.** MoE for both tokens of the draft, the earlier version. Unpublished notes: the cut gives the
same drafts and is within the noise (±0.5 ms) of the full version; it stays the default because it does less work.
`mechta mtp2` checks the double draft against the reference dump.

## Miscellaneous

### KV cache format
`MECHTA_KV=f16` (default), `f32` or `q8` (documented in `USAGE.md`) changes the attention kernels' access block
(`kv.wgsl` replaces the f32 block of `ops.wgsl`) and the kernels that write the cache (`kv_rope`, `kv_rope2`,
`kv_copy`). `f16sim` is for tests only: f32 storage with every value rounded to f16 when read. Unpublished notes:
`f16sim` gives exactly the f16 results, which shows the f16 code is right and the difference is the rounding.

### Flint
For a Flint GGUF the engine uses `flint_attn.wgsl`.
- `MECHTA_FLINT_ATTN=sink`: the first generation attention of Flint, one workgroup per query head, K and V read 8
  times; f32 cache only. Default: a flash-decoding `attn_fd`. Unpublished notes: attention at 30K 157.4 → 40.9 ms per
  token, generation 5.2 → 13.1 t/s.
- `MECHTA_FLINT_PF=f16`: the prefill attention `attn_pf` with Q, K and V in packed f16 in shared memory, as `attn_fa`
  of Qwen. Default: f32 in shared memory. Code comment: with packed f16 the K/V cache drifts to an error of ~1 by
  layer 46, with f32 to 4e-4.
- `MECHTA_FLINT_KV=f16` (documented in `USAGE.md`) selects the f16 cache.

### Switches that are not kernel variants

| variable | effect | status |
|---|---|---|
| `MECHTA_ATTN_PIECES=N` | attention dispatches per submit in the prefill (default 2), for desktop responsiveness. Unpublished notes: longest submit 373 → 98 ms for +1.8% of prefill time | documented (`USAGE.md`, `CALIBRATION.md`) |
| `MECHTA_DRAFT_HEAD=no` | MTP drafts with the full output head instead of the reduced one | documented (`USAGE.md`) |
| `MECHTA_POLL=sleeping` | sleep in the poll after a submit instead of busy waiting. `DEVLOG.md`, Day 4: busy waiting saves 0.6 ms per token | comparison |
| `MECHTA_STEP=old` | `mechta profile` only: token, logits read back (1 MB, second submit) and argmax on the CPU. `DEVLOG.md`, Day 4: the GPU argmax saves 0.8 ms | comparison |
| `MECHTA_FAKE_POS=N` | `mechta profile` only: measure at context N without a prefill (cache of zeros: times valid, output meaningless) | measurement-only |
| `MECHTA_PREFILL=blocks` | `mechta generate` and `mechta profile` only: prompt in blocks instead of one token at a time (the server always uses blocks) | developer check |
| `MECHTA_BENCH_EXPERTS=N` | `mechta bench` only: experts per dispatch | bench-only |
| `MECHTA_ORDER=weights-first` | loads the weights before the buffers the GPU writes. **Do not use**: `DEVLOG.md`, Day 4, it corrupts the page cache and kernel memory and gives wrong tokens | test only |

MTP acceptance and measurement switches (`MECHTA_MTP_ACCEPT`, `MECHTA_MTP_CHAIN`, `MECHTA_REJECT`,
`MECHTA_GEN_PROFILE`) do not select kernels; `USAGE.md` documents the first one.
