# Per-machine calibration (plan)

The speed settings of mechta were measured by hand on one machine (Radeon 780M, DDR5) and put as defaults in the
code. On another machine (760M, other RAM, other driver) the optimum may be different. A calibration step would
choose them on the real machine.

## What is fixed and matters for speed

### A. Kernel variants
This branch builds only the default kernels. The alternatives (double-step attention, expert and dense GEMM, prefill
attention, matvec layouts) and the commands that measure them (`block-cost`, `profile-prefill-at`, `bench-*`) are kept
in the `lab` branch, described in its `engine/VARIANTS.md`: a calibration would be built there and its winners
brought back here. One switch of this kind stays in this branch:

| variable | variants | default | where it weighs |
|---|---|---|---|
| `MECHTA_DRAFT_HEAD` | `draft-head.tsv`, `no` | reduced | MTP drafts |

### B. Numbers written in the code (changed by recompiling)
| constant | value | where |
|---|---|---|
| `BLOCKS` / `COST` | sizes 512/128/32/8 with costs 3500/1100/840/390 ms measured at ~14K of context | choice of the prefill blocks (`engine.rs`) |
| `FA_CH` / `FA_SPLIT` | 2048 positions / 8 pieces per dispatch | prefill flash attention; `FA_SPLIT` keeps the submits under ~56 ms |
| `MECHTA_ATTN_PIECES` | 2 | splitting of the prefill submits (desktop responsiveness versus speed) |
| workgroup geometry | in the WGSL files | all the kernels |

## Proposed plan (not implemented)
1. A calibration command (in the `lab` branch, where the variants are): for every kernel family it measures the
   variants with short, repeated runs (3 runs, alternated): prefill of N tokens at a fixed context, and double-step
   cycles at 2-3 contexts. It keeps the fastest.
2. Every variant that changes the numbers goes through the numerical check (`mechta logit-dump` + `mechta kl`, see
   `USAGE.md`) before it can win.
3. The costs in `COST` (table B) are measured again on the machine and replace the ones written in the code.
4. The result goes into a file (for example `~/.config/mechta/calibration.json`) that the engine reads at startup; an
   environment variable still wins over the file. Without the file: the current defaults.
5. The constants that affect responsiveness (`FA_SPLIT`, `MECHTA_ATTN_PIECES`) are not optimized for speed alone:
   they stay out of the automatic calibration.

The workgroup geometry items (table B, last row) require kernels compiled with different constants: out of
the first round.
