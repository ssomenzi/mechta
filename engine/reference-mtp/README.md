# Numerical reference for M4, MTP layer (llama.cpp `42d9581`, CPU only)

Produced by a llama.cpp build with the `mechta-dump-mtp` example (`examples/mechta-dump-mtp/`), run by `dump-mtp.sh` (11 s).
- Same model, prompt (`../reference/prompt.txt`) and parameters as M1 (`-ngl 0 -t 8 -c 1024`): **the same 56 tokens**
  as the M1 dump (24 of prompt + 32 greedy), verified: 0 different tokens.
- The MTP context is created the way llama-server does it (`common_speculative_init_result`, `LLAMA_CONTEXT_TYPE_MTP`), with
  its own KV cache: at position i+1 it receives the true token t_{i+1} and the main model's `h_nextn` at position i.
- `tensors.bin` (407 MB, 13,300 records, format of `mechta-dump.cpp`): for the main model as in M1 (without the
  state tensors `new_state-0`/`conv_state_*`, which have a different name with `load_mtp`) **plus `h_nextn`** (state after
  `output_norm`, 56 positions); for the MTP (prefix `mtp:`, 55 positions 1..55): `mtp_tok_embd-40`, `mtp_hnorm-40`,
  `mtp_enorm-40`, `mtp_concat-40`, `mtp_eh_proj-40`, `mtp_attn_norm-40`, `mtp_Qcur_full-40`, `mtp_Qcur_normed-40`,
  `mtp_gate-40`, `mtp_Kcur_normed-40`, `mtp_Vcur-40`, `mtp_attn_pregate-40`, `mtp_attn_out-40`, `mtp_attn_residual-40`,
  `mtp_attn_post_norm-40`, `mtp_ffn_moe_out-40`, `mtp_ffn_shexp-40`, `mtp_shared_expert_gate_sigmoid-40`,
  `mtp_ffn_shexp_gated-40`, `mtp_ffn_out-40`, `mtp_post_ffn-40`, `mtp:h_nextn`, `mtp_shared_head_norm`,
  `mtp:result_output` (logits of the draft for t_{i+2}).
- `token.tsv`: as in M1 plus `bozza_mtp` (the MTP's argmax for that position) and `bozza_ok`.
- **Greedy acceptance on this prompt: 45 out of 54 (83%); on the 32 generated tokens 32 out of 32 (100%)**; the 9 misses are
  all in the prompt (the user turn's words). Small and easy sample (a trivial function): LM Studio on real logs gave
  71-74%. The MTP graph is in `src/models/qwen35moe.cpp` (`graph_mtp`).
