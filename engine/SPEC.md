# mechta — specification of one token (decode), Qwen3.6-35B-A3B UD-Q4_K_S

Source: llama.cpp graph `42d9581` (`src/models/qwen35moe.cpp`, `delta-net-base.cpp`, `models.h`).
llama.cpp is the **numerical reference** for M1: mechta must produce the same tokens.
Formats from the GGUF header.

## Constants (from the GGUF)
hidden 2048 · 40 text layers (+ block 40 = MTP, ignored in M1) · full attention if
`(il + 1) % 4 == 0` · RMSNorm eps 1e-6 · vocabulary 248,320 · 256 experts, 8 used, intermediate 512.
Attention: 16 Q heads, 2 KV heads, head dim 256, RoPE on 64 dimensions, base 1e7, sections [11,11,10,0].
DeltaNet: 16 K/Q heads of 128, 32 V heads of 128, conv kernel 4 over 8192 channels, state 32 × 128×128.

## Formats to decode
| tensors | type |
|---|---|
| `token_embd`, all dense projections (`attn_qkv`, `attn_gate`, `ssm_out`, `attn_q/k/v/output`), shared experts | **Q8_0** |
| experts `gate`/`up`, `down` (38 layers) | **Q4_K** |
| `down` of 3 layers, `output` | **Q6_K** |
| router `ffn_gate_inp`, `ffn_gate_inp_shexp`, `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, norms, `ssm_a`, `ssm_dt` | **F32** |

## One token
```
x = embd[token]                                   # Q8_0 row -> f32 [2048]
for each layer il = 0..39:
  h = rmsnorm(x) * attn_norm
  if (il+1) % 4 != 0:  a = deltanet(h, il)        # 30 layers
  else:                a = attention(h, il)       # 10 layers
  x = x + a
  h = rmsnorm(x) * post_attention_norm
  x = x + moe(h, il)
x = rmsnorm(x) * output_norm
logits = output @ x                               # Q6_K [248320 x 2048]
token = argmax(logits)                            # M1: pick the most probable
```

### deltanet(h, il) — autoregressive step
```
qkv = attn_qkv @ h          [8192] = q[16x128] | k[16x128] | v[32x128]
z   = attn_gate @ h         [4096]
beta  = sigmoid(ssm_beta @ h)                         [32]
g     = softplus(ssm_alpha @ h + ssm_dt) * ssm_a      [32]   (ssm_a = -exp(A_log), already in the file)
conv: window of the last 4 qkv values per channel (conv state: 3 x 8192),
      out = sum_j conv1d[j][c] * window[j][c];  qkv' = silu(out)
q, k = l2norm(q), l2norm(k) per head: rmsnorm with eps/128, then * 1/sqrt(128)
V heads = 32, K heads = 16: V head j uses K head (j % 16)      # ggml_repeat_4d
q = q * 1/sqrt(128)
for each head j (S = 128x128 state in f32):
  S  = S * exp(g[j])
  sk = S^T k                       (sum over the rows)
  d  = (v - sk) * beta[j]
  S  = S + k ⊗ d
  o  = S^T q
y = rmsnorm(o per head, 128) * ssm_norm * silu(z)     # build_norm_gated
a = ssm_out @ y                    [2048]
```
✅ Verified against llama.cpp (step 3): S stored transposed as in ggml (`s[j*128 + i] = S[i][j]`),
V head j → K head `j % 16`, scale 1/sqrt(128) applied to the output.

### attention(h, il)
```
qg = attn_q @ h        [8192] = per head: q[256] | gate[256]  (interleaved per head)
k  = attn_k @ h        [512]  = 2 heads x 256
v  = attn_v @ h        [512]
q = rmsnorm per head * attn_q_norm ;  k = rmsnorm per head * attn_k_norm
RoPE "multi" (NEOX, sections [11,11,10,0]) on the first 64 dimensions of q and k, position = token index
attn = softmax(q k^T / sqrt(256)) v   with GQA 8:1 (Q heads 0..7 -> KV 0, 8..15 -> KV 1)
attn = attn * sigmoid(gate)
a = attn_output @ attn    [2048]
```

### moe(h, il)
```
p = softmax(ffn_gate_inp @ h)                  [256]
top = the 8 highest indices;  w = p[top] / sum(p[top])         # norm_w = true
per e in top:  out += w_e * down_e( silu(gate_e @ h) * (up_e @ h) )
s = sigmoid(ffn_gate_inp_shexp @ h)            [1]
out += s * down_sh( silu(gate_sh @ h) * (up_sh @ h) )
```

## How it is verified (M1)
1. Tokenizer: same ids as llama.cpp on the same text.
2. Layer by layer: dump from llama.cpp (same prompt) of the intermediate tensors, compared with mechta.
3. Tokens: same tokens in greedy decode for the first N steps.

Note (step 3): llama.cpp CPU quantizes x to 8 bits before every product with quantized weights and
accumulates the attention in f16 (flash attention). mechta stays in f32: more precise, hence not identical.
