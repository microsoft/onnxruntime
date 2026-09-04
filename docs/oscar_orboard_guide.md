# Enabling OSCAR 2-bit KV cache

The OSCAR 2-bit KV cache is an in-place extension of `com.microsoft.GroupQueryAttention` (GQA).
It applies to **any** ONNX model whose attention is exported as GQA nodes with the KV cache exposed
as graph I/O. Converting a model is: (1) an **offline preparation** step that builds the spectral
rotations and rewrites the graph, then (2) two session-config entries at inference (plus rolling an
extra KV window if you use mixed precision). CPU EP, `float` and `float16` compute.

Notation used below: `L` = number of GQA layers, `H_kv` = KV heads, `H_q` = query heads,
`D` = `head_size`, `G` = `kv_quant_group_size`.

## Step 0 — start from a GQA export

OSCAR patches **`GroupQueryAttention`** nodes, so the model must be exported with GQA (not
`MultiHeadAttention`) and expose the cache as `past_key_values.N.key/value` → `present.N.key/value`.
Any exporter that produces `com.microsoft.GroupQueryAttention` works (e.g. the ONNX Runtime GenAI
model builder). If you only have an fp16 MHA export, re-export/convert to an fp16 **GQA** graph first.

## Step 1 — offline preparation

This is a **post-training** step (no retraining): calibrate the spectral rotations, then rewrite the
GQA graph to the 2-bit layout. Both are done once, offline.

### 1a. Calibrate the spectral rotations (optional but recommended)

Rotations `R_K`/`R_V` are what make 2-bit accurate. Run the original (unquantized) model on a small,
representative calibration corpus and tap each attention layer for the **post-RoPE** query/key/value
and the attention probabilities `S`. Accumulate attention-aware covariances per layer (and per KV
head, or averaged over heads for one rotation per layer):

$$C_Q=\sum q^\top q \quad(\text{key target}),\qquad C_S=\sum (SV)^\top (SV)\quad(\text{value target})$$

Then, per (layer, KV head): symmetrize, take the eigenvectors ordered by descending eigenvalue
(`U_Q`, `U_S`), and compose the OSCAR rotation `R = U · H · P` (`H` = normalized Hadamard,
`P` = bit-reversal permutation; `D` should be a power of two for a padding-free Hadamard). Each
`R_K`/`R_V` is a `(H_kv, D, D)` orthogonal matrix per layer.

This is framework-agnostic — any way of capturing Q/K/V/`S` works. The reference implementation
hooks the model's eager-attention function in PyTorch:

```python
# framework-neutral sketch (any model that exposes post-RoPE q,k,v and attn probs S)
C_Q[layer] += einsum("...td,...te->de", q, q)          # q: [B, H_kv, n_rep, T, D]
sv          = einsum("...ts,...sd->...td", S, v)        # S: [B, H_kv, n_rep, T, T], v: [B,H_kv,T,D]
C_S[layer] += einsum("...td,...te->de", sv, sv)
# after the corpus:
U   = eigenvectors(0.5*(C+C.T), descending)             # C_Q -> R_K,  C_S -> R_V
R   = hadamard_bitreversal(U)                            # (D, D) orthogonal
```

Save `R_K`/`R_V` per layer. You can **skip** calibration entirely (pass no rotations) — the codec
still works, just with higher error. The outlier-clip percentiles `k_quant_rho`/`v_quant_rho` are
also chosen at this stage (typical `0.90–0.97`; `1.0` = no clipping).

### 1b. Rewrite the GQA graph (graph surgery)

For every GQA node in the graph:

1. add the PER_GROUP 2-bit **attributes** (Step 2);
2. **retype** the four KV tensors `past/present .key/.value` from `FLOAT[B, H_kv, S, D]` to
   `UINT8[B, H_kv, S, packed_head_size]` (Step 3);
3. *(optional, mixed precision)* add the high-precision window I/O and wire it into node inputs
   16/17 and outputs 4/5 (Step 4);
4. *(optional, rotations)* add `R_K`/`R_V` as **constant initializers** wired into node inputs 18/19
   (Step 5).

This is pure ONNX graph editing (no retraining) and can be scripted with the `onnx` Python API
against any GQA graph. Copy the model's non-weight sidecar files (tokenizer, config, etc.) alongside
the new model as usual.

## Step 2 — node attributes

| Attribute | Value | Notes |
|---|---|---|
| `k_quant_type` / `v_quant_type` | `"PER_GROUP"` | selects the asymmetric per-group codec |
| `kv_cache_bit_width` | `2` | |
| `kv_quant_group_size` | `G`, e.g. `32`/`64` | must divide `D`; `0` = whole head is one group |
| `k_quant_rho` / `v_quant_rho` | e.g. `0.96` / `0.92` | outlier-clip percentile `(0,1]`; `1.0` = no clip |
| `kv_quant_metadata_fp16` | `0` / `1` | inline scale/zero as fp32 (default) or fp16 |

Scales/zero-points are computed at append time and stored **inline** in the cache, so
`k_scale`/`v_scale` inputs are unused.

## Step 3 — packed KV layout

`past/present .key/.value` become **UINT8** with last dim:

```
packed_head_size = D/4 + num_groups * 2 * meta_bytes
num_groups = D / G ,  meta_bytes = 4 (fp32 metadata) | 2 (fp16 metadata)
```

Worked example (`D=128`, `G=64` → `num_groups=2`): `128/4 + 2·2·4 = 48 B` (fp32 metadata), or `40 B`
with `kv_quant_metadata_fp16=1` (≈2.5 bits/element). Initialize the empty past as
`UINT8[B, H_kv, 0, packed_head_size]`.

## Step 4 — mixed precision: I/O and how to drive it in the session

Keeps the first *sink* and last *recent* tokens unquantized; only the middle history is 2-bit.

**Graph I/O (added in Step 1b):**

- **Inputs 16/17**: `past_hp_key` / `past_hp_value`, shape `(B, H_kv, sink+recent, D)`, **same dtype
  as `query`** (fp32 or fp16 — independent of the uint8 2-bit cache).
- **Outputs 4/5**: `present_hp_key` / `present_hp_value`.

**Session-config** (window sizes; if unset, the hp path is inert):

```python
so.add_session_config_entry("gqa.kv_quant.sink", "64")
so.add_session_config_entry("gqa.kv_quant.recent", "256")
```

**Driving it in the decode loop** — treat the hp tensors as a *second* KV cache: seed each layer
with an empty hp past, then feed each step's `present_hp` back as the next `past_hp`. The kernel
manages the rolling window internally, so you never slice it yourself:

```python
hp_dtype = np.float16 if model_is_fp16 else np.float32
# seed empty hp past (0-length window) for every layer
for i in range(L):
    hp_empty = np.zeros((B, H_kv, 0, D), dtype=hp_dtype)
    feeds[f"past_hp_key_values.{i}.key"]   = hp_empty
    feeds[f"past_hp_key_values.{i}.value"] = hp_empty.copy()

# ... each step, after sess.run(...):
for i in range(L):
    feeds[f"past_key_values.{i}.key"]      = out[f"present.{i}.key"]      # uint8 2-bit cache
    feeds[f"past_key_values.{i}.value"]    = out[f"present.{i}.value"]
    feeds[f"past_hp_key_values.{i}.key"]   = out[f"present_hp.{i}.key"]   # fp window
    feeds[f"past_hp_key_values.{i}.value"] = out[f"present_hp.{i}.value"]
```

For pure 2-bit (no mixed precision), omit all `*_hp_*` tensors and just roll the uint8
`present`→`past`.

## Step 5 — spectral rotations (optional)

Inputs 18/19 `oscar_rotation_k` / `oscar_rotation_v`, shape `(H_kv, D, D)`, embedded as **constant
initializers** (from Step 1a). The kernel rotates post-RoPE Q/K by `R_K` before quantization (QK
scores are invariant to the shared orthogonal rotation) and un-rotates the value output by `R_Vᵀ`.

## Notes / limitations

- `float16` compute is supported (half↔float bridged at the kernel boundary); `attention_bias` and
  `output_qk` are **not** supported on the fp16 2-bit path.
- CPU-only for now (CUDA/WebGPU/DML kernels are future work).
- The rotation axis is `D` (`head_size`); a power-of-two `D` gives a padding-free Hadamard. `G` must
  divide `D`.
