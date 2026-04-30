---
name: cuda-attention-kernel-patterns
description: Patterns and pitfalls for the ONNX domain Attention operator (opset 23/24) CUDA implementation. Use when modifying the dispatch cascade in core/providers/cuda/llm/attention.cc, writing mask/bias CUDA kernels, debugging attention test routing, or adding features to the ONNX Attention op. NOT for contrib domain MultiHeadAttention/GroupQueryAttention.
---

# ONNX Domain Attention (Opset 23/24) CUDA Patterns

Reusable knowledge from ONNX Attention CUDA development in ORT.

> **Scope**: This skill covers the **ONNX domain** `Attention` operator (opset 23/24)
> implemented at `core/providers/cuda/llm/attention.cc`. This is **separate from** the
> contrib domain `MultiHeadAttention` / `GroupQueryAttention` at `contrib_ops/cuda/bert/`.
> They share some underlying kernels (CUTLASS FMHA, Flash Attention) and infrastructure
> (`attention_softmax.h`) but have **different dispatch logic, parameter structs, and eligibility checks**.
>
> - **Shared infrastructure**: CUTLASS FMHA kernel, Flash kernel, unified unfused kernel
>   (`unfused_attention.cu`), `attention_softmax.h`, `attention_impl.cu` (contrib only)
> - **ONNX-specific**: Dispatch cascade in `attention.cc`, `ConvertAttnMaskToBias`,
>   `mask_filter_value` cap, parameter bridge to contrib structs, `attention_mask_impl.cu`
> - **Contrib-specific**: Own dispatch in contrib MHA/GQA ops, uses `contrib::AttentionParameters`
>   directly, has XQA kernel, past-present buffer sharing

## 1. Runner Dispatch Cascade

CUDA attention dispatches in priority order: **Flash → MEA (Memory Efficient) → Unified Unfused Attention**.

```
// onnxruntime/core/providers/cuda/llm/attention.cc — ComputeInternal()
Flash eligible?      → RunFlashAttention()
  ↓ no
MEA eligible?        → RunMemoryEfficientAttention()
  ↓ no
Unified Unfused      → RunUnfusedAttention()
  (handles both MHA and GQA via reshape-Q trick)
```

**Flash eligibility**: fp16/bf16 only, SM≥8.0 (Ampere+), `head_size == v_head_size`, `head_size <= 256`, no `output_qk`, `attn_mask == nullptr`. Uses `mha_fwd` / `mha_fwd_kvcache`.

**MEA eligibility**: SM50+/53+/80+ by dtype, `head_size <= 1024` and divisible by 8, no `output_qk`. Decode requires `head_size == v_head_size` (for `LaunchConcatNewToPastKV`). Bias stride must satisfy `total_sequence_length % 4 == 0`. GQA with FP32 is excluded (LaunchUngroup only has fp16/bf16 instantiations). Supports `softcap + attn_mask` — CUTLASS applies softcap before bias in kernel tiles, matching ONNX spec ordering (onnx/onnx#7865).

**Unified Unfused Attention**: Always available as the final fallback. Handles both MHA (`num_heads == kv_num_heads`, group=1) and GQA (`num_heads != kv_num_heads`, group>1) via a reshape-Q trick with stride-based cuBLAS batched GEMM (no K/V head replication). Uses FP32 QK scratch for precision. Supports all features:
- softcap + attn_mask (spec-correct ordering)
- output_qk (kQK mode: copies raw QK before softcap/mask mutations)
- past_key + past_value with `head_size != v_head_size` (separate K/V concat)
- causal masking, nonpad_kv_seqlen, all dtypes (fp16/bf16/fp32)

## 2. CUTLASS kLog2e Overflow

CUTLASS `iterative_softmax` multiplies all attention scores by `kLog2e ≈ 1.4427` internally (for `exp2f` instead of `expf`). For float/bf16:

```
mask_filter_value = std::numeric_limits<float>::lowest() ≈ -3.40e+38
-3.40e+38 × 1.4427 ≈ -4.91e+38  →  overflows fp32  →  -inf
```

When all values become `-inf`, CUTLASS's special-case path produces `s_prime=0` → `1/s_prime=inf` → `0 × inf = NaN`.

**Fix**: Cap `mask_filter_value` to `-1.0e+30f` in `ConvertAttnMaskToBias`. This value is safe: `1e30 × 1.4427 ≈ 1.4e30 << FLT_MAX`, and `exp(-1e30) ≈ 0` (effectively masked).

**fp16 is NOT affected**: `lowest() = -65504`, and `-65504 × 1.4427 ≈ -94500` stays within fp32 range.

This cap is ONLY applied in MEA paths. The unfused path uses `lowest()` directly (its softmax subtracts max first, avoiding overflow).

**Subtlety**: When bias is present (`kSupportsBias=true`), CUTLASS pre-applies `p.scale` to QK (line 858) and uses `scaling=1.0f` in the softmax loop (line 981). So the full `kLog2e` multiplier hits the bias-dominated values — the overflow is head_size-independent. Without bias, `scaling = p.scale * kLog2e = kLog2e/sqrt(head_size)`, which is much smaller.

## 3. Bias Alignment

CUTLASS FMHA requires the attention bias row stride to satisfy minimum alignment. The bias has shape `[B, H, S, T]` where `T = total_sequence_length` is the row stride.

```cpp
constexpr int min_bias_align = 4;  // elements, not bytes
if (parameters.total_sequence_length % min_bias_align != 0) {
    mea_eligible = false;  // fall through to unfused
}
```

**Impact on tests**: If a test uses `total_sequence_length` not divisible by 4 (e.g., past=5 + new=6 = 11), MEA is rejected and unfused handles it. To test MEA with bias, ensure `total_sequence_length % 4 == 0`.

## 4. Softcap Ordering

ONNX spec ordering (onnx/onnx#7865): `QK → scale → softcap → add mask/bias → softmax`

- **MEA (CUTLASS)**: Fuses softcap before bias in kernel tile loop (`kernel_forward.h`). Matches spec ordering.
- **Flash**: Handles softcap natively in `mha_fwd`/`mha_fwd_kvcache` but rejects `attn_mask`, so ordering with mask is moot.
- **Unfused**: Handles spec-correct ordering in the fused softmax kernel: `QK → scale → softcap → add bias → softmax`.

All three paths apply softcap BEFORE mask/bias. If softcap were applied after masking, `tanh(-inf/sc) = -sc` (finite), leaking probability to masked positions.

The unfused path does: `QK → scale → softcap → add bias → softmax` (all fused in `UnfusedSoftmaxKernel`).

## 5. Grid-Stride Loops for CUDA Kernels

Always cap grid size to prevent exceeding `gridDim.x` limits, and use grid-stride loops for large workloads:

```cpp
constexpr int64_t kMaxGridDimX = 65535;
int threads = static_cast<int>(std::min(static_cast<int64_t>(max_threads_per_block), total));
int64_t blocks = (total + threads - 1) / threads;
unsigned int grid_size = static_cast<unsigned int>(std::min(blocks, kMaxGridDimX));

MyKernel<<<grid_size, threads, 0, stream>>>(...);

// Inside the kernel:
for (int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
     idx < total;
     idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    // work
}
```

**Never** cast `int64_t` block count directly to `unsigned int` without capping — it silently truncates.

Always call `CUDA_CALL(cudaGetLastError())` after kernel launches in standalone helper functions. This is the established pattern in the file (see `ConcatPastToPresent`, `PastPresentBufferShare`).

## 6. Fully-Masked Batches

All-false bool masks or `seqlens_k=0` produce NaN in CUTLASS MEA.

**Additive-bias path** (bool mask converted to bias): Fixed by capping `mask_filter_value` to `-1e+30f` (see section 2). CUTLASS then naturally computes uniform softmax → mean(V).

**Nonpad path** (`seqlens_k=0`): CUTLASS skips all K/V positions → `s_prime=0` → NaN. Fixed by `ZeroOutputForFullyMaskedBatches` kernel which zeros output for batches where `seqlens_k[b] == 0`. Note: this produces zeros, not mean(V) — a cross-EP consistency TODO exists.

**CPU/Unfused behavior**: `mask_filter_value = lowest()` (not `-inf`). All masked values are equal → `softmax(equal) = 1/N` → output = mean(V). This is the spec reference.

## 7. Test Runner Targeting

Use `ScopedEnvironmentVariables` to force specific CUDA runners:

```cpp
// Force MEA (disable Flash)
ScopedEnvironmentVariables scoped_env({
    {"ORT_DISABLE_FLASH_ATTENTION", "1"},
});

// Force Unfused (disable both Flash and MEA)
ScopedEnvironmentVariables scoped_env({
    {"ORT_DISABLE_FLASH_ATTENTION", "1"},
    {"ORT_DISABLE_MEMORY_EFFICIENT_ATTENTION", "1"},
});
```

**Always verify which runner a test actually hits.** A test designed for MEA may silently fall to unfused if:
- `total_sequence_length % 4 != 0` (bias alignment)
- `head_size != v_head_size` (decode path)
- fp32 dtype with GQA (LaunchUngroup fp16/bf16 only)
- fp32 dtype on SM < 80

Enable verbose logging to confirm: `LOGS_DEFAULT(VERBOSE) << "ONNX Attention: using ..."`.

## 8. Cross-EP Consistency

CPU is the spec reference implementation. CUDA outputs should match CPU for all valid inputs.

- CPU uses `mask_filter_value = std::numeric_limits<T>::lowest()` (finite, not `-inf`)
- CPU softmax: subtract-max-first → works correctly with extreme finite values
- CPU handles fully-masked batches naturally (uniform softmax → mean(V))

Run tests with `disable_cpu=false` to always validate against CPU. The C++ test framework (`RunTest4D`) supports `disable_cpu`, `disable_cuda`, `disable_dml` flags.

## 9. File Locations

### ONNX Domain (this op's code)

| File | Purpose |
|------|---------|
| `core/providers/cuda/llm/attention.cc` | ONNX Attention CUDA dispatch: Flash/MEA/Unfused cascade, `ConvertAttnMaskToBias`, parameter setup |
| `core/providers/cuda/llm/attention_mask_impl.cu` | ONNX-specific mask/bias CUDA kernels: bool→bias, nonpad→seqlens_k, ZeroOutput, bias composition |
| `core/providers/cuda/llm/attention_mask_impl.h` | Declarations for ONNX mask/bias kernels |
| `core/providers/cpu/llm/attention.cc` | CPU reference implementation (ONNX domain) |
| `core/providers/cpu/llm/attention_helper.h` | ONNX parameter validation and shape computation |
| `test/providers/cpu/llm/attention_op_test.cc` | C++ attention tests (all EPs) |
| `test/python/transformers/test_onnx_attention/test_mha.py` | Python parity tests |
| `test/python/transformers/test_onnx_attention/common.py` | Python test utilities and reference `attention_ref()` |

### Shared Infrastructure (used by both ONNX and contrib ops)

| File | Purpose |
|------|---------|
| `contrib_ops/cuda/bert/unfused_attention.cu` | Unified unfused attention: QK GEMM (FP32), fused softmax kernel (scale+softcap+bias+causal), V GEMM. Handles MHA and GQA. |
| `contrib_ops/cuda/bert/unfused_attention.h` | `UnfusedAttentionParams`, `LaunchUnfusedAttention`, workspace size |
| `contrib_ops/cuda/bert/attention_impl.cu` | Legacy unfused `QkvToContext` (contrib MHA only). Also `ApplySoftcap`, `ConcatPastToPresent` |
| `contrib_ops/cuda/bert/attention_softmax.h` | CUDA softmax kernels (`ComputeSoftmax`, `ComputeSoftmaxWithRawMask`) — used by legacy contrib path |
| `contrib_ops/cuda/bert/cutlass_fmha/` | CUTLASS FMHA (Memory Efficient Attention) kernels |
| `contrib_ops/cuda/bert/flash_attention/` | Flash Attention kernels |

### Contrib Domain (separate ops, NOT covered by this skill)

| File | Purpose |
|------|---------|
| `contrib_ops/cuda/bert/multihead_attention.cu` | Contrib `MultiHeadAttention` — own dispatch, uses `contrib::AttentionParameters` directly |
| `contrib_ops/cuda/bert/group_query_attention.cu` | Contrib `GroupQueryAttention` — has XQA kernel, past-present buffer sharing |

## 10. Parameter Bridge (ONNX → Contrib)

The ONNX Attention op uses `attention_helper::AttentionParameters` (in `core/providers/cpu/llm/attention_parameters.h`). The unified unfused kernel (`LaunchUnfusedAttention`) uses its own `UnfusedAttentionParams` struct populated directly from ONNX parameters in `RunUnfusedAttention`.

The contrib `QkvToContext` function (used by contrib MHA, NOT by ONNX Attention) uses `contrib::AttentionParameters`. ONNX Attention does **not** bridge to `contrib::AttentionParameters` — it routes through the unified unfused kernel instead.

## 11. Causal Alignment

The ONNX spec defines two causal alignment modes based on where query positions sit in the full attention matrix:

- **Upper-left**: `q_i` attends to `kv[0..i]`. Query positions start at 0 in the full matrix.
- **Lower-right**: `q_i` attends to `kv[kv_len - q_len + i..kv_len - 1]`. Query positions are at the end.

**ONNX spec rule**: `is_causal=1` always means upper-left in the full matrix. When `past_key` provides context, `past_sequence_length` shifts the query start position forward — the resulting `[S_q × total_kv]` sub-matrix effectively has lower-right alignment.

### Per-kernel behavior

| Kernel | Alignment | Mechanism |
|--------|-----------|-----------|
| **Flash** | Lower-right only | `is_causal` flag → `seqlen_k - seqlen_q` offset in kernel. No top-left option. |
| **MEA (CUTLASS)** | Both | `causal_from_top_left` flag in `MemoryEfficientAttentionParams`. `true` → `CausalFromTopLeft` (offset=0). `false` → `CausalFromBottomRight` (offset = num_keys - num_queries). |
| **Unfused** | Both | `past_kv_length` param. `0` → upper-left. `total_kv - S_q` → lower-right. |

### Dispatch logic in attention.cc

```cpp
// Flash cannot do upper-left → guarded by causal_cross_no_past
bool causal_cross_no_past = parameters.is_causal &&
    parameters.q_sequence_length != parameters.total_sequence_length &&
    parameters.past_sequence_length == 0;

// Flash: skip when causal_cross_no_past (no top-left support)
// MEA: NOT skipped — handles it via causal_from_top_left = (past_sequence_length == 0)
// Unfused: always correct via past_kv_length = parameters.past_sequence_length
```

### When S_q == S_kv

Upper-left and lower-right produce **identical** results when `S_q == S_kv` (the offset is 0 either way). The alignment distinction only matters for cross-attention shapes (`S_q != S_kv`).

### TensorScatter decode (opset 24 external KV cache)

TensorScatter manages KV cache externally — `past_key` is nullptr but K/V already contain the full sequence. Per the ONNX spec, `is_causal` with `S_q != S_kv` and no `past_key` means upper-left (q[0] sees only kv[0]), which is **not meaningful for decode**.

**Correct pattern**: TensorScatter decode must use `is_causal=0` and rely on `nonpad_kv_seqlen` to bound the active KV range. Models using `is_causal=1` with TensorScatter decode have a spec-invalid combination.
