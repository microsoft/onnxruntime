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

### Eligibility anchors (symbols are stable; line numbers as of cc34d0b914)

| Stage | Decision symbol | Hard caps | Dispatch gate |
|---|---|---|---|
| Flash | `flash::is_supported<T>` (`flash_api.cc:414`) | fp16/bf16 only, SM≥8.0, `head_size%8==0`, **`head_size<=256`** | `attention.cc:1385` `flash_eligible` (fp32 excluded at :1387) |
| MEA | `has_memory_efficient_attention` (`memory_efficient_attention.h:68`) | `(head_size&7)==0` **and `head_size<=kEfficientAttentionMaxHeadSize` (1024)**; **NO shared-memory feasibility check** (see #28388 + the head_size=512 caveat below) | `attention.cc:1415` `mea_eligible`; bias-stride `%4` at :1436 |
| Unfused | (none — catch-all) | all dtypes/shapes | `attention.cc:1485` `RunUnfusedAttention` |

**`head_size=512` IS routed to MEA, but its MEA kernel is NOT portably launchable — so it
is not a robust test probe.**
By the predicate, 512 > 256 fails Flash and 512 ≤ 1024 with `512 & 7 == 0` passes the MEA
predicate, **so dispatch selects MEA** — but the MEA eligibility check
(`memory_efficient_attention.h:68-73`) gates only on SM +
`head&7==0` + `head<=1024`, with **no shared-memory check**. For `head_size=512` FP16 the
CUTLASS MEA `SharedStorage` exceeds the dynamic-smem opt-in cap on capacity-limited arches
(sm86 ~99KB, sm80 ~163KB, sm90 ~227KB — **non-monotonic**, no clean SM-version guard).
`fmha_launch_template.h` calls `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`
but **ignores its return value and launches anyway**, so on sm86 the kernel dies at launch
with `CUDA failure 1: invalid argument` — there is **no fallback to unfused** (live bug
#28388; its fix PR #28383 was never merged). So `head_size=512`'s MEA kernel launches only
on large-smem arches like sm90/H100.

**To force the MEA path portably in a test, use `ORT_DISABLE_FLASH_ATTENTION=1` with a small
`head_size` (e.g. 64) whose SharedStorage fits every target arch — NOT `head_size=512`.**
Also guard with `SKIP_IF_MEA_NOT_COMPILED` (see §7) so a MEA-OFF build SKIPs rather than
false-greens via the (correct) unfused fallback.

**Flash eligibility**: fp16/bf16 only, SM≥8.0 (Ampere+), `head_size == v_head_size`, `head_size <= 256`, no `output_qk`, `attn_mask == nullptr`. Uses `mha_fwd` / `mha_fwd_kvcache`.

> **QUICK_BUILD caveat (false hypothesis trap).** *(General principle — build flags can
> silently reroute kernel dispatch — lives in the `ort-build` skill, "Agent tips". The
> attention-specific instance:)* With `onnxruntime_QUICK_BUILD=ON`
> (`-DORT_QUICK_BUILD`), Flash is compiled for **head_dim 128 only**:
> `flash_api.h:147` `is_supported<T>` returns false for `head_size != 128`, and
> `static_switch.h:80` `HEADDIM_SWITCH` only instantiates `kHeadDim=128`. So under
> QUICK_BUILD nearly every shape routes to **MEA**, not FlashAttention-2. If a
> `head_size!=128` test "fails only on some SM", suspect **MEA** (CUTLASS,
> arch-independent), NOT a Flash/FA2 hardware bug. `head_size=512` **is routed to MEA** in all
> MEA-enabled builds (Flash caps at 256), but its MEA kernel **fails to launch** on
> small-smem GPUs — see the `head_size=512` caveat above (#28388).

**MEA eligibility**: SM50+/53+/80+ by dtype, `head_size <= 1024` and divisible by 8 (enforced by `has_memory_efficient_attention`), no `output_qk`. GQA additionally requires `head_size == v_head_size` (for `LaunchUngroup`); decode also requires it (for `LaunchConcatNewToPastKV`). Bias stride must satisfy `total_sequence_length % 4 == 0`. GQA with FP32 is excluded (LaunchUngroup only has fp16/bf16 instantiations). Supports `softcap + attn_mask` — CUTLASS applies softcap before bias in kernel tiles, matching ONNX spec ordering (onnx/onnx#7867, supersedes the now-closed onnx/onnx#7865 issue).

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

ONNX Attention opset 23/24 spec ordering (per onnx/onnx#7867, which superseded
the now-closed onnx/onnx#7865 issue, and onnx/onnx#7913 which swapped
`qk_matmul_output_mode` values 1 and 2 to align with the corrected pipeline):

```
scale * (Q @ K^T)        # stage 0: raw scaled QK
    |
softcap (if > 0)         # stage 1: tanh(qk / softcap) * softcap
    |
+ attn_bias / + attn_mask # stage 2: additive (mask -inf survives to stage 3)
    |
softmax                  # stage 3
    |
@ V
```

`qk_matmul_output_mode` integer values follow pipeline stage order:
0 = raw scale*QK, 1 = post-softcap (pre-mask), 2 = post-mask/bias (pre-softmax),
3 = post-softmax.

CUDA implementation status (all spec-correct):
- **MEA (CUTLASS)**: `kernel_forward.h` applies softcap inside the score-compute
  tile loop BEFORE `attn_bias` is added.
- **Flash**: `mha_fwd` / `mha_fwd_kvcache` handle softcap natively; reject
  explicit `attn_mask`, so ordering with float mask is moot for this path.
- **Unfused**: `UnfusedSoftmaxKernel` does `QK -> scale -> softcap -> add bias -> softmax`
  (all fused).

CPU implementation status: `core/providers/cpu/llm/attention.cc::ComputeAttentionProbs<T>`
applies softcap BEFORE the mask add (post-fix; pre-fix it inverted the order
and leaked probability through masked positions).

Why this ordering matters: a -inf in `attn_mask` must survive to softmax. If
softcap were applied AFTER the mask-add, then `tanh(-inf/softcap) * softcap = -softcap`
(a finite value), and softmax would assign non-zero weight to the masked
position — leaking poison V values into the output. The CUDA-side guard tests
at `test_onnx_attention/test_gqa.py:1501` and `:1761`, and the CPU-side guards
at `TestONNXAttentionCPUSoftcapMaskOrdering` in the same file, exercise this
property by combining small softcap, a -inf mask entry, and a poison V value.

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

## 6. Fully-Masked Rows and Batches

All-false bool masks, an all-`-inf` `attn_mask` row, or a causal/nonpad frontier
with no allowed key produce NaN in CUTLASS MEA (the uniform/empty softmax degenerates:
`s_prime=0` → `1/s_prime=inf` → `0 × inf = NaN`). Per onnx/onnx#8068 (Bug-2), a
**fully-masked query row** — one with no key allowed by the composed causal + nonpad +
mask constraints — must output a **zero row** (`Y = 0`), **not** mean-of-V.

**This `Y = 0` behavior is now consistent on BOTH EPs** (the earlier mean(V)-vs-zero
cross-EP divergence is RESOLVED — there is no longer an open TODO here):

- **CUDA**: `ZeroFullyMaskedRowsKernel` (in `attention_mask_impl.cu`) runs after the
  MEA/CUTLASS output and zeros each fully-masked row with a **select** (not multiply,
  so `0 @ V = 0` even when V is poisoned). It detects a fully-masked row with an exact
  per-key predicate (within the causal/nonpad frontier AND the additive-bias slot is
  above the mask sentinel), matching the onnx#8068 `isneginf`-of-row-max reference. A
  finite (even very negative) user bias is not the sentinel, so its key stays unmasked
  and the row is left untouched.
- **CPU**: `core/providers/cpu/llm/attention.cc` applies the same Bug-2 guard — after
  softmax it zeros any row whose composed frontier admitted no unmasked key.

**Additive-bias path** (bool mask converted to bias): `mask_filter_value` is capped to
`-1e+30f` (see section 2) so CUTLASS does not overflow to NaN; a row that is nonetheless
fully masked is then zeroed by the per-row guard above.

**Whole-batch empty (`seqlens_k[b] == 0`)**: the structural case where an entire batch
has zero valid keys is additionally handled by `ZeroOutputForFullyMaskedBatches`, which
zeros that batch's output. (The per-row guard covers the finer-grained case where only
some query rows are fully masked.)

**`qk_matmul_output_mode` (mode 3 / post-softmax debug output)**: for a fully-masked row
the mode-3 snapshot is **mandated to be `0`** (zero row), consistent with `Y = 0`, per the
onnx#8068 SIG decision (this superseded the earlier "unspecified" proposal). The CPU
post-softmax snapshot is taken **after** the row-zeroing guard — matching the onnx
reference and the v23/v24 function bodies, where the guard runs before the mode-3
capture — so the debug tensor reflects the same zero row as the output. Note this
mode-3=0 behavior is served by the **CPU** path: CUDA `qk_matmul_output_mode` beyond
`kNone`/`kQK` (i.e. `kPostSoftCap`/`kPostMaskBias`/`kPostSoftMax`) returns
`NOT_IMPLEMENTED` (`attention.cc`), so an agent must not assume CUDA produces mode-3=0.

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

**`SKIP_IF_MEA_NOT_COMPILED`** is a local gtest macro (defined in
`test/providers/cpu/llm/attention_op_test.cc`) that `GTEST_SKIP`s — rather than silently
passes — when `USE_MEMORY_EFFICIENT_ATTENTION` is OFF, so an MEA-targeted test cannot
false-green via the (correct) unfused fallback. Use it in any test that must prove the MEA
path ran (see the `ort-test` skill → "Verify which path/kernel actually executed").

## 8. Cross-EP Consistency

CPU is the spec reference implementation. CUDA outputs should match CPU for all valid inputs.

- CPU uses `mask_filter_value = std::numeric_limits<T>::lowest()` (finite, not `-inf`)
- CPU softmax: subtract-max-first → works correctly with extreme finite values
- CPU zeros fully-masked query rows (onnx#8068 Bug-2 guard) — output `Y = 0`, matching
  CUDA's `ZeroFullyMaskedRowsKernel`. (Earlier docs claimed CPU produced mean(V) here;
  that divergence is resolved — both EPs now emit a zero row.)

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
| `test/providers/cpu/llm/attention_op_test.cc` | C++ tests for the **ONNX-domain** `Attention` op — suite `AttentionTest.*`, runs in `onnxruntime_provider_test` (all EPs). NOT to be confused with the contrib `test/contrib_ops/attention_op_test.cc` (`ContribOpAttentionTest.*`); see `ort-test` skill. |
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

- **Upper-left** (a.k.a. *top-left*): `q_i` attends to `kv[0..i]`. Query positions start at 0 in the full matrix.
- **Bottom-right** (a.k.a. *lower-right*): `q_i` attends to `kv[0 .. kv_len - q_len + i]` — i.e. keys `j` with `j <= i + offset`, where `offset = kv_len - q_len` (clamped `>= 0`). The causal diagonal is anchored at the end of the key axis. This is the term onnx/onnx#8068 uses; kernel flags spell it `CausalFromBottomRight`.

**ONNX spec rule**: causal alignment depends on how the KV context is supplied.

- **Internal cache / no cache** (`past_key`, or plain self-attention): `is_causal=1`
  is upper-left in the full matrix. When `past_key` provides context,
  `past_sequence_length` shifts the query start position forward — the resulting
  `[S_q × total_kv]` sub-matrix is effectively bottom-right.
- **External / static cache** (`nonpad_kv_seqlen`, no `past_key`, opset 24): per
  onnx/onnx#8068, `is_causal=1` uses **bottom-right** (offset-aware) alignment —
  query in-block index `i` attends key `j` iff `j <= i + offset[b]`, where
  `offset[b] = nonpad_kv_seqlen[b] - q_sequence_length` (clamped to `>= 0`).

### Per-kernel behavior

| Kernel | Alignment | Mechanism |
|--------|-----------|-----------|
| **Flash** | Bottom-right only | `is_causal` flag → `seqlen_k - seqlen_q` offset in kernel. No upper-left option. |
| **MEA (CUTLASS)** | Both | `causal_from_top_left` flag in `MemoryEfficientAttentionParams`. `true` → `CausalFromTopLeft` (offset=0). `false` → `CausalFromBottomRight` (offset = num_keys - num_queries). |
| **Unfused** | Both | `past_kv_length` param. `0` → upper-left. `total_kv - S_q` → bottom-right. |

### Dispatch logic in attention.cc

```cpp
// Pure cross-attention with NO external cache (S_q != S_kv, no past, no nonpad):
// this is the upper-left case Flash cannot express.
bool causal_cross_no_past = parameters.is_causal &&
    parameters.q_sequence_length != parameters.total_sequence_length &&
    parameters.past_sequence_length == 0;

// Flash: eligible UNLESS (causal_cross_no_past && nonpad_kv_seqlen == nullptr).
//   - No external cache  -> upper-left required -> skip Flash (no upper-left support).
//   - External cache (nonpad_kv_seqlen != nullptr) -> required frontier IS bottom-right
//     (onnx#8068), so Flash IS eligible and produces it natively via seqlens_k.
// MEA: external cache -> causal_from_top_left = false (bottom-right, offset = num_keys -
//   num_queries == nonpad_kv_seqlen[b] - q_len per batch); otherwise causal_from_top_left
//   = (past_sequence_length == 0).
// Unfused: always correct via past_kv_length (0 -> upper-left; total_kv - S_q -> bottom-right).
```

### When S_q == S_kv

Upper-left and bottom-right produce **identical** results when `S_q == S_kv` (the offset is 0 either way). The alignment distinction only matters for cross-attention shapes (`S_q != S_kv`).

### TensorScatter decode (opset 24 external KV cache)

TensorScatter manages KV cache externally — `past_key` is nullptr but K/V already
contain the full sequence, with `nonpad_kv_seqlen[b]` giving each batch's valid
(non-padded) key count. Per onnx/onnx#8068, `is_causal=1` with an external/static KV
cache (no `past_key`) uses **bottom-right** (offset-aware) alignment: query in-block
index `i` attends key `j` iff `j <= i + offset[b]`, where
`offset[b] = nonpad_kv_seqlen[b] - q_sequence_length` (clamped to `>= 0`). For decode
(`q_sequence_length == 1`) the single query row therefore attends all
`nonpad_kv_seqlen[b]` valid keys — the meaningful, spec-correct result (not the
degenerate "q[0] sees only kv[0]" of upper-left).

**Correct pattern**: `is_causal=1` with TensorScatter + `nonpad_kv_seqlen` (no `past_key`)
is **valid and supported** for both decode and continued-prefill — it yields bottom-right
causal attention bounded by the per-batch valid-key count. (`is_causal=0` is also valid
where a model wants no causal masking.) The earlier `is_causal=1` NOT_IMPLEMENTED reject
was **removed** in the onnx#8068 alignment work; the only still-invalid combination is
`nonpad_kv_seqlen` together with `past_key` (mutually exclusive internal-vs-external
cache, enforced at validation in `attention_helper.h`).

## 12. Signed Offsets in CUTLASS FMHA (uint wrap hazard)

This is a specific instance of the general **signed-vs-unsigned wrap** bug class — see
`AGENTS.md` → "Signed vs unsigned on negative-capable differences" for the principle. Below
are the **attention-specific** fix sites in `cutlass_fmha/kernel_forward.h`. See §11 for what
the offset *means* (bottom-right alignment); this section is purely the signed-arithmetic
hazard.

Any FMHA offset computed as a difference of counts — canonically
`causal_diagonal_offset = num_keys - num_queries` (`CausalFromBottomRight`) — is
**negative** whenever `num_keys < num_queries` (cross-attention / KV-trimmed /
`nonpad_kv_seqlen[b] < q_len`, onnx#8068 / ORT #28904). It **must** be stored and
compared as `int32_t`; a `uint32_t` wraps the negative value to ~4.29e9 (`0xFFFFFFFE`),
the causal-mask guard `min(iter_key_start + kKeysPerBlock, num_keys) >= query_start + offset`
becomes permanently false, the per-element causal mask is **silently skipped**, and boundary
query rows over-attend one extra key.

### Fix sites in `cutlass_fmha/kernel_forward.h` (symbols are stable; lines as of cc34d0b914)

| Symbol / guard | Line | What it must do |
|---|---|---|
| `int32_t causal_diagonal_offset` (field decl) | ~206 | Stay **`int32_t`** so the negative offset is preserved (rationale comment ~202-205). |
| `causal_diagonal_offset = num_keys - num_queries;` | ~354 | Set point for `CausalFromBottomRight`; may be negative (comment ~353). |
| `int32_t(query_start + causal_diagonal_offset + kQueriesPerBlock)` | ~366 | First (AttentionKernel) `num_keys` clamp. The inner sum **does** wrap to `0xFFFFFFFE`-style values in unsigned arithmetic when the offset is negative, but casting the **whole sum** to `int32_t` recovers the correct value by two's-complement modular arithmetic, and the result is consumed **arithmetically** (as a `fast_min` operand), so the wrap is harmless. Contrast the ~924 guard, where the value feeds a **relational** comparison — there the unsigned wrap flips the comparison result, so the operand `query_start` must be cast to `int32_t` **before** the compare. |
| "Mask out last if causal" guard: `static_cast<int32_t>(query_start) + p.causal_diagonal_offset` | ~924-926 | `query_start` is `uint32_t` (~707) — cast it to `int32_t` so the comparison is signed (rationale ~919-923). |
| Sliding-window guard ("L957"): `static_cast<int32_t>(query_start) + p.causal_diagonal_offset ...` | ~962-963 | Same cast hardening (rationale ~956-961). |

### Rules when editing `kernel_forward.h` (or any FMHA kernel)

- Keep `causal_diagonal_offset` **`int32_t`**.
- `query_start` in the iteration kernels is **`uint32_t`** — `static_cast<int32_t>(query_start)`
  before adding the offset in ANY relational guard.
- The same hazard is **dormant but real** at the `window_size > 0` guard: harden it the
  same way even though opset-24 Attention currently pins `window=-1` (a future
  sliding-window / KV-trim caller could combine `window_size>0` with a negative offset).
- Tests that exercise this need a **negative** offset: `num_keys < num_queries`. Force the
  MEA path **portably** with `ORT_DISABLE_FLASH_ATTENTION=1` + a small `head_size` (e.g. 64)
  — **not** `head_size=512`, whose MEA launch is arch-fragile on small-smem GPUs (#28388,
  see §1). The regression tests live in `test/providers/cpu/llm/attention_op_test.cc`
  (`Attention_Causal_NonPadKVSeqLen_MEA_*`), guarded by `SKIP_IF_MEA_NOT_COMPILED`.
