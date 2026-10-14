# CUDA SparsePagedAttention

`com.microsoft.SparsePagedAttention` executes attention over externally selected
request-local indices. It does not perform selection or own indexer state.

## Modes

| `attention_mode` | `selected_kv_source` | Reads |
|---|---|---|
| `selected_only` | `main` | Selected logical positions resolved through the main `block_table` |
| `local_plus_selected` | `auxiliary` | Main paged local window and selected contiguous auxiliary rows in one softmax |

The implementation also accepts the other mode/source combinations. Main-cache
selection is causal when `is_causal=1`. Invalid, out-of-range, non-causal, and
unmapped selected positions are ignored.

`selected_indices` has shape `[token_count, max_selected_entries]`;
`selected_counts` has shape `[token_count]`. Indices are request-local and unused
entries are `-1`.

The contiguous auxiliary cache has shape
`[batch_size, auxiliary_capacity, kv_heads_or_one, head_dim]`, with
`auxiliary_lengths[batch_size]` defining request boundaries. When
`auxiliary_kv_shared=1`, `auxiliary_key` supplies both K and V and
`auxiliary_value` must be absent.

## Support matrix

| Feature | CUDA support |
|---|---|
| Activation | FP16, BF16 |
| Main cache | FP16, BF16, INT8 per-tensor or per-channel |
| Auxiliary cache | Contiguous FP16/BF16, same type as activation |
| Auxiliary K=V | Supported |
| Paged auxiliary cache | Not supported |
| Main cache writes and `slot_mapping` | Same behavior as `PagedAttention` |
| QK-Norm and RoPE | Same prologue as `PagedAttention` |
| Attention sink | Supported in the joint softmax |
| CUDA Graphs | Fixed-capacity, device-resident counts; no host synchronization |

The functional kernel reads selected pages directly and performs an online joint
softmax. Selector fusion, paged auxiliary storage, and auxiliary quantization are
deferred.

## Opt-in CUDA experiments

These process-wide flags are read once. All default to `0` and leave the existing
dispatch and split heuristic unchanged when disabled.

| Flag | Candidate and eligibility |
|---|---|
| `ORT_SPARSE_MTP_GROUPED_HEADS=1` | FP16/BF16 selected-only main-cache attention with head dimension 256, GQA groups divisible by 12, and at most 8 query tokens. Uses the existing twelve-head scalar kernel with four independent dot rows, including split KV and the existing final reduction. Keeps each dot product's accumulation and warp reduction order. |
| `ORT_SPARSE_PREFILL_ASYNC_KV=1` | Uses `cp.async` for aligned sixteen-byte main/current K/V transfers in the twelve-head prefill path (more than 64 query tokens). Unaligned addresses retain the scalar copy fallback; invalid rows retain zero fill. |
| `ORT_SPARSE_PREFILL_TENSOR_CORE_QK=1` | Existing twelve-head prefill tensor-core QK path. |
| `ORT_SPARSE_PREFILL_TENSOR_CORE_PV=1` | Existing FP16 tensor-core PV path; requires prefill tensor-core QK. |

Tensor-core candidates require Ampere or later, compatible compiled kernels, and
sufficient shared memory. Async copies are committed and waited before the
existing block barrier exposes each staged tile to its consumers. This avoids
global-load intermediate registers without changing tile order, softmax, or
outputs. It does not add a cross-tile double buffer.

The small-MTP path deliberately does not use tensor cores, even when the existing
prefill tensor flags are enabled. A tensor-core small-MTP prototype passed
numerical operator parity but changed a paired 512-token output stream, and its
dispatch was rejected. Both retained candidates require bitwise operator parity
against their corresponding controls. The split heuristic and workspace layout
are unchanged.

### Standalone qualification

The standalone harness extracts the actual production device kernels and shared
MMA helpers, avoiding an ORT provider rebuild. Build and run on an otherwise idle
assigned GPU, using a project-local output directory:

```sh
python onnxruntime/test/contrib_ops/cuda_kernels/sparse_paged_attention_standalone.py \
  --output-dir attention-artifacts
CUDA_VISIBLE_DEVICES=7 SPARSE_BENCH_GROUPED=1 \
  attention-artifacts/sparse_benchmark 4 2048 22 0 0 0
```

Positional arguments are query tokens, selected capacity, split count, BF16,
edge-case inputs, and two-byte pointer misalignment. Four variants are reported:
four-head scalar control, twelve-head QK, twelve-head QK/PV (QK only for BF16 or
split KV), and the corresponding async copy variant. Set `SPARSE_BENCH_GROUPED=1`
to replace the second variant with the exact twelve-head scalar MTP candidate
and check its output bitwise against the four-head scalar control. Each report
preserves seven CUDA-event
timing samples over twenty captured operator invocations, including reduction
when split. It checks numerical parity against the scalar control and bitwise
parity of async copies against their synchronous control. Edge inputs include
empty and partial selections, invalid positions, unmapped blocks, current K/V,
invalid slots, attention sinks, and softcap. Set `SPARSE_BENCH_SANITIZER=1` to
reduce repetition when running Compute Sanitizer; those timings are not
performance measurements.

### H200 qualification (2026-10-13)

An isolated provider built at `bfd45538f44fe0e72dbb08350ae98d84bed67f53`
with these changes passed 23 standalone geometries with zero grouped/async
bit mismatches, 224 final NumPy-reference operator checks, and Compute Sanitizer
racecheck/memcheck. A bounded full-model test used an existing selected-logits
NVFP4 model, 8192 plain-text input tokens, MTP1, and exactly 512 output tokens.
Each fresh process ran one excluded warmup and two measured requests, ordered
control/grouped/async/combined/control. Published MoE and tensor-prefill flags
were identical across configurations.

| Candidate | Decode TPS | TTFT (ms) | E2E TPS |
|---|---:|---:|---:|
| Pooled four measured controls | 105.34 | 1186.32 | 84.80 |
| Grouped small-MTP only | 113.89 | 1184.92 | 90.27 |
| Async prefill only | 105.01 | 1121.18 | 85.51 |
| Both flags | 113.65 | 1122.62 | 91.12 |

All fifteen warmup/measured 512-token streams matched bitwise. The control
process summaries varied by 0.44% decode TPS; individual measured controls ranged
104.96–106.19 TPS. Grouped small-MTP improved decode by 8.12%; async prefill
reduced TTFT by 5.49% but did not improve decode. Combined E2E improved 7.45%.
These are bounded 8K results, not cached-prefix or long-context sweep claims.

### Original-provider Engine integration qualification

The same changes were integrated into the original provider and qualified using
only the Engine API, not Generator. Runner opt-ins are
`--sparse-mtp-grouped-heads` and `--sparse-prefill-async-kv`; both default off and
their environment values are recorded in the sweep manifest. The runner's
existing exact512 output default is unchanged.

The integrated source-review prompt contains 8190 tokens. With scheduler8192,
prefix caching enabled but no reusable checkpoint, and decoder/MTP GEMM
autotuning disabled, control/candidate/control each ran one excluded warmup
and three measured exact512 requests. Both flags improved decode
122.58→132.60 TPS, TTFT1203.56→1131.97ms, and E2E95.30→102.70 TPS.
All twelve full output streams matched. Initial/repeat control process decode
summaries were 122.31/122.85 TPS.

A separate genuinely cached comparison used scheduler4096 on both sides,
otherwise the same protocol with autotuning disabled. Every measured request
reported one hit and 4096 matched prefix tokens; all twelve full streams matched
their cold output and each other. Results were decode112.68→120.97 TPS,
TTFT665.35→628.90ms, and E2E98.46→105.50 TPS. Initial/repeat control decode
summaries were 113.39/111.98 TPS, with individual controls110.50–118.08 TPS.
The scheduler4096 results must not be represented as scheduler8192 cache reuse.

An earlier scheduler8192/autotuning-enabled comparison produced different
streams between controls themselves. Its measurements are retained but excluded
from output-equivalent gain claims. Disabling autotuning provided stable paired
streams; that does not prove autotuning was the cause of the earlier divergence.
The integrated provider also passed 224 FP16/BF16 NumPy-reference checks over
grouped decode and scalar/tensor async prefill. These remain opt-in experiments,
not a guarantee for untested models, contexts, or tuning configurations.
