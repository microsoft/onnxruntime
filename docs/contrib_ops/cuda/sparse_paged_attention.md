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

## CUDA dispatch

Eligible `selected_only`/`main` calls with matching activation/cache types, head
size 256, and query-to-KV head ratios divisible by four use tiled grouped-query
attention. Other geometries and quantized caches retain the general fallback.
All paths preserve suppressed-write prefix-cache reads; INT8 current rows are
read back from the quantized cache.

The older six-head FP16 grouped-prefill path remains opt-in through session
configuration `ep.cuda.sparse_paged_attention_grouped=1`, or environment variable
`ORT_SPARSE_PAGED_ATTENTION_GROUPED=1` when no session override is present.
`ep.cuda.sparse_paged_attention_grouped_vectorized=1` selects vectorized loads
within that path. Both session options accept only `0` or `1`.

`ORT_SPARSE_PREFILL_TENSOR_CORE_QK=1` enables experimental tensor-core query/key
scoring for eligible twelve-head prefill tiles. With that option enabled,
`ORT_SPARSE_PREFILL_TENSOR_CORE_PV=1` also permits the FP16 probability/value
path. These environment options default to off and are cached on first use.
Neither option establishes broad model-quality or throughput equivalence.

The CUDA Python regression `test_sparse_paged_attention_prefill_cuda.py` runs
scalar, QK-only, QK+PV, BF16 QK, and both grouped-load configurations in fresh
processes. Each compares to an independent reference with multiple/partial
tiles, invalid selections, suppressed-write prefix hits, and attention sinks.
Verbose dispatch messages are asserted so an unintended fallback cannot pass.
Run it against a build containing this change on SM80 or newer:

```bash
python -m pytest onnxruntime/test/python/transformers/test_sparse_paged_attention_prefill_cuda.py
```

Selector fusion, paged auxiliary storage, and auxiliary quantization remain
outside this operator's scope.
