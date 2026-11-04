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
softmax. Selector fusion, paged auxiliary storage, auxiliary quantization, and
decode-specialized sparse kernels are deferred.
