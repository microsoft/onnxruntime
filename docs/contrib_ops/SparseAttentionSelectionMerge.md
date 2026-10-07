# SparseAttentionSelectionMerge

`com.microsoft::SparseAttentionSelectionMerge`, version 1, forms a stable union
of a mapped base selection and an integer interval. CUDA and WebGPU kernels
support `policy_mode="append_range"`. The operator owns no persistent state.
It performs no attention, scoring, sorting, or TopK computation.

All tensors use `int32`. Required inputs are `base_indices [R,C]`,
`base_counts [R]`, `base_row_indices [N]`, `range_starts [N]`, and
`range_ends [N]`. Slots 5 and 6 are reserved and must be absent. The required
`max_output_entries` attribute is positive and at most `2^30`.

For query `i`, preserve the first occurrence of each valid base entry from
row `base_row_indices[i]`, then append unseen values from
`[range_starts[i],range_ends[i])` in ascending order. Padding outside the valid
base prefix is ignored. Outputs are `selected_indices [N,max_output_entries]`,
`selected_counts [N]`, and `status [N]`. Unused output columns are `-1`.

Status is `0` on success, `1` for invalid row/count/range metadata or a negative
valid-prefix index, and `2` when the deduplicated union exceeds capacity. On
failure, the entire row is `-1` and its count is zero. Shape, type, and attribute
errors are ordinary session/operator validation errors. `append_indices` is
not implemented and is rejected.

Callers must supply one index namespace, ensure causal visibility, and check
every status before publishing dependent results. A status output does not
prevent downstream nodes from executing in the same graph.

CUDA uses one cooperative block per query with invocation-local scratch for
stable duplicate removal. WebGPU uses one invocation per query, with a
workgroup hash table for capacities up to 4096 and an output-prefix scan for
larger capacities. Neither kernel modifies the base tensors or retains a
selection between runs.

Regression tests: `SparseAttentionSelectionMerge.*` in
`onnxruntime_provider_test` (or `onnxruntime_test_all` on branches predating
the provider-test split).