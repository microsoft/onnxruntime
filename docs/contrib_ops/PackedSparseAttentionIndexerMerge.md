# PackedSparseAttentionIndexerMerge

`com.microsoft::PackedSparseAttentionIndexerMerge`, version 1, forms a stable union
of a mapped base selection and additional indices. CUDA and WebGPU kernels
support `policy_mode="append_range"`; CUDA also supports `append_indices`.
The operator owns no persistent state.
It performs no attention, scoring, sorting, or TopK computation.

All tensors use `int32`. Common inputs are `base_indices [R,C]`,
`base_counts [R]`, and `base_row_indices [N]`. For `append_range`, slots 3 and 4
are `range_starts [N]` and `range_ends [N]`; slots 5 and 6 must be absent.
For `append_indices`, slots 3 and 4 must be absent, and slots 5 and 6 are
`additional_indices [N,A]` and `additional_counts [N]`. The required
`max_output_entries` attribute is positive and at most `2^30`.

For query `i`, preserve the first occurrence of each valid base entry from
row `base_row_indices[i]`, then append unseen values from
`[range_starts[i],range_ends[i])` in ascending order. Padding outside the valid
base prefix is ignored. For `append_indices`, append unseen values from the
additional valid prefix in their original order.
Outputs are `selected_indices [N,max_output_entries]`,
`selected_counts [N]`, and `status [N]`. Unused output columns are `-1`.

Status is `0` on success, `1` for invalid row/count/range metadata or a negative
valid-prefix index, and `2` when the deduplicated union exceeds capacity. On
failure, the entire row is `-1` and its count is zero. Shape, type, and attribute
errors are ordinary session/operator validation errors. WebGPU rejects
`append_indices`.

Callers must supply one index namespace, ensure causal visibility, and check
every status before publishing dependent results. A status output does not
prevent downstream nodes from executing in the same graph.

CUDA uses one cooperative block per query with invocation-local scratch for
stable duplicate removal. WebGPU uses one invocation per query, with a
workgroup hash table for capacities up to 4096 and an output-prefix scan for
larger capacities. Neither kernel modifies the base tensors or retains a
selection between runs.

Regression tests: `PackedSparseAttentionIndexerMerge.*` in
`onnxruntime_provider_test`.