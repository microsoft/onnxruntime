# SparseAttentionIndexer on WebGPU

The WebGPU execution provider implements version 1 of
`com.microsoft.SparseAttentionIndexer` for the `qsa` and `csa` policies. It uses
the provider-neutral schema and state ABI described in the
[operator documentation](../cuda/sparse_attention_indexer.md).

## Supported subset

- batched inputs;
- `qsa` and `csa` policy modes;
- `float32` and `float16`;
- explicit graph-visible key, compressed-key, and incomplete-window state;
- arbitrary boolean QSA visibility masks;
- deterministic score-descending, index-ascending TopK ties.

BF16 and packed/variable-length inputs are not registered by the WebGPU kernel.
Unknown policies and policy-incompatible inputs or attributes are rejected.

## Execution

State concatenation, visible-token grouping, QSA pooling, CSA overlap
compression, RMS normalization, rotary embedding, scoring, selection, and
output padding execute in WGSL. The implementation does not map GPU buffers,
read selected values back to the host, or retain state in the kernel object.
All reductions and softmax calculations accumulate in FP32, including for
FP16 inputs.

The initial implementation prioritizes correctness and uses one independently
writable workgroup per query or completed CSA window. Candidate scoring during
selection is recomputed rather than materialized, avoiding candidate-count
limits and GPU-to-CPU synchronization at the cost of additional computation.

## Follow-up work

- packed/variable-length input;
- specialized large-candidate TopK;
- subgroup-optimized reductions;
- fused projection, pooling, and scoring;
- reduced recomputation and temporary-buffer use;
- selector/executor fusion;
- WebGPU `DynamicSparseAttention` and `SparsePagedAttention`;
- additional element types.
