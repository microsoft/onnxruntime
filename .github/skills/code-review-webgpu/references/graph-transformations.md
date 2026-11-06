# Graph transformations

**Owns:** WebGPU fusion, layout rewrites, partitioning, and optimized-graph semantics.
**Update when:** Eligibility, fused schemas, partition/layout support, or export
patterns change. Registration and arithmetic remain in their owning references.

- Trace original graph → selector → replacement → provider → executed kernel.
  Inspect optimized domain/opset/type/layout and partitions; preferred NHWC does
  not prove execution. Count added casts/transposes/readbacks in performance claims.
- Preserve optional inputs/outputs, graph outputs, extra consumers, subgraphs,
  symbolic shapes, initializer/overridable inputs, and attribute presence/defaults.
  Reject unrepresentable semantics, including mixed dtypes or distinct Q/K epsilons.
- Preserve already-fused activation/behavior when composing another, or reject
  fusion. Test activation chains, explicit-zero parameters, optional bias slots,
  and parameter semantics independently of operator names.
- Trace legal empty LoRA/adapter branches, zero-sized weights, dynamic/overridable
  inputs, and fallbacks through the whole rewrite; dense-export assumptions fail.
- Compare optimized and unfused controls with identical exported arithmetic,
  I/O, dtype, and options, including auxiliary/cache outputs and discrete
  decisions. Test accepted/rejected patterns and prove the expected graph rewrite
  and WebGPU execution; CPU or unfused success is insufficient.
- Review other providers/minimal builds for shared optimizer edits. Align
  native/plugin predicates through [plugin contracts](runtime.md#plugin-contracts).

Inspect `core/optimizer/graph_transformer_utils.cc`, `conv_activation_fusion.cc`,
`group_query_attention_pre_norm_fusion.cc`, `matmul_nbits_*fusion*`, capability
discovery, and `test/optimizer/`; recheck schemas/guards at the reviewed head.
