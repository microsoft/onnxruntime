# Graph transformations

**Owns:** WebGPU-related fusion, layout rewrites, partitioning, and optimized
graph semantics. Kernel registrations and arithmetic have separate owners.
**Update when:** Optimizer eligibility, fused schemas, partition support,
layout conversion, or model/export patterns change.

- Trace original graph, selector, replacement graph, provider assignment, and
  executed kernel. A preferred NHWC layout does not prove NHWC execution.
  Inspect optimized node domain/opset/type/layout and partition boundaries;
  added casts/transposes/readbacks can outweigh a local shader improvement.
- Preserve optional inputs/outputs, graph outputs, additional consumers,
  subgraphs, symbolic shapes, initializer versus overridable inputs, and
  attribute presence/defaults. Reject a fusion when the replacement cannot
  represent source semantics, including mixed dtypes or distinct Q/K epsilons.
- A node may already contain an activation or other fused behavior. Composing
  another activation must preserve both or decline fusion. Test chains of
  activations, explicit zero parameters, optional bias slots, and parameters
  independently of operator names.
- Empty LoRA/adapter branches are legal model patterns. Trace zero-sized
  weights, dynamic/overridable inputs, and fallback eligibility through the
  entire rewrite rather than assuming ordinary dense-export shapes.
- Compare optimized and unfused/control execution with identical exported
  arithmetic, I/O, dtype, and provider options. Check auxiliary/cache outputs
  and downstream discrete decisions as applicable. Test accepted and rejected
  patterns and prove the expected graph changed; a passing model on CPU or an
  unfused graph does not validate the fusion.
- Changes in shared optimizers need other-provider and minimal-build review.
  Keep native/plugin support predicates aligned via [plugin contracts](plugin-contracts.md).

Source landmarks: `core/optimizer/graph_transformer_utils.cc`,
`conv_activation_fusion.cc`, `group_query_attention_pre_norm_fusion.cc`,
`matmul_nbits_*fusion*`, native/plugin capability discovery, and
`test/optimizer/` tests. Regression examples include
[#33363](https://github.com/microsoft/onnxruntime/pull/33363) for normalization
fusion and [#32929](https://github.com/microsoft/onnxruntime/pull/32929) for
WebGPU Conv activation paths. Recheck exact schemas and guards in the PR head.
