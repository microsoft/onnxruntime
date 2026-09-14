# Playbook 07: Graph Optimizations and Fusion Passes

## Outcome

By the end of this playbook, you will be able to add or modify a graph optimization pass, reason about where it runs in the optimizer pipeline, and validate correctness with targeted transformer tests.

This playbook assumes you have already completed [Playbook 04](04-session-lifecycle-from-load-to-run.md) and [Playbook 05](05-adding-or-changing-a-kernel.md).

## Start Here

- [onnxruntime/core/optimizer/graph_transformer.cc](../../onnxruntime/core/optimizer/graph_transformer.cc)
- [onnxruntime/core/optimizer/rule_based_graph_transformer.cc](../../onnxruntime/core/optimizer/rule_based_graph_transformer.cc)
- [onnxruntime/core/optimizer/conv_add_fusion.cc](../../onnxruntime/core/optimizer/conv_add_fusion.cc)
- [onnxruntime/core/optimizer/gather_fusion.cc](../../onnxruntime/core/optimizer/gather_fusion.cc)
- [onnxruntime/test/optimizer/graph_transform_test.cc](../../onnxruntime/test/optimizer/graph_transform_test.cc)

## Mental Model

Graph optimization in ONNX Runtime is a sequence of transformer passes that rewrite the graph before execution.

There are two common pass styles:

- rule-based passes that check a local pattern and rewrite one region
- graph-wide passes that scan topology and perform broader rewrites

Every pass should be conservative. If constraints are not proven, do not rewrite.

## Where Optimizations Run

The core execution contract is in [onnxruntime/core/optimizer/graph_transformer.cc](../../onnxruntime/core/optimizer/graph_transformer.cc):

- `GraphTransformer::Apply(...)` calls `ApplyImpl(...)`
- if the graph was modified, `graph.Resolve()` is called to restore graph validity for subsequent passes

This means your pass can assume a valid graph at entry and must leave a graph that can be resolved after modification.

Rule dispatch behavior is in [onnxruntime/core/optimizer/rule_based_graph_transformer.cc](../../onnxruntime/core/optimizer/rule_based_graph_transformer.cc):

- rules are registered per op type or as any-op rules
- rules run in topological order
- if a rule removes the current node, rule application for that node stops
- subgraphs are processed recursively

## Concrete Fusion Pattern: Conv + Add

Use [onnxruntime/core/optimizer/conv_add_fusion.cc](../../onnxruntime/core/optimizer/conv_add_fusion.cc) as a template for local pattern fusion.

The pass structure is:

1. `SatisfyCondition(...)` guards the rewrite using strict checks.
2. `Apply(...)` constructs replacement initializer state and rewires nodes.
3. `FinalizeNodeFusion(...)` transfers edges and removes the old node.

Important guard ideas in this pass:

- op type, opset, and domain checks
- constant initializer requirements
- execution provider compatibility checks
- output and edge shape expectations
- graph output safety checks

Use this pattern when your fusion replaces a short operator chain with a semantically equivalent compact form.

## Concrete Graph-Wide Pattern: Gather or Slice to Split

Use [onnxruntime/core/optimizer/gather_fusion.cc](../../onnxruntime/core/optimizer/gather_fusion.cc) as a template for topology-scanning rewrites.

The pass demonstrates:

- opset gating to avoid schema mismatches
- helper functions for axis and initializer decoding
- coverage checks to ensure rewrites are complete and non-overlapping
- ordered output reconstruction and node replacement

This style is appropriate when a rewrite depends on relationships among multiple consumer nodes and not just one immediate neighbor.

## Step-by-Step Workflow for a New Optimization

1. Pick one existing pass that is structurally similar to your intended rewrite.
2. Write exact rewrite preconditions first.
3. Implement the rewrite with explicit graph utility calls.
4. Preserve execution-provider boundaries unless cross-provider rewrite is explicitly safe.
5. Validate graph outputs and node ownership before removing nodes.
6. Add tests that prove both positive and negative behavior.

If preconditions are complex, encode them in helper functions so the rewrite body stays readable.

## Test Strategy

Primary test harness references:

- [onnxruntime/test/optimizer/graph_transform_test.cc](../../onnxruntime/test/optimizer/graph_transform_test.cc)
- [onnxruntime/test/optimizer/rule_based_graph_transformer_test.cc](../../onnxruntime/test/optimizer/rule_based_graph_transformer_test.cc)

Common test pattern in `graph_transform_test.cc`:

- load a small model
- count key ops before transformation
- register transformer in `GraphTransformerManager`
- apply at a target transformer level
- assert post-transform op counts and structure

For each new optimization, include:

- one positive test where rewrite must occur
- one negative test where rewrite must not occur
- one safety test for an edge case (graph output, provider mismatch, unsupported shape, or unsupported opset)

## Fast Local Validation Loop

Run targeted optimizer tests first from the build output directory:

Linux:

```bash
./onnxruntime_test_all --gtest_filter="*GraphTransformationTests*:*RuleBasedGraphTransformer*"
```

Windows:

```powershell
.\onnxruntime_test_all.exe --gtest_filter="*GraphTransformationTests*:*RuleBasedGraphTransformer*"
```

Then narrow further to your specific test names when iterating.

## Design Rules for Safe Rewrites

- do not rewrite across execution-provider boundaries unless explicitly safe
- require constants and shapes only when they are proven
- avoid introducing non-deterministic rewrite behavior
- avoid relying on exporter quirks unless guarded by explicit checks
- keep rewrite semantics identical to the original subgraph

When in doubt, skip the optimization and preserve correctness.

## Common Failure Modes

- over-permissive conditions that rewrite unsupported patterns
- removing nodes that still feed graph outputs or surviving nodes
- missing `graph.Resolve()` expectations after modifications
- assuming one opset behavior while test models use another
- missing negative tests that prevent future regressions

## Exit Checklist

- [ ] The optimization has explicit and conservative precondition checks.
- [ ] Rewrite logic preserves graph semantics and provider constraints.
- [ ] Positive and negative tests exist in optimizer test suites.
- [ ] You validated with targeted optimizer test filters locally.