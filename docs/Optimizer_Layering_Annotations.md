# Optimizer Layering Annotations

## Overview

Layering annotations are per-node metadata strings that guide graph partitioning by indicating which execution provider (EP) layer a node belongs to. They are loaded from the ONNX model's `NodeProto` metadata (key `"layer_ann"`) and consumed during the partitioning phase to influence EP assignment.

## Execution Pipeline

Graph optimizers run in ordered levels:

```
Level 0 (Basic) ─► Level 1 (Extended) ─► Partitioning ─► Level 2+ (Layout, etc.)
```

1. **Level 0 and Level 1** optimizers run **before** partitioning. At this point, layering annotations are present on nodes and must be preserved through any graph transformations.
2. **Partitioning** reads the annotations to assign nodes to execution providers.
3. After partitioning, `Graph::RemoveAllLayeringAnnotations()` clears all annotations.
4. **Level 2, 3, and 4** optimizers run **after** annotations have been cleared. They do not need to handle annotations.

**Key rule: Only Level 1 (and Level 0) optimizers need to propagate layering annotations.**

## Why Propagation Matters

When an optimizer replaces, fuses, or decomposes nodes, the original annotated node is removed and new nodes are created. If the new nodes do not carry the original annotation, partitioning loses the assignment hint for that subgraph, potentially causing incorrect EP placement.

## How to Propagate Annotations

### Preferred: Use the `AddNode` Overload with `annotation_source`

`Graph::AddNode` provides overloads that accept a `const Node& annotation_source` parameter. The new node automatically inherits the layering annotation from the source node.

```cpp
// Instead of:
Node& new_node = graph.AddNode(name, op_type, description, inputs, outputs);
// Missing annotation propagation!

// Use:
Node& new_node = graph.AddNode(name, op_type, description, inputs, outputs,
                               original_node);  // annotation_source
```

All standard `AddNode` signatures have a corresponding `annotation_source` variant:

```cpp
// With const NodeAttributes*
Node& AddNode(name, op_type, description,
              gsl::span<NodeArg* const> inputs,
              gsl::span<NodeArg* const> outputs,
              const Node& annotation_source,
              const NodeAttributes* attributes = nullptr,
              const std::string& domain = kOnnxDomain);

// With NodeAttributes&&
Node& AddNode(name, op_type, description,
              gsl::span<NodeArg* const> inputs,
              gsl::span<NodeArg* const> outputs,
              const Node& annotation_source,
              NodeAttributes&& attributes,
              const std::string& domain = kOnnxDomain);

// initializer_list variants also available
```

### Legacy: `DuplicateNodeAnnotation`

The utility function `optimizer_utils::DuplicateNodeAnnotation(src, dst)` copies annotations between existing nodes. This is still used when the annotation source is conditional (e.g., when the source node pointer may be null). Prefer the `AddNode` overload for unconditional propagation.

### Automatic Propagation

`Graph::AddNode(const Node& other)` — the copy overload used for duplicating nodes — automatically copies annotations. No additional action is needed when duplicating a node via this overload.

## Post-Partitioning: Propagating EP Assignments

Although Level 2+ optimizers do not deal with layering annotations directly (they have been cleared), they must still propagate **execution provider (EP) assignments**. EP assignments are the downstream result of the annotation-driven partitioning step. After partitioning, each node carries an EP assignment (e.g., `CUDAExecutionProvider`, `CPUExecutionProvider`) that determines where the node's kernel runs.

When a Level 2+ optimizer creates new nodes that replace or derive from existing ones, it must copy the EP assignment from the source node:

```cpp
Node& new_node = graph.AddNode(name, op_type, description, inputs, outputs);
new_node.SetExecutionProviderType(original_node.GetExecutionProviderType());
```

Failing to propagate the EP assignment causes the new node to fall back to the default provider (typically CPU), silently breaking the intended placement and potentially degrading performance or correctness. This requirement predates the layering annotation feature and applies to all optimizers that run after partitioning.

> **Note:** The `AddNode` overload with `annotation_source` propagates both the layering annotation *and* nothing else — EP assignment is still set separately. Layering annotations and EP assignments serve different stages of the pipeline and are managed independently.

## When You Do NOT Need to Propagate Annotations

- **Level 2+ optimizers** — annotations have already been consumed and cleared (but EP assignments must still be propagated, see above).
- **Training optimizers** — training runs after partitioning.
- **Optimizers that only remove nodes** (e.g., identity elimination) — no new nodes are created.
- **Optimizers that modify nodes in-place** — the annotation remains on the existing node.

## Examples

### Fusion (replacing multiple nodes with one)

```cpp
// GeluFusion: fusing Div + Erf + Add + Mul + Mul into a single Gelu
Node& gelu_node = graph.AddNode(
    graph.GenerateNodeName("Gelu"),
    "Gelu", "fused Gelu subgraphs",
    {gelu_input}, {gelu_output},
    div_node);  // propagate annotation from the root matched node
```

### Decomposition (replacing one node with many)

```cpp
// STFT decomposition: each new node inherits from the original STFT node
auto [reshape_node, reshape_out] = AddNode(graph, "Reshape", ep, inputs, &stft);
auto [conv_node, conv_out]       = AddNode(graph, "Conv", ep, conv_inputs, &stft);
auto [concat_node, concat_out]   = AddNode(graph, "Concat", ep, concat_inputs, &stft);
```

### Conditional source (use DuplicateNodeAnnotation)

```cpp
Node& q_node = graph.AddNode(...);
if (src_node) {
    optimizer_utils::DuplicateNodeAnnotation(*src_node, q_node);
}
```

## Checklist for New Level 1 Optimizers

1. Identify the "source" node whose annotation should propagate (typically the root of the matched pattern).
2. For every `graph.AddNode(...)` call that creates a replacement node, use the `annotation_source` overload.
3. If the source is conditional (may be null), use `optimizer_utils::DuplicateNodeAnnotation` after the `AddNode` call.
4. Test with an annotated model to verify annotations survive the transformation.
