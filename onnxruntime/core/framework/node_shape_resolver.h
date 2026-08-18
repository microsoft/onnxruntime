// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>

#include "core/common/inlined_containers.h"
#include "core/framework/max_shape_override.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {

/// Resolves the input shapes for a node using a max-shape shadow-inference result.
/// For each input:
///   - Use the concrete shape propagated through the shadow graph when available.
///   - Otherwise use a fully static shape from the executable graph.
///   - If any input remains dynamic, return nullopt.
///
/// @param node             The node whose input shapes to resolve.
/// @param graph_identity   Address of the node's containing graph.
/// @param inferred_shapes  Shapes propagated from maximum graph inputs.
/// @return                 Vector of shapes for each input, or nullopt if any input has unresolvable dims.
template <typename TNode>
inline std::optional<InlinedVector<TensorShape>> ResolveNodeInputShapes(
    const TNode& node,
    const void* graph_identity,
    const MaxShapeInferenceResult& inferred_shapes) {
  InlinedVector<TensorShape> result;
  result.reserve(node.InputDefs().size());

  for (const auto* input_def : node.InputDefs()) {
    if (!input_def || !input_def->Exists()) {
      return std::nullopt;  // Cannot resolve shape for this input
    }

    if (const TensorShape* inferred_shape = inferred_shapes.GetShape(graph_identity, input_def->Name())) {
      result.push_back(*inferred_shape);
      continue;
    }

    const auto* shape_proto = input_def->Shape();
    if (!shape_proto) return std::nullopt;

    TensorShapeVector dims;
    dims.reserve(shape_proto->dim_size());

    for (const auto& dim : shape_proto->dim()) {
      if (dim.has_dim_value() && dim.dim_value() >= 0) {
        dims.push_back(dim.dim_value());
      } else {
        return std::nullopt;
      }
    }

    result.emplace_back(dims);
  }

  return result;
}

}  // namespace onnxruntime
