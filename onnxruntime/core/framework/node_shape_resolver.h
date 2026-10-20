// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/framework/max_shape_override.h"
#include "core/framework/workspace_input_shape.h"

namespace onnxruntime {

/// Resolves the input shapes for a node using a max-shape shadow-inference result.
/// For each explicit input:
///   - Preserve its position and mark an omitted optional input as Missing.
///   - Use the concrete shape propagated through the shadow graph when available.
///   - Otherwise preserve the rank and dimensions from the executable graph's shape proto,
///     representing each symbolic, absent, or negative dimension as -1. Multiple -1 dimensions
///     are independent unknowns and do not imply symbolic equality.
///   - Mark a present input without rank or dimension metadata as PresentWithoutShape.
/// Shapes propagated from max-shape overrides are estimation hints, not proven upper bounds.
///
/// @param node             The node whose input shapes to resolve.
/// @param graph_identity   Address of the node's containing graph.
/// @param inferred_shapes  Shapes propagated from maximum graph inputs.
/// @return                 One presence-aware shape entry per Node::InputDefs() element. Implicit
///                         inputs are excluded, matching OpKernelContext::Input(i).
template <typename TNode>
inline InlinedVector<WorkspaceInputShape> ResolveNodeInputShapes(
    const TNode& node,
    const void* graph_identity,
    const MaxShapeInferenceResult& inferred_shapes) {
  InlinedVector<WorkspaceInputShape> result;
  result.reserve(node.InputDefs().size());

  for (const auto* input_def : node.InputDefs()) {
    if (!input_def || !input_def->Exists()) {
      result.emplace_back();
      continue;
    }

    if (const TensorShape* inferred_shape = inferred_shapes.GetShape(graph_identity, input_def->Name())) {
      result.push_back(WorkspaceInputShape::PresentWithShape(*inferred_shape));
      continue;
    }

    const auto* shape_proto = input_def->Shape();
    if (!shape_proto) {
      result.push_back(WorkspaceInputShape::PresentWithoutShape());
      continue;
    }

    TensorShapeVector dims;
    dims.reserve(shape_proto->dim_size());

    for (const auto& dim : shape_proto->dim()) {
      dims.push_back(dim.has_dim_value() && dim.dim_value() >= 0 ? dim.dim_value() : -1);
    }

    result.push_back(WorkspaceInputShape::PresentWithShape(TensorShape{dims}));
  }

  return result;
}

}  // namespace onnxruntime
