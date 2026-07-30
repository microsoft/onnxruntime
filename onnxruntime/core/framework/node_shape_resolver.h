// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>

#include "core/common/inlined_containers.h"
#include "core/framework/resource_accountant.h"
#include "core/framework/tensor_shape.h"
#include "core/graph/graph_viewer.h"

namespace onnxruntime {

/// Resolves the input shapes for a node using graph shape inference results and max shape overrides.
/// For each input:
///   - If the graph has a fully static shape, use it directly.
///   - If a dimension is symbolic/dynamic (-1), substitute from max_shape_overrides if available.
///   - If any dimension remains unknown after override, returns nullopt (cannot estimate).
///
/// @param node             The node whose input shapes to resolve.
/// @param graph            The graph containing the node (for NodeArg shape access).
/// @param shape_overrides  Max shape overrides from session option (may be empty).
/// @return                 Vector of shapes for each input, or nullopt if any input has unresolvable dims.
inline std::optional<InlinedVector<TensorShape>> ResolveNodeInputShapes(
    const Node& node,
    const GraphViewer& graph,
    const MaxShapeOverrideMap& shape_overrides) {
  ORT_UNUSED_PARAMETER(graph);

  InlinedVector<TensorShape> result;
  result.reserve(node.InputDefs().size());

  for (const auto* input_def : node.InputDefs()) {
    if (!input_def || !input_def->Exists() || !input_def->HasTensorOrScalarShape()) {
      return std::nullopt;  // Cannot resolve shape for this input
    }

    const auto* shape_proto = input_def->Shape();
    if (!shape_proto) {
      return std::nullopt;
    }

    const auto& name = input_def->Name();
    const MaxShapeOverrideMap::const_iterator override_it = shape_overrides.find(name);

    TensorShapeVector dims;
    dims.reserve(shape_proto->dim_size());

    for (int i = 0; i < shape_proto->dim_size(); ++i) {
      const auto& dim = shape_proto->dim(i);
      if (dim.has_dim_value() && dim.dim_value() > 0) {
        dims.push_back(dim.dim_value());
      } else {
        // Symbolic or unknown dimension — try override
        if (override_it != shape_overrides.end()) {
          const auto& override_shape = override_it->second;
          if (static_cast<int>(override_shape.NumDimensions()) > i) {
            int64_t override_dim = override_shape[i];
            if (override_dim > 0) {
              dims.push_back(override_dim);
              continue;
            }
          }
        }
        // No override available for this dimension
        return std::nullopt;
      }
    }

    result.emplace_back(dims);
  }

  return result;
}

}  // namespace onnxruntime
