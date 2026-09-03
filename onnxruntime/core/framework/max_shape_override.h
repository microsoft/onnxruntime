// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <optional>
#include <string>
#include <string_view>

#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {

/// Input shape overrides supplied by the user for estimation.
using MaxShapeOverrideMap = InlinedHashMap<std::string, TensorShape>;

/// Fully concrete shapes inferred in a disposable shadow graph for workspace estimation.
/// Shapes are grouped by source Graph identity so equal NodeArg names in nested graphs
/// remain distinct. The executable graph is never modified. Inferred downstream shapes
/// are estimation hints and are not guaranteed upper bounds because operator shape
/// transformations are not necessarily monotonic.
class MaxShapeInferenceResult {
 public:
  /// graph_identity must be the address of the source Graph passed to InferMaxShapes.
  const TensorShape* GetShape(const void* graph_identity, const std::string& node_arg_name) const {
    const auto graph_it = graph_shapes_.find(graph_identity);
    if (graph_it == graph_shapes_.end()) return nullptr;
    const auto shape_it = graph_it->second.find(node_arg_name);
    return shape_it == graph_it->second.end() ? nullptr : &shape_it->second;
  }

  bool Empty() const noexcept { return graph_shapes_.empty(); }

 private:
  friend class MaxShapeInferenceBuilder;
  using ShapeMap = InlinedHashMap<std::string, TensorShape>;
  InlinedHashMap<const void*, ShapeMap> graph_shapes_;
};

/// Parses the session.max_shape_override config string into a name -> shape map.
///
/// Format: "name1:[d0,d1,...];name2:[d0,d1,...]"
/// Example: "input_ids:[8,4096];attention_mask:[8,4096]"
///
/// Rules:
/// - Names and dimensions are separated by semicolons.
/// - Each entry is "name:[dim0,dim1,...]".
/// - Dimensions must be positive integers.
/// - Whitespace around names, brackets, and commas is tolerated.
/// - Empty string returns an empty map (success).
///
/// @param config_value  The raw string from the session option.
/// @param out           Populated on success.
/// @return              OK on success, INVALID_ARGUMENT on parse failure.
Status ParseMaxShapeOverride(std::string_view config_value, MaxShapeOverrideMap& out);

}  // namespace onnxruntime
