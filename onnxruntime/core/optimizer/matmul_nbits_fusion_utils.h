// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <cstdint>

#include "core/common/common.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"

// Small helpers shared by the WebGPU MatMulNBits MLP and QKV graph fusion
// transformers. Both fusions need the same node/attribute predicates over
// MatMulNBits and [Skip]SimplifiedLayerNormalization, so they are kept in one
// place to avoid divergence.
namespace onnxruntime {
namespace matmul_nbits_fusion_utils {

// Returns true if the node has an input defined at \p index (present and non-empty name).
inline bool HasInput(const Node& node, size_t index) {
  return index < node.InputDefs().size() &&
         node.InputDefs()[index] != nullptr &&
         !node.InputDefs()[index]->Name().empty();
}

// Returns true if the node has produced an output at \p index (present and non-empty name).
inline bool HasProducedOutput(const Node& node, size_t index) {
  return index < node.OutputDefs().size() &&
         node.OutputDefs()[index] != nullptr &&
         !node.OutputDefs()[index]->Name().empty();
}

inline bool IsSupportedSimplifiedLayerNormalization(const Node& node) {
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "SimplifiedLayerNormalization", {1})) {
    return false;
  }

  // Fusion assumes the default normalization axis (-1).
  const auto* axis_attr = graph_utils::GetNodeAttribute(node, "axis");
  return axis_attr == nullptr || axis_attr->i() == -1;
}

inline bool IsSupportedSkipSimplifiedLayerNormalization(const Node& node) {
  // Must be the correct version and domain.
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "SkipSimplifiedLayerNormalization", {1}, kMSDomain)) {
    return false;
  }
  // We only support the default form: inputs [x, skip, scale] without optional bias,
  // and axis attribute must be default (-1) if present.
  // Bias input at index 3 would be silently dropped by the fused kernel.
  if (HasInput(node, 3)) {
    return false;  // Has optional bias input; fusion would drop it.
  }
  // Check for non-default axis attribute.
  const auto* axis_attr = graph_utils::GetNodeAttribute(node, "axis");
  if (axis_attr != nullptr && axis_attr->i() != -1) {
    return false;  // Non-default axis; fusion assumes axis=-1.
  }
  return true;
}

// Reads an integer attribute. If absent, returns \p default_value (or fails when \p required is true).
inline int64_t GetIntAttr(const Node& node, const char* name, int64_t default_value, bool required = false) {
  const auto* attr = graph_utils::GetNodeAttribute(node, name);
  if (attr == nullptr) {
    ORT_ENFORCE(!required, "Missing required attribute ", name, " on node ", node.Name());
    return default_value;
  }
  return attr->i();
}

// Reads a float attribute. If absent, returns \p default_value.
inline float GetFloatAttr(const Node& node, const char* name, float default_value) {
  const auto* attr = graph_utils::GetNodeAttribute(node, name);
  return attr == nullptr ? default_value : attr->f();
}

}  // namespace matmul_nbits_fusion_utils
}  // namespace onnxruntime
