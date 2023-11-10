// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/qdq_transformer/qdq_util.h"

namespace onnxruntime {

using ONNX_NAMESPACE::TensorProto;
using ONNX_NAMESPACE::TensorProto_DataType;
using QDQ::InputIndex;

/**
 * @Class DoubleQDQPairsRemover
 * @brief Remove one pair of Q-DQ from Double Q-DQ pairs.
 */
class DoubleQDQPairsRemover : public GraphTransformer {
 public:
  DoubleQDQPairsRemover() : GraphTransformer("DoubleQDQPairsRemover", {}) {}

 private:
  Status ApplyImpl(
      Graph& graph,
      bool& modified,
      int graph_level,
      const logging::Logger& logger) const override;

  static bool IsNodeRemovable(
      Graph& graph,
      const NodeIndex& self_index,
      NodeIndex& parent_index,
      NodeIndex& child_index,
      NodeIndex& grandchild_index);

  template <typename T>
  static bool FindNewZeroPointAndScale(
      const Graph& graph,
      const Node& node1,
      const Node& node2,
      float& new_scale,
      T& new_zero_point);

  template <typename T>
  static void ApplyNewInputValue(
      Graph& graph,
      Node& node,
      const InputIndex& index,
      T value);
};
}  // namespace onnxruntime
