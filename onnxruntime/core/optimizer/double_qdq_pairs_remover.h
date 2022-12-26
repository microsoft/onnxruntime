// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {
  /**
   * @Class DoubleQDQPairsRemover
   * @brief Remove one pair of Q-DQ Double Q-DQ pairs.
   */
class DoubleQDQPairsRemover : public GraphTransformer {
public:
  DoubleQDQPairsRemover() : GraphTransformer("DoubleQDQPairsRemover", {}) {}

private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;
  bool IsPairDQQRemovable(const Node& dq_node, const Node& q_node) const;
  std::vector<const Node*> GetQualifiedQChildren(const Node& dq_node) const;
};
} // namespace onnxruntime
