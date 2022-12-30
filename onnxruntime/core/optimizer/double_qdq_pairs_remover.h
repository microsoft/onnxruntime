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
  Status ApplyImpl(
      Graph &graph,
      bool &modified,
      int graph_level,
      const logging::Logger &logger
                  ) const override;

  static bool IsNodeRemovable(
      const Graph &graph,
      const logging::Logger &logger,
      const NodeIndex &node_index,
      NodeIndex &parent_index,
      NodeIndex &child_index,
      NodeIndex &grandchild_index
                             );

};
} // namespace onnxruntime
