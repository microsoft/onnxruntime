// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * @Class Int16QDQPairsRemover
 * @brief Remove pairs of int16 or uint16 Q-DQ ops.
 * Specifically, this transformer converts the sequence Q1 -> DQ1 -> Q2 -> DQ2, where the first pair has (zp1, scale1)
 * and the second pair has (zp2, scale2), into the sequence Q1 -> DQ2 by removing the middle two nodes. The zero-point
 * and scale of the final QDQ pair is recomputed to preserve equality to the original sequence.
 *
 * Also supports multiple identical DQ2 nodes, which may have been inserted by the EnsureUniqueDQNodeUnit optimizer.
 * Q1 --> DQ1 --> Q2 --+--> DQ2
 *                     |
 *                     +--> DQ2'
 *
 * The above becomes:
 * Q1 ---+--> DQ2
 *       |
 *       +--> DQ2'
 */
class Int16QDQPairsRemover : public GraphTransformer {
 public:
  Int16QDQPairsRemover(const InlinedHashSet<std::string_view>& compatible_execution_providers = {})
      : GraphTransformer("Int16QDQPairsRemover", compatible_execution_providers) {}

 private:
  Status ApplyImpl(
      Graph& graph,
      bool& modified,
      int graph_level,
      const logging::Logger& logger) const override;
};
}  // namespace onnxruntime
