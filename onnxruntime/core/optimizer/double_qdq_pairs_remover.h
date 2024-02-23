// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * @Class DoubleQDQPairsRemover
 * @brief Remove one pair of Q-DQ from Double Q-DQ pairs.
 * Specifically, this transformer converts the sequence Q1 -> DQ1 -> Q2 -> DQ2, where the first pair has (zp1, scale1)
 * and the second pair has (zp2, scale2), into the sequence Q1 -> DQ2 by removing the middle two nodes. The zero-point
 * and scale of the final QDQ pair is recomputed to preserve equality to the original sequence.
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
};
}  // namespace onnxruntime
