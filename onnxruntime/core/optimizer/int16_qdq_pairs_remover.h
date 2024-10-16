// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
 * @Class Int16QDQPairsRemover
 * @brief Remove pairs of int16 or uint16 Q-DQ ops that cancel eachother.
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
