// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class SumAccumulateTransformer

A ----\
B ---- (Sum) ------- (InPlaceAccumulator) -- Buffer
C ----/             /
           Buffer--/

into

Buffer ---\
A -------- (InPlaceAccumulator) ---\
Buffer ---\                         \
B -------- (InPlaceAccumulator) ---- (Deduplicate) --- Buffer
Buffer ---\                         /
C -------- (InPlaceAccumulator) ---/

*/
class SumAccumulateTransformer : public GraphTransformer {
 public:
  SumAccumulateTransformer(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("SumAccumulateTransformer", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  bool SatisfyCondition(const Node& node) const;
};

}  // namespace onnxruntime
