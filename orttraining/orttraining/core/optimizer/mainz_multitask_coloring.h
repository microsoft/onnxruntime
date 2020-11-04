// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class MainzMultitaskColoring

Color Mainz's multi-task graph
*/
class MainzMultitaskColoring : public GraphTransformer {
 public:
  MainzMultitaskColoring(const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("MainzMultitaskColoring", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

private:
  const Node* SatisfyCondition(const Node& node) const;
};

}  // namespace onnxruntime
