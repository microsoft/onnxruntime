// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class PropagateCastOps

Propagate FP16 Cast operations up the graph and FP32 Cast operations down the graph

*/
class PropagateCastOps : public GraphTransformer {
public:
  PropagateCastOps(int level, const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("PropagateCastOps", compatible_execution_providers), level_(level) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  int level_;
};

}  // namespace onnxruntime
