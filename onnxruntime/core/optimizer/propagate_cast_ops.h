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
  PropagateCastOps(size_t level, std::vector<std::string> allow_list = {},
                   const std::unordered_set<std::string>& compatible_execution_providers = {}) noexcept;

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

private:
  size_t level_;
};

}  // namespace onnxruntime
