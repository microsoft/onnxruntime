// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_config.h"

namespace onnxruntime {

/**
@Class PropagateCastOps

Propagate FP16 Cast operations up the graph and FP32 Cast operations down the graph

*/
class PropagateCastOps : public GraphTransformer {
 public:
  PropagateCastOps(GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy strategy =
                       GraphTransformerConfiguration::PropagateCastOpsConfiguration::Strategy::FloodFill,
                   size_t level = 0, const gsl::span<std::string const>& allow_list = {},
                   const std::unordered_set<std::string>& compatible_execution_providers = {});

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:

  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace onnxruntime
