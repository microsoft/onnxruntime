// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

struct ConfigOptions;

/// <summary>
/// This graph transformer attempts to minimize
/// memcpy transfers between CPU and GPU or more
/// formally between CPU based EPs and non CPU based EPs
/// in Loop subgraphs that are designed to execute
/// very often.
/// </summary>
class SubgraphMemcpyMinimizer : public GraphTransformer {
 public:
  SubgraphMemcpyMinimizer(const ConfigOptions&,
                          const InlinedHashSet<std::string_view>& compatible_execution_providers = {});

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
  float non_cpu_to_cpu_provider_ratio_;
};

}  // namespace onnxruntime