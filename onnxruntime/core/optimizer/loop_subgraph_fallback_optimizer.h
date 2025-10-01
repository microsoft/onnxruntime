// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/gsl>
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
class LoopSubgraphCpuFallbackOptimizer : public GraphTransformer {
 public:
  using ProviderTypeToProviderMap = InlinedHashMap<std::string_view, gsl::not_null<const IExecutionProvider*>>;

  LoopSubgraphCpuFallbackOptimizer(const ConfigOptions&, gsl::span<gsl::not_null<const IExecutionProvider*>> execution_providers,
                                   const KernelRegistryManager& registry_manager);

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool IsCandidateForFallback(const Graph& graph, const Node& node, const logging::Logger& logger) const;

  float non_cpu_to_cpu_provider_ratio_;
  const KernelRegistryManager& registry_manager_;
  ProviderTypeToProviderMap providers_by_type_;
};

}  // namespace onnxruntime