// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_set>

#include "core/framework/config_options.h"
#include "core/framework/utils.h"
#include "core/optimizer/subgraph_memcpy_minimizer.h"

namespace onnxruntime {

constexpr const char* kOrtSubgraphMemcpyMinimizerNonCpuToCpuProviderRatio = "subgraph_memcpy_minimizer_min_nocpu_cpu_ratio";

SubgraphMemcpyMinimizer::SubgraphMemcpyMinimizer(
    const ConfigOptions& config_options, const InlinedHashSet<std::string_view>& compatible_execution_providers)
    : GraphTransformer("SubgraphMemcpyMinimizer", compatible_execution_providers),
      non_cpu_to_cpu_provider_ratio_(0.5f) {
  std::string config_value;
  if (config_options.TryGetConfigEntry(kOrtSubgraphMemcpyMinimizerNonCpuToCpuProviderRatio,
                                        config_value)) {
    try {
      non_cpu_to_cpu_provider_ratio_ = std::stof(config_value);
    } catch (const std::exception&) {
      // ignore error and use default value for now
    }
  }
}

static bool ShouldProcessSubgraph(const Node& parent_node) {
  const auto& op_type = parent_node.OpType();
  return (op_type == "Loop" || op_type == "SequenceMapLoop");
}

Status SubgraphMemcpyMinimizer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                          const logging::Logger& logger) const {
  modified = false;
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  size_t nodes_cpu_based = 0;
  std::unordered_set<Node*> nodes_not_cpu_based;

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);

    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    // Do not process the main graph nodes
    if (graph_level == 0) {
      continue;
    }

    assert(graph.IsSubgraph());
    const Node* parent_node = graph.ParentNode();
    // Do not process nodes of the subgraph if the parent node is not one of the loops
    if (!ShouldProcessSubgraph(*parent_node)) {
      continue;
    }

    const auto& provider_type = node->GetExecutionProviderType();
    // XXX: Refine this check
    // How do we handle multiple non-cpu providers?
    if (provider_type == kCudaExecutionProvider) {
      nodes_not_cpu_based.insert(node);
    } else {
      nodes_cpu_based++;
    }
  }

  if (!nodes_not_cpu_based.empty() && nodes_cpu_based > 0) {
    const float assignment_ratio = static_cast<float>(nodes_not_cpu_based.size()) / (nodes_cpu_based);
    if (assignment_ratio < non_cpu_to_cpu_provider_ratio_) {
      for (const auto& node : nodes_not_cpu_based) {
        // Fallback to CPU
        node->SetExecutionProviderType(kCpuExecutionProvider);
      }
    }
    modified = true;
  }

  return Status::OK();
}

}  // namespace onnxruntime