// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_set>

#include "core/common/common.h"
#include "core/framework/config_options.h"
#include "core/framework/utils.h"
#include "core/optimizer/subgraph_memcpy_minimizer.h"

namespace onnxruntime {

constexpr const char* kOrtSubgraphMemcpyMinimizerNonCpuToCpuProviderRatio = "subgraph_memcpy_minimizer_min_nocpu_cpu_ratio";

SubgraphMemcpyMinimizer::SubgraphMemcpyMinimizer(
    const ConfigOptions& config_options, gsl::span<gsl::not_null<const IExecutionProvider*>> execution_providers)
    : GraphTransformer("SubgraphMemcpyMinimizer", {}), execution_providers_(execution_providers), non_cpu_to_cpu_provider_ratio_(0.5f) {
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
    auto hit = std::find_if(execution_providers_.begin(), execution_providers_.end(),
                            [&provider_type](const IExecutionProvider* ep) {
                              return ep->Type() == provider_type;
                            });
    ORT_ENFORCE(hit != execution_providers_.end(), "Node: ", node->Name(),
                " assigned to an execution provider: ", provider_type, " not among session options");
    if (!utils::ProviderIsCpuBased(**hit)) {
      nodes_not_cpu_based.insert(node);
    }
  }

  if (!nodes_not_cpu_based.empty()) {
    const float assignment_ratio = static_cast<float>(nodes_not_cpu_based.size()) / graph.NumberOfNodes();
    if (assignment_ratio < non_cpu_to_cpu_provider_ratio_) {
      LOGS(logger, WARNING) << " Falling back to CPU nodes in a Loop from " << graph.ParentNode()->OpType()
                            << " node. Non-CPU to CPU node ratio: " << assignment_ratio
                            << " is below the threshold: " << non_cpu_to_cpu_provider_ratio_;
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