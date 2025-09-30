// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_set>

#include "core/common/common.h"
#include "core/framework/config_options.h"
#include "core/framework/utils.h"
#include "core/optimizer/subgraph_memcpy_minimizer.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

namespace onnxruntime {

using ProviderTypeToProviderMap = SubgraphMemcpyMinimizer::ProviderTypeToProviderMap;

static ProviderTypeToProviderMap GetProvidersByType(
    gsl::span<gsl::not_null<const IExecutionProvider*>> providers) {
  SubgraphMemcpyMinimizer::ProviderTypeToProviderMap providers_by_type{};
  for (const auto provider : providers) {
    providers_by_type.emplace(provider->Type(), provider);
  }
  return providers_by_type;
}

static const IExecutionProvider* FindProviderByType(
    const ProviderTypeToProviderMap& providers_by_type,
    std::string_view provider_type) {
  const auto it = providers_by_type.find(provider_type);
  if (it != providers_by_type.end()) {
    return &*it->second;
  }
  return nullptr;
}

SubgraphMemcpyMinimizer::SubgraphMemcpyMinimizer(
    const ConfigOptions& config_options,
    gsl::span<gsl::not_null<const IExecutionProvider*>> execution_providers,
    const KernelRegistryManager& registry_manager)
    : GraphTransformer("SubgraphMemcpyMinimizer", {}),
      execution_providers_(execution_providers),
      non_cpu_to_cpu_provider_ratio_(0.5f),
      registry_manager_(registry_manager),
      providers_by_type_(GetProvidersByType(execution_providers_)) {
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

static inline bool IsLoopSubgraph(const Node& parent_node) {
  const auto& op_type = parent_node.OpType();
  return (op_type == "Loop");
}

static inline bool IsControlFlowNode(const Node& node) {
  const auto& op_type = node.OpType();
  return (op_type == "If" || op_type == "Loop" || op_type == "Scan");
}

// This function checks if the node following OutputOnCpu
// is a candidate for fallback to CPU. It should satisfy the following
// requirements:
// - It should not be a control flow node (we ignore them earlier in the loop)
// - It should be on a non-CPU based EP
// - Its inputs should be either on CPU or initializers
// - It's outputs should be consumed by CPU nodes or be graph outputs
bool SubgraphMemcpyMinimizer::IsCandidateForFallback(const Graph& graph, const Node& node,
                                                     const logging::Logger& logger) const {
  const auto& provider_type = node.GetExecutionProviderType();
  const auto* provider = FindProviderByType(providers_by_type_, provider_type);
  if (utils::ProviderIsCpuBased(*provider)) {
    return false;
  }

  // Its inputs should be either on CPU, initializer or graph inputs
  const auto input_defs = node.InputDefs();
  // If this node has only one input, no need to check because it is the input
  // from the producer node that lead us here
  if (input_defs.size() > 1) {
    for (size_t i = 0, lim = input_defs.size(); i < lim; ++i) {
      const auto* input_def = input_defs[i];
      const auto& input_name = input_def->Name();

      // check if this NodeArg is an initializer defined in current outer graph level
      const auto* initializer_tensor_proto = graph.GetInitializer(input_name, true);
      if (initializer_tensor_proto != nullptr) {
        continue;
      }

      const auto* input_node = graph.GetProducerNode(input_name);
      if (input_node != nullptr) {
        const auto& input_provider_type = input_node->GetExecutionProviderType();
        const auto* input_provider = FindProviderByType(providers_by_type_, input_provider_type);
        // Check if the provider is CPU-based OR if its output is on CPU
        if (utils::ProviderIsCpuBased(*input_provider)) {
          continue;
        }

        // Check if the producer node's output is on CPU
        const KernelCreateInfo* kci = nullptr;
        ORT_IGNORE_RETURN_VALUE(registry_manager_.SearchKernelRegistry(node, logger, &kci));
        if (!utils::IsOutputOnCpu(*input_node, kci, i)) {
          return false;
        }
      } else {
        // If there is no producer node, it should a graph input
        continue;
      }
    }
  }

  // Here we need to check if all its outputs are consumed by CPU nodes or are graph outputs
  const auto output_defs = node.OutputDefs();
  for (size_t i = 0, lim = output_defs.size(); i < lim; ++i) {
    const auto* output_def = output_defs[i];
    if (graph.IsOutput(output_def)) {
      continue;
    }
    const auto& output_name = output_def->Name();
    const auto& consumer_nodes = graph.GetConsumerNodes(output_name);
    for (const auto* consumer_node : consumer_nodes) {
      const auto& consumer_provider_type = consumer_node->GetExecutionProviderType();
      const auto* consumer_provider = FindProviderByType(providers_by_type_, consumer_provider_type);
      if (utils::ProviderIsCpuBased(*consumer_provider)) {
        continue;
      }
      // Check if the consumer node's input is on CPU
      const KernelCreateInfo* kci = nullptr;
      ORT_IGNORE_RETURN_VALUE(registry_manager_.SearchKernelRegistry(node, logger, &kci));
      if (!utils::IsInputOnCpu(*consumer_node, kci, i)) {
        return false;
      }
    }
  }
  return true;
}

Status SubgraphMemcpyMinimizer::ApplyImpl(Graph& graph, bool& modified, int graph_level,
                                          const logging::Logger& logger) const {
  modified = false;
  GraphViewer graph_viewer(graph);
  const auto& order = graph_viewer.GetNodesInTopologicalOrder();

  size_t control_flow_nodes = 0;
  std::unordered_set<Node*> nodes_not_cpu_based;
  std::unordered_set<Node*> candidates_for_fallback;
  const Node* parent_node = graph.ParentNode();
  const bool process_this_graph = (graph_level > 0) && IsLoopSubgraph(*parent_node);

  for (NodeIndex i : order) {
    auto* node = graph.GetNode(i);
    ORT_RETURN_IF_ERROR(Recurse(*node, modified, graph_level, logger));

    if (!process_this_graph) {
      continue;
    }

    if (IsControlFlowNode(*node)) {
      // do not change provider assignment for control flow nodes
      control_flow_nodes++;
      continue;
    }

    const auto* execution_provider = FindProviderByType(providers_by_type_, node->GetExecutionProviderType());
    if (!utils::ProviderIsCpuBased(*execution_provider)) {
      nodes_not_cpu_based.insert(node);

      // Let's scan the consumers of its outputs that on CPU only
      // and use IsCandidateForFallback to decide if we should fallback
      const auto output_defs = node->OutputDefs();
      for (size_t j = 0, lim = output_defs.size(); j < lim; ++j) {
        const auto* output_def = output_defs[j];
        const auto& output_name = output_def->Name();

        const KernelCreateInfo* kci = nullptr;
        ORT_IGNORE_RETURN_VALUE(registry_manager_.SearchKernelRegistry(*node, logger, &kci));
        if (!utils::IsOutputOnCpu(*node, kci, j)) {
          continue;
        }

        const auto& consumer_nodes = graph.GetConsumerNodes(output_name);
        for (const auto* consumer_node : consumer_nodes) {
          if (IsCandidateForFallback(graph, *consumer_node, logger)) {
            candidates_for_fallback.insert(node);
            break;
          }
        }
      }
    }
  }

  if (!nodes_not_cpu_based.empty() && (graph.NumberOfNodes() > control_flow_nodes)) {
    const float assignment_ratio = static_cast<float>(nodes_not_cpu_based.size()) /
                                   (graph.NumberOfNodes() - control_flow_nodes);
    if (assignment_ratio < non_cpu_to_cpu_provider_ratio_) {
      LOGS(logger, WARNING) << " Falling back to CPU nodes in a Loop from " << graph.ParentNode()->OpType()
                            << " node. Non-CPU to CPU node ratio: " << assignment_ratio
                            << " is below the threshold: " << non_cpu_to_cpu_provider_ratio_;
      for (const auto& node : nodes_not_cpu_based) {
        // Fallback to CPU
        node->SetExecutionProviderType(kCpuExecutionProvider);
      }
      modified = true;
    }
  }

  // If overall fallback did not occur, check if we found any candidates that
  // that may be surrounded by memcpy
  if (!modified) {
    for (const auto& node : candidates_for_fallback) {
      node->SetExecutionProviderType(kCpuExecutionProvider);
      modified = true;
      LOGS(logger, VERBOSE) << " Falling back to CPU node: " << node->Name()
                            << " of type: " << node->OpType()
                            << " in a Loop from " << graph.ParentNode()->OpType() << " node.";
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime