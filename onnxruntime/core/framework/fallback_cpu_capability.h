// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/graph/graph_viewer.h"
// #include "core/framework/op_kernel.h"
// #include "core/framework/fuse_nodes_funcs.h"
#include <queue>

namespace onnxruntime {

#if !defined(ORT_MINIMAL_BUILD)
const int64_t Small_Initializer_Threshold = 100;

bool IsSmallInitializer(const onnxruntime::GraphViewer& graph, const NodeArg* arg) {
  const ONNX_NAMESPACE::TensorProto* initializer_tensor;
  if (!graph.GetInitializedTensor(arg->Name(), initializer_tensor))
    return false;
  int64_t size = 1;
  for (auto& dim : initializer_tensor->dims()) {
    size *= dim;
  }
  return size <= Small_Initializer_Threshold;
}

/**
  Returns a list of compute capabilities that are prefered on CPU. 
  They are commonly shape-related computation subgraphs.
  @param graph Graph viewer 
  @param provider The targe execution provider
  @param capabilities Capabilities returned by target EP's GetCapacity() function
  */
std::vector<std::unique_ptr<ComputeCapability>>
GetCpuPreferedCapability(const onnxruntime::GraphViewer& graph,
                         const std::unique_ptr<IExecutionProvider>& provider,
                         const KernelRegistryManager& kernel_registry_mgr,
                         const std::vector<std::unique_ptr<ComputeCapability>>& capabilities) {
  const std::vector<NodeIndex>& ordered_nodes = graph.GetNodesInTopologicalOrder();
  std::vector<size_t> node_id_to_order_map(graph.MaxNodeIndex());
  for (size_t id = 0; id < ordered_nodes.size(); ++id) {
    const NodeIndex& node_id = ordered_nodes[id];
    node_id_to_order_map[node_id] = id;
  }

  // If return false, n1 will be output first; If return true, n2 will be output first
  auto greater_order_comp = [&](const NodeIndex n1, const NodeIndex n2) {
    return node_id_to_order_map[n1] > node_id_to_order_map[n2];
  };

  std::priority_queue<NodeIndex, std::vector<NodeIndex>, decltype(greater_order_comp)> candidates(greater_order_comp);
  std::unordered_set<NodeIndex> visited;

  std::unordered_set<const NodeArg*> cpu_output_args;
  std::unordered_set<NodeIndex> provider_nodes;

  for (auto& capability : capabilities) {
    const IndexedSubGraph* indexed_sub_graph = capability->sub_graph.get();
    if (indexed_sub_graph->GetMetaDef() != nullptr) {
      continue;
    }
    // The <provider> can run a single node in the <graph> if not using meta-defs.
    ORT_ENFORCE(1 == indexed_sub_graph->nodes.size());

    const NodeIndex node_id = indexed_sub_graph->nodes[0];
    provider_nodes.insert(node_id);
    const Node* node = graph.GetNode(node_id);

    const KernelCreateInfo* kernel_info = nullptr;
    std::vector<const KernelRegistry*> kernel_registries =
        kernel_registry_mgr.GetKernelRegistriesByProviderType(provider->Type());
    for (auto registry : kernel_registries) {
      auto st = registry->TryFindKernel(*node, provider->Type(), &kernel_info);
      if (st.IsOK())
        break;
    }
    // at least one registry has a CUDA kernel for this node
    ORT_ENFORCE(kernel_info != nullptr);

    // first, find all the direct consumer of cpu tensors.
    ORT_THROW_IF_ERROR(node->ForEachWithIndex(
        node->OutputDefs(),
        [&](const NodeArg& node_arg, size_t out_index) {
          if (kernel_info->kernel_def->IsOutputOnCpu(out_index)) {
            cpu_output_args.insert(&node_arg);
            auto consumer_nodes = graph.GetConsumerNodes(node_arg.Name());
            for (auto& consumer_node : consumer_nodes) {
              candidates.push(consumer_node->Index());
              LOGS_DEFAULT(INFO) << "Canditiate for fallback CPU execution: " << consumer_node->Name();
            }
          }
          return Status::OK();
        }));
  }

  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputs();
  std::unordered_set<NodeIndex> cpu_nodes;
  // The algo below is trying to identity a subgraph that only depends on cpu tensors.
  // Usually it is a subgraph that doing shape calculation based on a GPU tensor, then reshape it back.
  // The detail:
  // for each candidate, if one of its input is a cpu tensor and the cuda kernel doesn't mark it as cpu input,
  // force the node to CPU to avoid memory cpu and add its output to the small cpu tensors.
  while (!candidates.empty()) {
    NodeIndex cur = candidates.top();
    candidates.pop();
    if (visited.count(cur) != 0)
      continue;
    visited.insert(cur);

    if (provider_nodes.find(cur) == provider_nodes.end())
      continue;

    auto* node = graph.GetNode(cur);
    // skip placing current node on CPU if no kernel is found
    if (!KernelRegistryManager::HasImplementationOf(kernel_registry_mgr, *node, kCpuExecutionProvider))
      continue;

    const KernelCreateInfo* kernel_info;
    Status st = provider->GetKernelRegistry()->TryFindKernel(*node, provider->Type(), &kernel_info);

    bool place_in_cpu = true;
    for (size_t i = 0; i < node->InputDefs().size(); ++i) {
      auto* input = node->InputDefs()[i];

      // allow placing on CPU if it's a small initializer or graph input
      if (IsSmallInitializer(graph, input) ||
          std::find(graph_inputs.begin(), graph_inputs.end(), input) != graph_inputs.end()) {
        continue;
      }

      // the input is not a CPU tensor
      if (cpu_output_args.find(input) == cpu_output_args.end()) {
        place_in_cpu = false;
        break;
      }

      // input is a CPU tensor, but it's intended to be consumed as CPU input by the target EP
      if (kernel_info->kernel_def->IsInputOnCpu(i)) {
        place_in_cpu = false;
        break;
      }
    }

    if (place_in_cpu) {
      cpu_nodes.insert(cur);
      LOGS_DEFAULT(WARNING) << "Force fallback to CPU execution for node: " << node->Name();
      for (auto* output : node->OutputDefs()) {
        cpu_output_args.insert(output);
      }
      for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
        candidates.push((*it).Index());
      }
    }
  }

  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (auto index : cpu_nodes) {
    std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
    sub_graph->nodes.push_back(index);
    result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
  }
  return result;
}
#endif

}  // namespace onnxruntime
