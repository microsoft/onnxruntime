// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/fallback_cpu_capability.h"

#include <queue>

#include "onnx/defs/data_type_utils.h"

#include "core/framework/execution_providers.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/op_kernel.h"
#include "core/providers/cpu/cpu_execution_provider.h"

using namespace ONNX_NAMESPACE::Utils;

namespace onnxruntime {

namespace {
const int64_t kSmallInitializerThreshold = 100;

bool IsSmallInitializer(const onnxruntime::GraphViewer& graph, const NodeArg* arg) {
  const ONNX_NAMESPACE::TensorProto* initializer_tensor;
  if (!graph.GetInitializedTensor(arg->Name(), initializer_tensor))
    return false;
  int64_t size = 1;
  for (auto& dim : initializer_tensor->dims()) {
    size *= dim;
  }

  return size <= kSmallInitializerThreshold;
}
}  // namespace

std::unordered_set<NodeIndex> GetCpuPreferredNodes(const onnxruntime::GraphViewer& graph,
                                                   const std::string& provider_type,
                                                   const std::vector<const KernelRegistry*>& kernel_registries,
                                                   const std::vector<NodeIndex>& tentative_nodes) {
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

  // If return false, n2 will be output first; If return true, n1 will be output first
  auto lesser_order_comp = [&](const NodeIndex n1, const NodeIndex n2) {
    return node_id_to_order_map[n1] < node_id_to_order_map[n2];
  };

  std::priority_queue<NodeIndex, std::vector<NodeIndex>, decltype(greater_order_comp)> candidates_fw(greater_order_comp);
  std::priority_queue<NodeIndex, std::vector<NodeIndex>, decltype(lesser_order_comp)> candidates_bw(lesser_order_comp);
  std::unordered_set<NodeIndex> visited;

  std::unordered_set<const NodeArg*> cpu_args;
  std::unordered_set<NodeIndex> provider_nodes;
  std::unordered_map<NodeIndex, const KernelCreateInfo*> node_to_kernel;
  std::unordered_set<NodeIndex> cpu_kernel_available;

  // create a temp CPU kernel registry
  KernelRegistryManager mgr;
  ExecutionProviders cpu_ep;
  CPUExecutionProviderInfo epi{false};
  ORT_ENFORCE(cpu_ep.Add(kCpuExecutionProvider, std::make_unique<CPUExecutionProvider>(epi)).IsOK());
  ORT_ENFORCE(mgr.RegisterKernels(cpu_ep).IsOK());
  std::vector<const KernelRegistry*> cpu_kernel_registries = mgr.GetKernelRegistriesByProviderType(kCpuExecutionProvider);

  for (auto& node_id : tentative_nodes) {
    provider_nodes.insert(node_id);
    const Node* node = graph.GetNode(node_id);

    const KernelCreateInfo* kernel_info = nullptr;

    // Get the CPU kernel availability for this node
    for (auto registry : cpu_kernel_registries) {
      auto st = registry->TryFindKernel(*node, kCpuExecutionProvider, &kernel_info);
      if (st.IsOK()) {
        cpu_kernel_available.insert(node_id);
        break;
      }
    }

    for (auto registry : kernel_registries) {
      auto st = registry->TryFindKernel(*node, provider_type, &kernel_info);
      if (st.IsOK())
        break;
    }
    // at least one registry has a target provider's kernel for this node
    ORT_ENFORCE(kernel_info != nullptr);
    node_to_kernel.insert({node_id, kernel_info});

    // first, find all the direct consumer of cpu tensors.
    ORT_THROW_IF_ERROR(node->ForEachWithIndex(
        node->OutputDefs(),
        [&](const NodeArg& node_arg, size_t out_index) {
          if (kernel_info->kernel_def->IsOutputOnCpu(out_index)) {
            cpu_args.insert(&node_arg);
            auto consumer_nodes = graph.GetConsumerNodes(node_arg.Name());
            for (auto& consumer_node : consumer_nodes) {
              candidates_fw.push(consumer_node->Index());
              LOGS_DEFAULT(INFO) << "Candidate for fallback CPU execution in forward trace: " << consumer_node->Name();
            }
          }
          return Status::OK();
        }));

    // then, find all the direct producers of cpu tensors.
    ORT_THROW_IF_ERROR(node->ForEachWithIndex(
        node->InputDefs(),
        [&](const NodeArg& node_arg, size_t in_index) {
          if (kernel_info->kernel_def->IsInputOnCpu(in_index)) {
            cpu_args.insert(&node_arg);
            auto producer_node = graph.GetProducerNode(node_arg.Name());
            if (producer_node != nullptr) {
              candidates_bw.push(producer_node->Index());
              LOGS_DEFAULT(INFO) << "Candidate for fallback CPU execution in backward trace: " << producer_node->Name();
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
  // for each candidate, if one of its input is a cpu tensor and the Non-CPU kernel doesn't mark it as cpu input,
  // force the node to CPU to avoid memory cpu and add its output to the small cpu tensors.
  while (!candidates_fw.empty()) {
    NodeIndex cur = candidates_fw.top();
    candidates_fw.pop();
    if (visited.count(cur) != 0)
      continue;
    visited.insert(cur);

    if (provider_nodes.find(cur) == provider_nodes.end())
      continue;

    auto* node = graph.GetNode(cur);
    bool place_in_cpu = true;
    for (size_t i = 0; i < node->InputDefs().size(); ++i) {
      auto* input = node->InputDefs()[i];

      // skip placing on CPU if the data typs is float16 or bfloat16
      if (input->Type() == DataTypeUtils::ToType("float16") ||
          input->Type() == DataTypeUtils::ToType("bfloat16")) {
        place_in_cpu = false;
        break;
      }

      // allow placing on CPU if it's a small initializer or graph input
      if (IsSmallInitializer(graph, input) ||
          std::find(graph_inputs.begin(), graph_inputs.end(), input) != graph_inputs.end()) {
        continue;
      }

      // the input is not a CPU tensor
      if (cpu_args.find(input) == cpu_args.end()) {
        place_in_cpu = false;
        break;
      }

      // input is a CPU tensor, but it's intended to be consumed as CPU input by the target EP
      if (node_to_kernel[cur]->kernel_def->IsInputOnCpu(i)) {
        place_in_cpu = false;
        break;
      }
    }

    if (place_in_cpu && cpu_kernel_available.count(cur) != 0) {
      cpu_nodes.insert(cur);
      LOGS_DEFAULT(INFO) << "ORT optimization- Force fallback to CPU execution for node: " << node->Name()
                         << " because the CPU execution path is deemed faster than overhead involved with execution on other EPs "
                         << " capable of executing this node";
      for (auto* output : node->OutputDefs()) {
        cpu_args.insert(output);
      }
      for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
        candidates_fw.push((*it).Index());
      }
    }
  }
  // clear the visited to prepare for backward trace
  visited.clear();
  // Trace the graph backwards to find additional CPU nodes
  // Starting from nodes that must produce an output on CPU, trace the producer nodes
  // The trace stops when we find that
  // 1) The node is already picked for CPU
  // 2) Input/Output type is unsupported on CPU(float16/bfloat16)
  // 3) The output is not a CPU tensor
  // 4) The search hits a node that produces a CPU output
  while (!candidates_bw.empty()) {
    NodeIndex cur = candidates_bw.top();
    candidates_bw.pop();
    if (visited.count(cur) != 0)
      continue;
    visited.insert(cur);

    // node is already picked for CPU
    if (cpu_nodes.count(cur) != 0)
      continue;

    if (provider_nodes.find(cur) == provider_nodes.end())
      continue;

    auto* node = graph.GetNode(cur);
    bool place_in_cpu = true;
    for (size_t i = 0; i < node->OutputDefs().size(); ++i) {
      auto* output = node->OutputDefs()[i];

      // skip placing on CPU if the data typs is float16 or bfloat16
      if (output->Type() == DataTypeUtils::ToType("float16") ||
          output->Type() == DataTypeUtils::ToType("bfloat16")) {
        place_in_cpu = false;
        break;
      }

      // the output is not a CPU tensor
      if (cpu_args.find(output) == cpu_args.end()) {
        place_in_cpu = false;
        break;
      }

      // output is a CPU tensor, but it's intended to be consumed as CPU output by the target EP
      if (node_to_kernel[cur]->kernel_def->IsOutputOnCpu(i)) {
        place_in_cpu = false;
        break;
      }
    }
    // Next, check if the node inputs are of supported type
    if (place_in_cpu) {
      for (size_t i = 0; i < node->InputDefs().size(); ++i) {
        auto* input = node->InputDefs()[i];

        // skip placing on CPU if the data typs is float16 or bfloat16
        if (input->Type() == DataTypeUtils::ToType("float16") ||
            input->Type() == DataTypeUtils::ToType("bfloat16")) {
          place_in_cpu = false;
          break;
        }
      }
    }

    if (place_in_cpu && cpu_kernel_available.count(cur) != 0) {
      cpu_nodes.insert(cur);
      LOGS_DEFAULT(INFO) << "ORT optimization- Force fallback to CPU execution for node: " << node->Name()
                         << " because the CPU execution path is deemed faster than overhead involved with execution on other EPs "
                         << " capable of executing this node";
      for (auto* input : node->InputDefs()) {
        cpu_args.insert(input);
      }
      for (auto it = node->InputNodesBegin(); it != node->InputNodesEnd(); ++it) {
        candidates_bw.push((*it).Index());
      }
    }
  }

  return cpu_nodes;
}

}  // namespace onnxruntime
