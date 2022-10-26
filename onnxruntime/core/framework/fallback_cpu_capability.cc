// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/fallback_cpu_capability.h"
#include "core/common/inlined_containers.h"

#include <queue>

#include "onnx/defs/data_type_utils.h"

#include "core/framework/op_kernel.h"

using namespace ONNX_NAMESPACE::Utils;

namespace onnxruntime {

namespace {
constexpr int64_t kSmallInitializerThreshold = 100;

static bool IsSmallInitializer(const onnxruntime::GraphViewer& graph, const NodeArg* arg) {
  // 'true' in the function call is to let the searching for the initializer
  // continue in the outer scopes of the current (sub-)graph if applicable
  const ONNX_NAMESPACE::TensorProto* initializer_tensor =
      graph.GetGraph().GetInitializer(arg->Name(), true);

  // Not an initializer at all
  if (initializer_tensor == nullptr) {
    return false;
  }

  // Check if "small" enough
  int64_t size = 1;
  for (auto& dim : initializer_tensor->dims()) {
    size *= dim;
  }

  return size <= kSmallInitializerThreshold;
}
}  // namespace

std::unordered_set<NodeIndex> GetCpuPreferredNodes(const onnxruntime::GraphViewer& graph,
                                                   const IExecutionProvider::IKernelLookup& kernel_lookup,
                                                   gsl::span<const NodeIndex> tentative_nodes) {
  // automatic conversion from const std::vector&
  const auto& ordered_nodes = graph.GetNodesInTopologicalOrder();
  InlinedVector<size_t> node_id_to_order_map(graph.MaxNodeIndex());
  for (size_t id = 0, limit = ordered_nodes.size(); id < limit; ++id) {
    const NodeIndex& node_id = ordered_nodes[id];
    node_id_to_order_map[node_id] = id;
  }

  // If return false, n1 will be output first; If return true, n2 will be output first
  auto greater_order_comp = [&](const NodeIndex n1, const NodeIndex n2) {
    return node_id_to_order_map[n1] > node_id_to_order_map[n2];
  };

  std::priority_queue<NodeIndex, std::vector<NodeIndex>, decltype(greater_order_comp)> candidates(greater_order_comp);

  InlinedHashSet<const NodeArg*> cpu_output_args;

  InlinedHashSet<NodeIndex> provider_nodes;
  provider_nodes.reserve(tentative_nodes.size());

  InlinedHashMap<NodeIndex, const KernelCreateInfo*> node_to_kernel;
  node_to_kernel.reserve(tentative_nodes.size());

  for (auto& node_id : tentative_nodes) {
    provider_nodes.insert(node_id);
    const Node* node = graph.GetNode(node_id);

    if (node->Name() == "Unsqueeze_983" || node->Name() == "Concat_984" || node->Name() == "Div_980") {
      printf("Adding %s to provider_nodes\n", node->Name().c_str());
    }

    const KernelCreateInfo* kernel_info = kernel_lookup.LookUpKernel(*node);
    // at least one registry has a target provider's kernel for this node
    ORT_ENFORCE(kernel_info != nullptr);
    node_to_kernel.insert({node_id, kernel_info});

    // first, find all the direct consumer of cpu tensors.
    ORT_THROW_IF_ERROR(node->ForEachWithIndex(
        node->OutputDefs(),
        [&](const NodeArg& node_arg, size_t out_index) {
          if (kernel_info->kernel_def->IsOutputOnCpu(out_index)) {
            if (node->Name() == "Unsqueeze_983" || node->Name() == "Concat_984" || node->Name() == "Div_980") {
              printf("%s output is on CPU\n", node->Name().c_str());
            }

            cpu_output_args.insert(&node_arg);
            auto consumer_nodes = graph.GetConsumerNodes(node_arg.Name());
            for (auto& consumer_node : consumer_nodes) {
              if (consumer_node->Name() == "Unsqueeze_983" || consumer_node->Name() == "Concat_984" || node->Name() == "Div_980") {
                printf("%s IS A CONSUMER NODE of %s!!!\n", consumer_node->Name().c_str(), node->Name().c_str());
              }

              if (node->Name() == "Unsqueeze_983" || node->Name() == "Concat_984" || node->Name() == "Div_980") {
                printf("Adding consumer node (%s) for node %s\n", consumer_node->Name().c_str(), node->Name().c_str());
              }

              candidates.push(consumer_node->Index());
              LOGS_DEFAULT(INFO) << "Candidate for fallback CPU execution: " << consumer_node->Name();
            }
          } else {
            if (node->Name() == "Unsqueeze_983" || node->Name() == "Concat_984" || node->Name() == "Div_980") {
              printf("%s output is NOT on CPU\n", node->Name().c_str());
            }
          }
          return Status::OK();
        }));
  }

  const auto& graph_inputs = graph.GetInputs();
  InlinedHashSet<NodeIndex> visited;
  visited.reserve(candidates.size());
  std::unordered_set<NodeIndex> cpu_nodes;
  cpu_nodes.reserve(candidates.size());
  // The algo below is trying to identity a subgraph that only depends on cpu tensors.
  // Usually it is a subgraph that doing shape calculation based on a GPU tensor, then reshape it back.
  // The detail:
  // for each candidate, if one of its input is a cpu tensor and the Non-CPU kernel doesn't mark it as cpu input,
  // force the node to CPU to avoid memory cpu and add its output to the small cpu tensors.
  while (!candidates.empty()) {
    NodeIndex cur = candidates.top();
    candidates.pop();

    auto* node3 = graph.GetNode(cur);
    if (node3->Name() == "Unsqueeze_983" || node3->Name() == "Concat_984" || node3->Name() == "Div_980") {
      printf("Popping %s\n", node3->Name().c_str());
    }

    auto p = visited.insert(cur);
    if (!p.second) {
      auto* node2 = graph.GetNode(cur);
      if (node2->Name() == "Unsqueeze_983" || node2->Name() == "Concat_984" || node2->Name() == "Div_980") {
        printf("%s already visited\n", node2->Name().c_str());
      }
      continue;
    }

    if (provider_nodes.find(cur) == provider_nodes.end()) {
      auto* node2 = graph.GetNode(cur);
      if (node2->Name() == "Unsqueeze_983" || node2->Name() == "Concat_984" || node2->Name() == "Div_980") {
        printf("%s not found in provider_nodes\n", node2->Name().c_str());
      }
      continue;
    }

    auto* node = graph.GetNode(cur);
    bool place_in_cpu = true;
    for (size_t i = 0; i < node->InputDefs().size(); ++i) {
      auto* input = node->InputDefs()[i];

      // skip placing on CPU if the data typs is float16 or bfloat16
      if (input->Type() == DataTypeUtils::ToType("float16") ||
          input->Type() == DataTypeUtils::ToType("bfloat16")) {
        if (node->Name() == "Unsqueeze_983" || node->Name() == "Concat_984" || node->Name() == "Div_980") {
          printf("%s: skip placing on CPU if the data typs is float16 or bfloat16\n", node->Name().c_str());
        }
        place_in_cpu = false;
        break;
      }

      // allow placing on CPU if it's a small initializer or graph input
      if (IsSmallInitializer(graph, input) ||
          std::find(graph_inputs.begin(), graph_inputs.end(), input) != graph_inputs.end()) {
        continue;
      }

      // the input is not a CPU tensor
      if (cpu_output_args.find(input) == cpu_output_args.end()) {
        if (node->Name() == "Unsqueeze_983" || node->Name() == "Concat_984" || node->Name() == "Div_980") {
          printf("%s: the input is not a CPU tensor\n", node->Name().c_str());
        }
        place_in_cpu = false;
        break;
      }

      // input is a CPU tensor, but it's intended to be consumed as CPU input by the target EP
      if (node_to_kernel[cur]->kernel_def->IsInputOnCpu(i)) {
        place_in_cpu = false;
        if (node->Name() == "Unsqueeze_983" || node->Name() == "Concat_984" || node->Name() == "Div_980") {
          printf("%s: input is a CPU tensor, but it's intended to be consumed as CPU input by the target EP\n", node->Name().c_str());
        }
        break;
      }
    }

    if (place_in_cpu) {
      if (node->Name() == "Unsqueeze_983" || node->Name() == "Concat_984" || node->Name() == "Div_980") {
        printf("Inserting %s in cpu_nodes\n", node->Name().c_str());
      }

      cpu_nodes.insert(cur);
      LOGS_DEFAULT(INFO) << "ORT optimization- Force fallback to CPU execution for node: " << node->Name()
                         << " because the CPU execution path is deemed faster than overhead involved with execution on other EPs "
                         << " capable of executing this node";
      for (auto* output : node->OutputDefs()) {
        cpu_output_args.insert(output);
      }
      for (auto it = node->OutputNodesBegin(); it != node->OutputNodesEnd(); ++it) {
        candidates.push((*it).Index());
      }
    }
    else
    {
      if (node->Name() == "Unsqueeze_983" || node->Name() == "Concat_984" || node->Name() == "Div_980") {
        printf("Not inserting %s in cpu_nodes\n", node->Name().c_str());
      }
    }
  }

  return cpu_nodes;
}

}  // namespace onnxruntime
