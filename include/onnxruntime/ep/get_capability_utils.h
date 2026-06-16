// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <gsl/span>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace onnxruntime {
namespace ep {

using NodeId = size_t;
constexpr int64_t kSmallInitializerThreshold = 100;

constexpr inline bool MemTypeOnCpuExplicitly(OrtMemType mem_type) {
  return mem_type == OrtMemTypeCPUInput || mem_type == OrtMemTypeCPUOutput;
}

// Get all output nodes that consume an output from the given node.
inline std::vector<Ort::ConstNode> GetOutputNodes(gsl::span<Ort::ConstValueInfo const> node_outputs) {
  std::vector<Ort::ConstNode> output_nodes;
  output_nodes.reserve(node_outputs.size());  // May have more

  // Gather the OrtNode consumers of every output.
  for (Ort::ConstValueInfo output : node_outputs) {
    if (output == nullptr) continue;  // Skip missing optional output

    auto consumers_info = output.GetConsumers();
    for (const auto& consumer : consumers_info) {
      output_nodes.push_back(consumer.node);
    }
  }

  return output_nodes;
}

// Returns nodes that should be assigned to CPU EP instead of this example EP to avoid costly I/O copies.
// Based on GetCpuPreferredNodes from onnxruntime/core/framework/fallback_cpu_capability.cc
inline OrtStatus* GetCpuPreferredNodes(const OrtGraph& ort_graph, OrtEpGraphSupportInfo& graph_support_info,
                                       const OrtLogger& logger, gsl::span<const OrtNode* const> tentative_nodes,
                                       /*out*/ std::unordered_set<const OrtNode*>& cpu_preferred_nodes) noexcept {
  try {
    const OrtApi& ort_api = Ort::GetApi();
    const OrtEpApi& ep_api = Ort::GetEpApi();
    Ort::ConstGraph graph{&ort_graph};
    std::vector<Ort::ConstNode> ordered_nodes = graph.GetNodes();

    if (ordered_nodes.empty()) {
      return nullptr;
    }

    std::unordered_map<NodeId, Ort::ConstNode> node_id_to_node;
    std::unordered_map<NodeId, size_t> node_id_to_order_map;
    for (size_t i = 0, num_nodes = ordered_nodes.size(); i < num_nodes; i++) {
      NodeId node_id = ordered_nodes[i].GetId();
      node_id_to_node[node_id] = ordered_nodes[i];
      node_id_to_order_map[node_id] = i;
    }

    // If return false, n1 will be output first; If return true, n2 will be output first
    auto greater_order_comp = [&](const NodeId node_id1, const NodeId node_id2) {
      return node_id_to_order_map[node_id1] > node_id_to_order_map[node_id2];
    };
    std::priority_queue<NodeId, std::vector<NodeId>, decltype(greater_order_comp)> candidates(greater_order_comp);
    std::unordered_set<const OrtValueInfo*> cpu_output_args;

    std::unordered_set<NodeId> provider_nodes;
    provider_nodes.reserve(tentative_nodes.size());

    std::unordered_map<NodeId, Ort::ConstKernelDef> node_to_kernel;
    node_to_kernel.reserve(tentative_nodes.size());

    for (const OrtNode* ort_node : tentative_nodes) {
      Ort::ConstNode node(ort_node);
      NodeId node_id = node.GetId();

      provider_nodes.insert(node_id);

      // Expect at least one registry has a target provider's kernel for this node.
      const OrtKernelDef* ort_kernel_def = nullptr;
      RETURN_IF_ERROR(ep_api.EpGraphSupportInfo_LookUpKernel(&graph_support_info, node, &ort_kernel_def));
      RETURN_IF(ort_kernel_def == nullptr, ort_api, "Must have a registered kernel definition on the target EP");

      Ort::ConstKernelDef kernel_def(ort_kernel_def);
      node_to_kernel.insert({node_id, kernel_def});

      // Find all the direct consumers of CPU tensors.
      std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
      for (size_t out_index = 0; out_index < outputs.size(); out_index++) {
        Ort::ConstValueInfo output = outputs[out_index];
        if (output == nullptr) continue;  // Skip missing optional output

        bool is_output_on_cpu = MemTypeOnCpuExplicitly(kernel_def.GetOutputMemType(out_index));
        if (is_output_on_cpu) {
          cpu_output_args.insert(output);

          auto consumer_infos = output.GetConsumers();
          for (const auto& consumer_info : consumer_infos) {
            candidates.push(consumer_info.node.GetId());
            ORT_CXX_LOGF(Ort::Logger(&logger), ORT_LOGGING_LEVEL_INFO, "Candidate for fallback CPU execution: %s\n",
                         consumer_info.node.GetName().c_str());
          }
        }
      }
    }

    std::unordered_set<NodeId> visited;
    visited.reserve(candidates.size());

    std::unordered_set<const OrtNode*> cpu_nodes;
    cpu_nodes.reserve(candidates.size());

    // The algo below is trying to identity a subgraph that only depends on cpu tensors.
    // Usually it is a subgraph that doing shape calculation based on a GPU tensor, then reshape it back.
    // The detail:
    // for each candidate, if one of its input is a cpu tensor and the Non-CPU kernel doesn't mark it as cpu input,
    // force the node to CPU to avoid memory cpu and add its output to the small cpu tensors.
    while (!candidates.empty()) {
      NodeId cur = candidates.top();
      candidates.pop();

      auto p = visited.insert(cur);
      if (!p.second) {
        continue;
      }

      auto node_iter = node_id_to_node.find(cur);
      RETURN_IF(node_iter == node_id_to_node.end(), ort_api, "Unable to get OrtNode for a given node ID");
      Ort::ConstNode node = node_iter->second;

      if (provider_nodes.find(cur) == provider_nodes.end()) {
        // Nodes not in provider_nodes are either have EP assigned or no kernel found on target EP.
        // we assume these nodes will fallback to CPU, so add all direct consumers of all outputs to candidates.
        std::string ep_name = node.GetEpName();
        if (ep_name.empty() || ep_name == "CPUExecutionProvider") {
          std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();

          for (Ort::ConstValueInfo output : outputs) {
            if (output == nullptr) continue;  // Skip missing optional output
            cpu_output_args.insert(output);
          }

          for (Ort::ConstNode downstream_node : GetOutputNodes(outputs)) {
            candidates.push(downstream_node.GetId());
          }
        }
        continue;
      }

      std::vector<Ort::ConstValueInfo> inputs = node.GetInputs();
      bool place_in_cpu = true;

      for (size_t i = 0; i < inputs.size(); i++) {
        Ort::ConstValueInfo input = inputs[i];
        if (input == nullptr) continue;  // Skip missing optional input

        // skip placing on CPU if the data types is float16 or bfloat16 or
        // float8e4m3fn, float8e4m3fnuz, float8e5m2, float8e5m2fnuz or float4e2m1
        Ort::ConstTypeInfo type_info = input.TypeInfo();
        auto type_shape_info = type_info.GetTensorTypeAndShapeInfo();
        auto elem_type = type_shape_info.GetElementType();
        if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ||
            elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16 ||
            elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN ||
            elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ ||
            elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2 ||
            elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ ||
            elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT4E2M1) {
          place_in_cpu = false;
          break;
        }

        bool is_small_initializer = input.IsConstantInitializer() &&
                                    type_shape_info.GetElementCount() <= kSmallInitializerThreshold;

        // Allow placing on CPU if it's a small initializer or graph input
        if (is_small_initializer || input.IsRequiredGraphInput() || input.IsOptionalGraphInput()) {
          continue;
        }

        // the input is not a CPU tensor
        if (cpu_output_args.find(input) == cpu_output_args.end()) {
          place_in_cpu = false;
          break;
        }

        // input is a CPU tensor, but it's intended to be consumed as CPU input by the target EP
        bool is_input_on_cpu = MemTypeOnCpuExplicitly(node_to_kernel[cur].GetInputMemType(i));
        if (is_input_on_cpu) {
          place_in_cpu = false;
          break;
        }
      }

      if (place_in_cpu) {
        cpu_nodes.insert(node);
        ORT_CXX_LOGF(Ort::Logger(&logger), ORT_LOGGING_LEVEL_WARNING,
                     "EP optimization: Force fallback to CPU execution for node %s because the CPU execution path "
                     "is deemed faster than overhead involved with execution on other EPs capable of executing "
                     "this node.\n",
                     node.GetName().c_str());

        std::vector<Ort::ConstValueInfo> outputs = node.GetOutputs();
        for (Ort::ConstValueInfo output : outputs) {
          if (output == nullptr) continue;  // Skip missing optional output
          cpu_output_args.insert(output);
        }

        for (Ort::ConstNode downstream_node : GetOutputNodes(outputs)) {
          candidates.push(downstream_node.GetId());
        }
      }
    }

    cpu_preferred_nodes = std::move(cpu_nodes);

    return nullptr;
  } catch (const Ort::Exception& ex) {
    Ort::Status status(ex);
    return status.release();
  } catch (const std::exception& ex) {
    Ort::Status status(ex.what(), ORT_EP_FAIL);
    return status.release();
  } catch (...) {
    Ort::Status status("Unknown exception", ORT_EP_FAIL);
    return status.release();
  }
}

}  // namespace ep
}  // namespace onnxruntime
