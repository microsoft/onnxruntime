// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/fast_gelu_fusion.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// FastGelu supports limited data types.
static std::vector<std::string> gpu_supported_data_types{"tensor(float16)", "tensor(float)", "tensor(bfloat16)"};
static std::vector<std::string> cpu_supported_data_types{"tensor(float)"};

static bool IsSupportedDataType(const Node& node) {
  if (node.GetExecutionProviderType() == kCpuExecutionProvider) {
    return optimizer_utils::IsSupportedDataType(node, cpu_supported_data_types);
  } else {
    return optimizer_utils::IsSupportedDataType(node, gpu_supported_data_types);
  }
}

static bool CheckNode(Graph& graph, const Node& node, const std::string& op_name,
                      const std::initializer_list<ONNX_NAMESPACE::OperatorSetVersion>& opset_versions,
                      ProviderType provider, bool require_single_output) {
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, op_name, opset_versions) &&
         node.GetExecutionProviderType() == provider &&
         IsSupportedDataType(node) &&
         (!require_single_output || node.GetOutputEdgesCount() == 1) &&
         graph.GetNodeOutputsInGraphOutputs(node).empty();
}

MatchResult FastGeluFusion::CheckFirstFormula(Graph& graph, Node& mul1_node,
                                              std::vector<std::reference_wrapper<Node>>& nodes_to_fuse) const {
  MatchResult matchResult{false, nullptr, nullptr};
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul1_node, "Mul", {7, 13}) ||
      !graph_utils::IsSupportedProvider(mul1_node, GetCompatibleExecutionProviders()) ||
      mul1_node.GetOutputEdgesCount() != 1 ||
      !IsSupportedDataType(mul1_node)) {
    return matchResult;
  }

  int32_t input_index = -1;
  const float mul_val = 0.044715f;
  for (auto i = 0; i < 2; i++) {
    if (optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul1_node.InputDefs()[i]), mul_val, true)) {
      input_index = i;
      break;
    }
  }

  if (input_index == -1) return matchResult;

  NodeArg* gelu_without_bias_input_arg = mul1_node.MutableInputDefs()[(input_index + 1) % 2];
  nodes_to_fuse.push_back(mul1_node);

  Node& mul2_node = *graph.GetNode(mul1_node.OutputNodesBegin()->Index());
  input_index = optimizer_utils::IndexOfNodeInput(mul2_node, *mul1_node.MutableOutputDefs()[0]);
  if (!CheckNode(graph, mul2_node, "Mul", {7, 13}, mul1_node.GetExecutionProviderType(), true) ||
      mul2_node.MutableInputDefs()[(input_index + 1) % 2]->Name() != gelu_without_bias_input_arg->Name()) {
    return matchResult;
  }
  nodes_to_fuse.push_back(mul2_node);

  Node& add1_node = *graph.GetNode(mul2_node.OutputNodesBegin()->Index());
  input_index = optimizer_utils::IndexOfNodeInput(add1_node, *mul2_node.MutableOutputDefs()[0]);
  if (!CheckNode(graph, add1_node, "Add", {7, 13}, mul1_node.GetExecutionProviderType(), true) ||
      !optimizer_utils::IsInitializerWithExpectedValue(graph, *(add1_node.InputDefs()[(input_index + 1) % 2]), 1.0f, true)) {
    return matchResult;
  }
  nodes_to_fuse.push_back(add1_node);

  Node& mul3_node = *graph.GetNode(add1_node.OutputNodesBegin()->Index());
  if (!CheckNode(graph, mul3_node, "Mul", {7, 13}, mul1_node.GetExecutionProviderType(), true)) {
    return matchResult;
  }
  nodes_to_fuse.push_back(mul3_node);

  input_index = optimizer_utils::IndexOfNodeInput(mul3_node, *add1_node.MutableOutputDefs()[0]);
  const Node* p_mul3_input_node = graph_utils::GetInputNode(mul3_node, (input_index + 1) % 2);
  if (p_mul3_input_node == nullptr) return matchResult;
  Node& mul4_node = const_cast<Node&>(*p_mul3_input_node);
  if (!CheckNode(graph, mul4_node, "Mul", {7, 13}, mul1_node.GetExecutionProviderType(), true)) {
    return matchResult;
  }

  input_index = -1;
  const float mul4_val = 0.7978845834732056f;
  for (auto i = 0; i < 2; i++) {
    if (optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul4_node.InputDefs()[i]), mul4_val, true)) {
      input_index = i;
      break;
    }
  }

  if (input_index == -1 || mul4_node.InputDefs()[(input_index + 1) % 2]->Name() != gelu_without_bias_input_arg->Name())
    return matchResult;
  nodes_to_fuse.push_back(mul4_node);

  matchResult.matched = true;
  matchResult.gelu_without_bias_input_arg = gelu_without_bias_input_arg;
  matchResult.tanh_input_node = &mul3_node;
  return matchResult;
}

MatchResult FastGeluFusion::CheckSecondFormula(Graph& graph, Node& pow1_node,
                                               std::vector<std::reference_wrapper<Node>>& nodes_to_fuse) const {
  MatchResult matchResult{false, nullptr, nullptr};
  if (!graph_utils::IsSupportedOptypeVersionAndDomain(pow1_node, "Pow", {7, 12, 13}) ||
      !graph_utils::IsSupportedProvider(pow1_node, GetCompatibleExecutionProviders()) ||
      pow1_node.GetOutputEdgesCount() != 1 ||
      !IsSupportedDataType(pow1_node)) {
    return matchResult;
  }

  if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(pow1_node.InputDefs()[1]), 3.0f, true)) {
    return matchResult;
  }

  NodeArg* pow_input_arg = pow1_node.MutableInputDefs()[0];
  nodes_to_fuse.push_back(pow1_node);

  Node& mul1_node = *graph.GetNode(pow1_node.OutputNodesBegin()->Index());
  auto input_index = optimizer_utils::IndexOfNodeInput(mul1_node, *pow1_node.MutableOutputDefs()[0]);
  if (!CheckNode(graph, mul1_node, "Mul", {7, 13}, pow1_node.GetExecutionProviderType(), true) ||
      !optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul1_node.InputDefs()[(input_index + 1) % 2]),
                                                       0.044714998453855515f, true)) {
    return matchResult;
  }
  nodes_to_fuse.push_back(mul1_node);

  Node& add1_node = *graph.GetNode(mul1_node.OutputNodesBegin()->Index());
  input_index = optimizer_utils::IndexOfNodeInput(add1_node, *mul1_node.MutableOutputDefs()[0]);
  if (!CheckNode(graph, add1_node, "Add", {7, 13}, pow1_node.GetExecutionProviderType(), true) ||
      add1_node.MutableInputDefs()[(input_index + 1) % 2]->Name() != pow_input_arg->Name()) {
    return matchResult;
  }
  nodes_to_fuse.push_back(add1_node);

  Node& mul2_node = *graph.GetNode(add1_node.OutputNodesBegin()->Index());
  input_index = optimizer_utils::IndexOfNodeInput(mul2_node, *add1_node.MutableOutputDefs()[0]);
  if (!CheckNode(graph, mul2_node, "Mul", {7, 13}, pow1_node.GetExecutionProviderType(), true) ||
      !optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul2_node.InputDefs()[(input_index + 1) % 2]),
                                                       0.7978845834732056f, true)) {
    return matchResult;
  }
  nodes_to_fuse.push_back(mul2_node);

  matchResult.matched = true;
  matchResult.gelu_without_bias_input_arg = pow_input_arg;
  matchResult.tanh_input_node = &mul2_node;
  return matchResult;
}

Status FastGeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    Node& node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level, logger));

    std::vector<std::reference_wrapper<Node>> nodes_to_fuse;
    MatchResult matchRet = CheckFirstFormula(graph, node, nodes_to_fuse);
    if (!matchRet.matched) {
      nodes_to_fuse.clear();
      matchRet = CheckSecondFormula(graph, node, nodes_to_fuse);

      if (!matchRet.matched) continue;
    };

    Node& tanh_node = *graph.GetNode(matchRet.tanh_input_node->OutputNodesBegin()->Index());
    if (!CheckNode(graph, tanh_node, "Tanh", {6, 13}, node.GetExecutionProviderType(), true)) {
      continue;
    }

    Node& add2_node = *graph.GetNode(tanh_node.OutputNodesBegin()->Index());
    if (!CheckNode(graph, add2_node, "Add", {7, 13}, node.GetExecutionProviderType(), true)) {
      continue;
    }

    auto input_index = optimizer_utils::IndexOfNodeInput(add2_node, *tanh_node.MutableOutputDefs()[0]);
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(add2_node.InputDefs()[(input_index + 1) % 2]), 1.0f, true)) {
      continue;
    }

    Node& mul5_node = *graph.GetNode(add2_node.OutputNodesBegin()->Index());
    // This is the output of the Gelu subgraph, we don't need check it has single edge.
    if (!CheckNode(graph, mul5_node, "Mul", {7, 13}, node.GetExecutionProviderType(), false)) {
      continue;
    }

    input_index = optimizer_utils::IndexOfNodeInput(mul5_node, *add2_node.MutableOutputDefs()[0]);
    const Node* p_mul5_input_node = graph_utils::GetInputNode(mul5_node, (input_index + 1) % 2);
    if (p_mul5_input_node == nullptr) continue;
    Node& mul6_node = const_cast<Node&>(*p_mul5_input_node);
    if (!CheckNode(graph, mul6_node, "Mul", {7, 13}, node.GetExecutionProviderType(), false)) {
      continue;
    }

    input_index = -1;
    for (auto i = 0; i < 2; i++) {
      if (optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul6_node.InputDefs()[i]), 0.5f, true)) {
        input_index = i;
        break;
      }
    }

    if (input_index == -1 || mul6_node.InputDefs()[(input_index + 1) % 2]->Name() != matchRet.gelu_without_bias_input_arg->Name())
      continue;

    std::vector<NodeArg*> gelu_input_defs{matchRet.gelu_without_bias_input_arg};
    nodes_to_fuse.insert(nodes_to_fuse.end(), {tanh_node, add2_node, mul6_node, mul5_node});

    auto type_info = *node.MutableOutputDefs()[0]->TypeAsProto();
    // TODO: re-use node arg of mul5 so that it is allowed to be graph output (Need modify CheckNode as well).
    auto& shape_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("fast_gelu_output"), &type_info);
    Node& fast_gelu_node = graph.AddNode(graph.GenerateNodeName("GPT2Gelu"),
                                         "FastGelu",
                                         "fused GPT2Gelu subgraphs ",
                                         gelu_input_defs,
                                         {&shape_output}, {}, kMSDomain);

    // assign provider to this new node, provider should be same as the provider for old node.
    fast_gelu_node.SetExecutionProviderType(node.GetExecutionProviderType());

    // move input edges to node (first in list) across to the fast_gelu_node.
    // move output definitions and output edges from mul5_node (last in list) to fast_gelu_node.
    // remove all nodes.
    graph_utils::FinalizeNodeFusion(graph, nodes_to_fuse, fast_gelu_node);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
