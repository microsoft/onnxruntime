// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "orttraining/core/optimizer/fast_gelu_fusion.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// Gelu supports limited data types.
static std::vector<std::string> supported_data_types{"tensor(float16)", "tensor(float)", "tensor(double)"};

Status FastGeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node == nullptr)
      continue;

    Node& mul1_node = *p_node;
    ORT_RETURN_IF_ERROR(Recurse(mul1_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul1_node, "Mul", {7}) ||
        !graph_utils::IsSupportedProvider(mul1_node, GetCompatibleExecutionProviders()) ||
        mul1_node.GetOutputEdgesCount() != 1 ||
        !optimizer_utils::IsSupportedDataType(mul1_node, supported_data_types)) {
      continue;
    }

    // Check the const input is 0.044715
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul1_node.InputDefs()[1]), 0.044715f, true)) {
      continue;
    }

    Node& mul2_node = *graph.GetNode(mul1_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul2_node, "Mul", {7}) ||
        mul2_node.GetExecutionProviderType() != mul1_node.GetExecutionProviderType() ||
        mul2_node.GetOutputEdgesCount() != 1 ||
        !optimizer_utils::IsSupportedDataType(mul2_node, supported_data_types)) {
      continue;
    }

    Node& add1_node = *graph.GetNode(mul2_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add1_node, "Add", {7}) ||
        add1_node.GetExecutionProviderType() != mul1_node.GetExecutionProviderType() ||
        add1_node.GetOutputEdgesCount() != 1 ||
        !optimizer_utils::IsSupportedDataType(add1_node, supported_data_types)) {
      continue;
    }

    auto input_index = optimizer_utils::IndexOfNodeInput(add1_node, *mul2_node.MutableOutputDefs()[0]);
    const float one = 1.0f;
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(add1_node.InputDefs()[(input_index + 1) % 2]), one, true)) {
      continue;
    }

    Node& mul3_node = *graph.GetNode(add1_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul3_node, "Mul", {7}) ||
        mul3_node.GetExecutionProviderType() != mul1_node.GetExecutionProviderType() ||
        mul3_node.GetOutputEdgesCount() != 1 ||
        !optimizer_utils::IsSupportedDataType(mul3_node, supported_data_types)) {
      continue;
    }

    input_index = optimizer_utils::IndexOfNodeInput(mul3_node, *add1_node.MutableOutputDefs()[0]);
    if (input_index == -1) {
      continue;
    }

    Node& mul4_node = const_cast<Node&>(*graph_utils::GetInputNode(mul3_node, (input_index + 1) % 2));
    if (!(mul4_node.OpType().compare("Mul") == 0 && mul4_node.InputDefs()[0]->Name() == mul1_node.InputDefs()[0]->Name() &&
          optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul4_node.InputDefs()[1]), 0.7978845834732056f, true))) {
      continue;
    }

    Node& tanh_node = *graph.GetNode(mul3_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(tanh_node, "Tanh", {6}) ||
        tanh_node.GetExecutionProviderType() != mul1_node.GetExecutionProviderType() ||
        tanh_node.GetOutputEdgesCount() != 1 ||
        !optimizer_utils::IsSupportedDataType(tanh_node, supported_data_types)) {
      continue;
    }

    Node& add2_node = *graph.GetNode(tanh_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add2_node, "Add", {7}) ||
        add2_node.GetExecutionProviderType() != mul1_node.GetExecutionProviderType() ||
        add2_node.GetOutputEdgesCount() != 1 ||
        !optimizer_utils::IsSupportedDataType(add2_node, supported_data_types)) {
      continue;
    }

    input_index = optimizer_utils::IndexOfNodeInput(add2_node, *tanh_node.MutableOutputDefs()[0]);
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(add2_node.InputDefs()[(input_index + 1) % 2]), one, true)) {
      continue;
    }

    Node& mul5_node = *graph.GetNode(add2_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul5_node, "Mul", {7}) ||
        mul5_node.GetExecutionProviderType() != mul1_node.GetExecutionProviderType() ||
        !optimizer_utils::IsSupportedDataType(mul5_node, supported_data_types)) {
      continue;
    }

    input_index = optimizer_utils::IndexOfNodeInput(mul5_node, *add2_node.MutableOutputDefs()[0]);
    Node& mul6_node = const_cast<Node&>(*graph_utils::GetInputNode(mul5_node, (input_index + 1) % 2));
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul6_node, "Mul", {7}) ||
        mul6_node.GetExecutionProviderType() != mul1_node.GetExecutionProviderType() ||
        mul6_node.GetOutputEdgesCount() != 1 ||
        !optimizer_utils::IsSupportedDataType(mul6_node, supported_data_types)) {
      continue;
    }

    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *mul6_node.InputDefs()[1], 0.5f, true)) {
      continue;
    }

    const std::vector<NodeArg*> gelu_input_defs{mul1_node.MutableInputDefs()[0]};
    auto type_info = *mul1_node.MutableOutputDefs()[0]->TypeAsProto();  // copy
    auto& shape_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("gelu_output"), &type_info);

    Node& gelu_node = graph.AddNode(graph.GenerateNodeName("GPT2Gelu"),
                                    "FastGelu",
                                    "fused GPT2Gelu subgraphs ",
                                    gelu_input_defs,
                                    {&shape_output}, {}, kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    gelu_node.SetExecutionProviderType(mul1_node.GetExecutionProviderType());

    // move input edges to node (first in list) across to the gelu_node.
    // move output definitions and output edges from mul5_node (last in list) to gelu_node.
    // remove all nodes.
    graph_utils::FinalizeNodeFusion(graph,
                                    {mul1_node, mul2_node, add1_node, mul3_node, mul4_node, tanh_node,
                                     add2_node, mul6_node, mul5_node},
                                    gelu_node);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
