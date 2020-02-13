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
static std::vector<std::string> supported_data_types{"tensor(float16)", "tensor(float)"};

static bool CheckNode(const Node& node, const std::string op_name, const int32_t opset_version, const ProviderType provider,
  bool require_single_output=false){
  return graph_utils::IsSupportedOptypeVersionAndDomain(node, op_name, {opset_version}) &&
        node.GetExecutionProviderType() == provider &&
        optimizer_utils::IsSupportedDataType(node, supported_data_types) &&
        (!require_single_output || node.GetOutputEdgesCount() == 1);
}

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

    int32_t input_index = -1;
    const float mul_val = 0.044715f;
    for (auto i = 0; i < 2; i++) {
      if (optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul1_node.InputDefs()[i]), mul_val, true)){
        input_index = i;
        break;
      }
    }

    if (input_index == -1) continue;

    auto p_mul1_input_node = graph_utils::GetInputNode(mul1_node, (input_index + 1) % 2);
    NodeArg* gelu_input_arg = mul1_node.InputDefs()[(input_index + 1) % 2];
    NodeArg* bias_arg = nullptr;
    if (p_mul1_input_node != nullptr) {
      Node& mul1_input_node = const_cast<Node&>(*p_mul1_input_node);
       CheckNode(, "Add", 7, 
      mul1_node.GetExecutionProviderType(), true)) {
        gelu_input_arg = mul1_input_node->MutableInputDefs()[0];
        bias_arg = p_mul1_input_node->MutableInputDefs()[1];
    }
  
    Node& mul2_node = *graph.GetNode(mul1_node.OutputNodesBegin()->Index());
    input_index = optimizer_utils::IndexOfNodeInput(mul2_node, *mul1_node.MutableOutputDefs()[0]);
    if (!CheckNode(mul2_node, "Mul", 7,  mul1_node.GetExecutionProviderType(), true) ||
        mul2_node.MutableInputDefs()[(input_index + 1) % 2]->Name() != gelu_input_arg->Name()) {
      continue;
    }


    Node& add1_node = *graph.GetNode(mul2_node.OutputNodesBegin()->Index());
    input_index = optimizer_utils::IndexOfNodeInput(add1_node, *mul2_node.MutableOutputDefs()[0]);
    const float one = 1.0f;
    if (!CheckNode(add1_node, "Add", 7, mul1_node.GetExecutionProviderType(), true) ||
        !optimizer_utils::IsInitializerWithExpectedValue(graph, *(add1_node.InputDefs()[(input_index + 1) % 2]), one, true)) {
      continue;
    }


    Node& mul3_node = *graph.GetNode(add1_node.OutputNodesBegin()->Index());
    if (!CheckNode(mul3_node, "Mul", 7, mul1_node.GetExecutionProviderType(), true)) {
      continue;
    }


    input_index = optimizer_utils::IndexOfNodeInput(mul3_node, *add1_node.MutableOutputDefs()[0]);
    Node& mul4_node = const_cast<Node&>(*graph_utils::GetInputNode(mul3_node, (input_index + 1) % 2));
    if (!CheckNode(mul4_node, "Mul", 7, mul1_node.GetExecutionProviderType(), true)) {
      continue;
    }

    input_index = -1;
    const float mul4_val = 0.7978845834732056f;
    for (auto i = 0; i < 2; i++) {
      if (optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul4_node.InputDefs()[i]), mul4_val, true)){
        input_index = i;
        break;
      }
    }

    if (input_index == -1 || mul4_node.InputDefs()[(input_index + 1) % 2]->Name() != gelu_input_arg->Name())
        continue;


    Node& tanh_node = *graph.GetNode(mul3_node.OutputNodesBegin()->Index());
    if (!CheckNode(tanh_node, "Tanh", 6, mul1_node.GetExecutionProviderType(), true)) {
      continue;
    }


    Node& add2_node = *graph.GetNode(tanh_node.OutputNodesBegin()->Index());
    if (!CheckNode(add2_node, "Add", 7, mul1_node.GetExecutionProviderType(), true)) {
      continue;
    }

    input_index = optimizer_utils::IndexOfNodeInput(add2_node, *tanh_node.MutableOutputDefs()[0]);
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(add2_node.InputDefs()[(input_index + 1) % 2]), one, true)) {
      continue;
    }


    Node& mul5_node = *graph.GetNode(add2_node.OutputNodesBegin()->Index());
    if (!CheckNode(mul5_node, "Mul", 7, mul1_node.GetExecutionProviderType(), true)) {
      continue;
    }


    input_index = optimizer_utils::IndexOfNodeInput(mul5_node, *add2_node.MutableOutputDefs()[0]);
    Node& mul6_node = const_cast<Node&>(*graph_utils::GetInputNode(mul5_node, (input_index + 1) % 2));
    if (!CheckNode(mul6_node, "Mul", 7, mul1_node.GetExecutionProviderType(), false)) {
      continue;
    }

    const float mul6_val = 0.5f;
    input_index = -1;
    for (auto i = 0; i < 2; i++) {
      if (optimizer_utils::IsInitializerWithExpectedValue(graph, *(mul6_node.InputDefs()[i]), mul6_val, true)){
        input_index = i;
        break;
      }
    }

    if (input_index == -1 || mul6_node.InputDefs()[(input_index + 1) % 2]->Name() != gelu_input_arg->Name())
      continue;


    std::vector<NodeArg*> gelu_input_defs{mul1_node.MutableInputDefs()[0]};
    if (bias_arg != nullptr)
      gelu_input_defs.push_back(bias_arg);

    auto type_info = *mul1_node.MutableOutputDefs()[0]->TypeAsProto();
    auto& shape_output = graph.GetOrCreateNodeArg(graph.GenerateNodeArgName("fast_gelu_output"), &type_info);

    Node& fast_gelu_node = graph.AddNode(graph.GenerateNodeName("GPT2Gelu"),
                                         "FastGelu",
                                         "fused GPT2Gelu subgraphs ",
                                         gelu_input_defs,
                                         {&shape_output}, {}, kMSDomain);

    // assign provider to this new node, provider should be same as the provider for old node.
    fast_gelu_node.SetExecutionProviderType(mul1_node.GetExecutionProviderType());

    // move input edges to node (first in list) across to the fast_gelu_node.
    // move output definitions and output edges from mul5_node (last in list) to fast_gelu_node.
    // remove all nodes.
    graph_utils::FinalizeNodeFusion(graph,
                                    {mul1_node, mul2_node, add1_node, mul3_node, mul4_node, tanh_node,
                                     add2_node, mul6_node, mul5_node},
                                    fast_gelu_node);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
