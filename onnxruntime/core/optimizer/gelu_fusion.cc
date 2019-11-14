// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "float.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;
namespace onnxruntime {

// Gelu supports limited data types.
static std::vector<std::string> supported_data_types{"tensor(float16)", "tensor(float)", "tensor(double)"};

static bool IsSupportedDataType(const Node& node) {
  for (const auto& input_arg : node.InputDefs()) {
    if (std::find(supported_data_types.begin(), supported_data_types.end(),
                  *(input_arg->Type())) == supported_data_types.end()) {
      return false;
    }
  }
  return true;
}

Status GeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_div = graph.GetNode(node_index);
    if (p_div == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& div = *p_div;
    ORT_RETURN_IF_ERROR(Recurse(div, modified, graph_level));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(div, "Div", {7}) ||
        !graph_utils::IsSupportedProvider(div, GetCompatibleExecutionProviders()) ||
        div.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(div)) {
      continue;
    }

    // Check second input is sqrt(2)
    if (!optimizer_utils::CheckConstantInput(graph, *(div.InputDefs()[1]), static_cast<float>(M_SQRT2))) {
      continue;
    }

    Node& erf_node = *graph.GetNode(div.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(erf_node, "Erf", {9}) ||
        erf_node.GetExecutionProviderType() != div.GetExecutionProviderType() ||
        erf_node.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(erf_node)) {
      continue;
    }

    Node& add_node = *graph.GetNode(erf_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7}) ||
        add_node.GetExecutionProviderType() != div.GetExecutionProviderType() ||
        add_node.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(add_node)) {
      continue;
    }

    // Check the other input node(e.g. not of type Erf) is 1.0f.
    const Node& add_first_input_node = *(add_node.InputNodesBegin());
    int add_const_input_index = 0;
    if (add_first_input_node.OpType().compare("Erf") == 0) {
      add_const_input_index = 1;
    }
    const auto& add_const_input_arg = add_node.InputDefs()[add_const_input_index];
    if (!optimizer_utils::CheckConstantInput(graph, *add_const_input_arg, 1.0f)) {
      continue;
    }

    Node& mul_node = *graph.GetNode(add_node.OutputNodesBegin()->Index());
    // note: output edges count doesn't matter as the new Gelu node will produce outputs with the same names
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7}) ||
        mul_node.GetExecutionProviderType() != div.GetExecutionProviderType() ||
        !IsSupportedDataType(mul_node)) {
      continue;
    }

    const Node* p_mul2_node = nullptr;
    for (auto iter = mul_node.InputNodesBegin(); iter != mul_node.InputNodesEnd(); ++iter) {
      if ((*iter).OpType().compare("Mul") == 0) {
        // find the other input node of Mul
        p_mul2_node = &(*iter);
        break;
      }
    }
    if (p_mul2_node == nullptr) {
      continue;
    }

    Node& mul2_node = *graph.GetNode(p_mul2_node->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul2_node, "Mul", {7}) ||
        mul2_node.GetExecutionProviderType() != div.GetExecutionProviderType() ||
        mul2_node.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(mul2_node)) {
      continue;
    }

    // Check the other input node(e.g. not of type Add) is 0.5f.
    int mul_const_input_index = 0;
    if (mul2_node.InputDefs()[0]->Name() == div.MutableInputDefs()[0]->Name()) {
      mul_const_input_index = 1;
    }

    const auto& mul_const_input_arg = mul2_node.InputDefs()[mul_const_input_index];
    if (!optimizer_utils::CheckConstantInput(graph, *mul_const_input_arg, 0.5f)) {
      continue;
    }

    const std::vector<NodeArg*> gelu_input_defs{div.MutableInputDefs()[0]};
    Node& gelu_node = graph.AddNode(graph.GenerateNodeName("Gelu"),
                                    "Gelu",
                                    "fused Gelu subgraphs ",
                                    gelu_input_defs,
                                    {}, {}, kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    gelu_node.SetExecutionProviderType(div.GetExecutionProviderType());

    // move input edges to div (first in list) across to the gelu_node.
    // move output definitions and output edges from mul_node (last in list) to gelu_node.
    // remove all the other nodes.
    graph_utils::FinalizeNodeFusion(graph, {div, erf_node, add_node, mul2_node, mul_node}, gelu_node);

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
