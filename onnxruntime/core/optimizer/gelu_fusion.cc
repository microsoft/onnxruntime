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
/*
     This function fuses subgraph like the following into one Gelu node.
     Subgraph pattern 1:
                   +-------Mul(0.5)---------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul ==>
                          (B=1.4142...)        (1)

      Subgraph pattern 2:
                   +------------------------------------+
                   |                                    |
                   |                                    v
                [root] --> Div -----> Erf  --> Add --> Mul -->Mul ==>
                          (B=1.4142...)        (1)            (0.5)

       After Fusion:
                [root]--> Gelu ==>
*/
Status GeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_div = graph.GetNode(node_index);
    if (p_div == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& div = *p_div;
    ORT_RETURN_IF_ERROR(Recurse(div, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(div, "Div", {7, 13, 14}) ||
        !graph_utils::IsSupportedProvider(div, GetCompatibleExecutionProviders()) ||
        !optimizer_utils::CheckOutputEdges(graph, div, 1) ||
        !IsSupportedDataType(div)) {
      continue;
    }

    // Check second input is sqrt(2)
    // Some Bert model uses this approximation of SQRT2 in the Gelu function
    float approximated_sqrt_two = 1.4142099618911743f;
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(div.InputDefs()[1]), approximated_sqrt_two, true) &&
        !optimizer_utils::IsInitializerWithExpectedValue(graph, *(div.InputDefs()[1]), static_cast<float>(M_SQRT2), true)) {
      continue;
    }

    Node& erf_node = *graph.GetNode(div.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(erf_node, "Erf", {9, 13}) ||
        erf_node.GetExecutionProviderType() != div.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, erf_node, 1) ||
        !IsSupportedDataType(erf_node)) {
      continue;
    }

    Node& add_node = *graph.GetNode(erf_node.OutputNodesBegin()->Index());
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7, 13, 14}) ||
        add_node.GetExecutionProviderType() != div.GetExecutionProviderType() ||
        !optimizer_utils::CheckOutputEdges(graph, add_node, 1) ||
        !IsSupportedDataType(add_node)) {
      continue;
    }

    // Check the other input node (e.g. not the Erf) is 1.0f.
    bool is_erf_first_input = (add_node.InputDefs()[0]->Name() == erf_node.MutableOutputDefs()[0]->Name());
    const auto& add_const_input_arg = add_node.InputDefs()[is_erf_first_input ? 1 : 0];
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *add_const_input_arg, 1.0f, true)) {
      continue;
    }

    Node& mul_node = *graph.GetNode(add_node.OutputNodesBegin()->Index());
    // note: output edges count doesn't matter as the new Gelu node will produce outputs with the same names
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7, 13, 14}) ||
        mul_node.GetExecutionProviderType() != div.GetExecutionProviderType() ||
        !IsSupportedDataType(mul_node)) {
      continue;
    }

    bool is_pattern_1 = true;
    const Node* p_mul2_node = graph_utils::FirstParentByType(mul_node, "Mul");
    if (p_mul2_node != nullptr) {
      // Match subgraph pattern 1
      Node& mul2_node = *graph.GetNode(p_mul2_node->Index());
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul2_node, "Mul", {7, 13, 14}) ||
          mul2_node.GetExecutionProviderType() != div.GetExecutionProviderType() ||
          !optimizer_utils::CheckOutputEdges(graph, mul2_node, 1) ||
          !IsSupportedDataType(mul2_node)) {
        continue;
      }

      // One input of mul2_node shall be the subgraph input
      auto root_index = optimizer_utils::IndexOfNodeInput(*p_mul2_node, *div.InputDefs()[0]);
      if (root_index < 0)
        continue;

      // Check the other input node is 0.5f.
      int mul_const_input_index = (root_index == 0 ? 1 : 0);

      const auto& mul_const_input_arg = mul2_node.InputDefs()[mul_const_input_index];
      if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *mul_const_input_arg, 0.5f, true)) {
        continue;
      }
    } else {
      is_pattern_1 = false;

      // Match subgraph pattern 2
      if (!optimizer_utils::CheckOutputEdges(graph, mul_node, 1)) {
        continue;
      }

      // Another input of Mul node shall be the subgraph input.
      auto root_index = optimizer_utils::IndexOfNodeInput(mul_node, *div.InputDefs()[0]);
      if (root_index < 0)
        continue;

      Node& mul2_node = *graph.GetNode(mul_node.OutputNodesBegin()->Index());
      if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul2_node, "Mul", {7, 13, 14}) ||
          mul_node.GetExecutionProviderType() != div.GetExecutionProviderType() ||
          !IsSupportedDataType(mul_node)) {
        continue;
      }

      int mul_const_input_index = 0;
      if (mul2_node.InputDefs()[0]->Name() == mul_node.MutableOutputDefs()[0]->Name()) {
        mul_const_input_index = 1;
      }
      const auto& mul_const_input_arg = mul2_node.InputDefs()[mul_const_input_index];
      if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *mul_const_input_arg, 0.5f, true)) {
        continue;
      }

      p_mul2_node = &mul2_node;
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
    Node& mul2 = *graph.GetNode(p_mul2_node->Index());
    if (is_pattern_1) {
      graph_utils::FinalizeNodeFusion(graph, {div, erf_node, add_node, mul2, mul_node}, gelu_node);
    } else {
      graph_utils::FinalizeNodeFusion(graph, {div, erf_node, add_node, mul_node, mul2}, gelu_node);
    }

    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
