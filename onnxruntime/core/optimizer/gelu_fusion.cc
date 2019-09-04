// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/initializer.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

static void IsInputConstant(const Graph& graph, const Node& node, std::tuple<bool, NodeArg*, NodeArg*>& ret) {
  const auto& inputs = node.InputDefs();
  bool found_constant = false;
  NodeArg* gelu_non_const_input = nullptr;
  NodeArg* gelu_const_input = nullptr;
  for (auto& i : inputs) {
    if (graph_utils::NodeArgIsConstant(graph, *i)) {
      // Todo: check the constant for example be sqrt(2.0) or 1
      found_constant = true;
      gelu_const_input = const_cast<NodeArg*>(i);
    } else {
      gelu_non_const_input = const_cast<NodeArg*>(i);
    }
  }

  ret = std::make_tuple(found_constant, gelu_non_const_input, gelu_const_input);
}

static bool HasConsumers(Graph& graph, const NodeArg* arg) {
  for (auto& node : graph.Nodes()) {
    auto ret = std::find(node.MutableInputDefs().begin(), node.MutableInputDefs().end(), arg);
    if (ret != node.MutableInputDefs().end()) {
      return true;
    }

    ret = std::find(node.MutableImplicitInputDefs().begin(), node.MutableImplicitInputDefs().end(), arg);
    if (ret != node.MutableImplicitInputDefs().end()) {
      return true;
    }
  }
  return false;
}

Status GeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::deque<onnxruntime::NodeIndex> removed_nodes;
  std::deque<NodeArg*> removed_initializers;

  for (auto node_index : node_topology_list) {
    auto& node = *graph.GetNode(node_index);
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(node, "Div", {7}) ||
        !graph_utils::IsSupportedProvider(node, GetCompatibleExecutionProviders()) ||
        node.GetOutputEdgesCount() != 1) {
      continue;
    }
    std::tuple<bool, NodeArg*, NodeArg*> t;
    IsInputConstant(graph, node, t);
    if (!std::get<0>(t) || std::get<1>(t) == nullptr || std::get<2>(t) == nullptr) {
      continue;
    }

    auto erf_node_itr = node.OutputNodesBegin();
    if (erf_node_itr == node.OutputNodesEnd()) {
      continue;
    }

    const Node& erf_node = (*erf_node_itr);
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(erf_node, "Erf", {9}) ||
        erf_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        erf_node.GetOutputEdgesCount() != 1) {
      continue;
    }

    auto add_node_itr = erf_node.OutputNodesBegin();
    if (add_node_itr == erf_node.OutputNodesEnd()) {
      continue;
    }

    const Node& add_node = (*add_node_itr);
    std::tuple<bool, NodeArg*, NodeArg*> add_input_check;
    IsInputConstant(graph, add_node, add_input_check);
    if (!std::get<0>(add_input_check) || std::get<1>(add_input_check) == nullptr || std::get<2>(add_input_check) == nullptr) {
      continue;
    }
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(add_node, "Add", {7}) ||
        add_node.GetExecutionProviderType() != node.GetExecutionProviderType() ||
        add_node.GetOutputEdgesCount() != 1) {
      continue;
    }

    auto mul_node_itr = add_node.OutputNodesBegin();
    if (mul_node_itr == add_node.OutputNodesEnd()) {
      continue;
    }

    const Node& mul_node = *(mul_node_itr);
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7}) ||
        mul_node.GetExecutionProviderType() != node.GetExecutionProviderType()) {
      continue;
    }

    const Node* mul2_node = nullptr;
    for (auto iter = mul_node.InputNodesBegin(); iter != mul_node.InputNodesEnd(); ++iter) {
      if ((*iter).OpType().compare("Mul") == 0) {
        // find the other input node of Mul
        mul2_node = &(*iter);
        break;
      }
    }
    if (mul2_node == nullptr) {
      continue;
    }

    std::tuple<bool, NodeArg*, NodeArg*> mul2_input_check;
    IsInputConstant(graph, *mul2_node, mul2_input_check);
    if (!std::get<0>(mul2_input_check) || std::get<1>(mul2_input_check) == nullptr || std::get<2>(mul2_input_check) == nullptr) {
      continue;
    }
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*mul2_node, "Mul", {7}) ||
        mul2_node->GetExecutionProviderType() != node.GetExecutionProviderType() ||
        mul2_node->GetOutputEdgesCount() != 1) {
      continue;
    }

    const std::vector<NodeArg*> gelu_input_defs{std::get<1>(t)};
    const std::vector<NodeArg*> gelu_output_defs{const_cast<NodeArg*>(mul_node.OutputDefs()[0])};
    Node& gelu_node = graph.AddNode(graph.GenerateNodeName("Gelu"),
                                    "Gelu",
                                    "fused Gelu subgraphs ",
                                    gelu_input_defs,
                                    gelu_output_defs, {}, kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    gelu_node.SetExecutionProviderType(node.GetExecutionProviderType());

    removed_nodes.push_front(node.Index());
    removed_nodes.push_front(erf_node.Index());
    removed_nodes.push_front(add_node.Index());
    removed_nodes.push_front(mul2_node->Index());
    removed_nodes.push_front(mul_node.Index());
    removed_initializers.push_front(std::get<2>(t));
    removed_initializers.push_front(std::get<2>(add_input_check));
    removed_initializers.push_front(std::get<2>(mul2_input_check));
  }

  // Have to remove node in reversed order for now to walk around the issue in RemoveNode
  for (onnxruntime::NodeIndex removed_node : removed_nodes) {
    graph.RemoveNode(removed_node);
  }

  for (auto& tensor_arg : removed_initializers) {
    if (!HasConsumers(graph, tensor_arg)) {
      graph.RemoveInitializedTensor(tensor_arg->Name());
    }
  }

  if (!removed_nodes.empty() || !removed_initializers.empty()) {
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
