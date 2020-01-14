// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

<<<<<<< HEAD
#include <deque>
#include "core/optimizer/initializer.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/graph/graph_utils.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {
// Todo: use graph_utils::NodeArgIsConstant once the Constant change (e.g. represented as an initalier only)
// in master is merged. We can do this way purely because Gelu subgraphs in
// BERT does not have inputs connected directly.
bool NodeArgIsConstant(const Graph& graph, const NodeArg& node_arg) {
  const onnx::TensorProto* initializer = nullptr;
  return graph.GetInitializedTensor(node_arg.Name(), initializer);
}

static void IsInputConstant(const Graph& graph, const Node& node, std::tuple<bool, NodeArg*, NodeArg*>& ret) {
  const auto& inputs = node.InputDefs();
  bool found_constant = false;
  NodeArg* gelu_non_const_input = nullptr;
  NodeArg* gelu_const_input = nullptr;
  for (auto& i : inputs) {
    if (NodeArgIsConstant(graph, *i)) {
      // Todo: check the constant for example be sqrt(2.0) or 1
      found_constant = true;
      gelu_const_input = const_cast<NodeArg*>(i);
    } else {
      gelu_non_const_input = const_cast<NodeArg*>(i);
    }
  }

  ret = std::make_tuple(found_constant, gelu_non_const_input, gelu_const_input);
}

Status GeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::deque<onnxruntime::NodeIndex> removed_nodes;
  std::deque<std::string> removed_initializers;

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
=======
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

Status GeluFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_div = graph.GetNode(node_index);
    if (p_div == nullptr)
      continue;  // we removed the node as part of an earlier fusion

    Node& div = *p_div;
    ORT_RETURN_IF_ERROR(Recurse(div, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(div, "Div", {7}) ||
        !graph_utils::IsSupportedProvider(div, GetCompatibleExecutionProviders()) ||
        div.GetOutputEdgesCount() != 1 ||
        !IsSupportedDataType(div)) {
      continue;
    }

    // Check second input is sqrt(2)
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *(div.InputDefs()[1]), static_cast<float>(M_SQRT2), true)) {
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
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *add_const_input_arg, 1.0f, true)) {
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
    if (!optimizer_utils::IsInitializerWithExpectedValue(graph, *mul_const_input_arg, 0.5f, true)) {
      continue;
    }

    const std::vector<NodeArg*> gelu_input_defs{div.MutableInputDefs()[0]};
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677
    Node& gelu_node = graph.AddNode(graph.GenerateNodeName("Gelu"),
                                    "Gelu",
                                    "fused Gelu subgraphs ",
                                    gelu_input_defs,
<<<<<<< HEAD
                                    gelu_output_defs);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    gelu_node.SetExecutionProviderType(node.GetExecutionProviderType());

    removed_nodes.push_front(node.Index());
    removed_nodes.push_front(erf_node.Index());
    removed_nodes.push_front(add_node.Index());
    removed_nodes.push_front(mul2_node->Index());
    removed_nodes.push_front(mul_node.Index());
    removed_initializers.push_front(std::get<2>(t)->Name());
    removed_initializers.push_front(std::get<2>(add_input_check)->Name());
    removed_initializers.push_front(std::get<2>(mul2_input_check)->Name());
  }

  // Have to remove node in reversed order for now to walk around the issue in RemoveNode
  for (onnxruntime::NodeIndex removed_node : removed_nodes) {
    graph.RemoveNode(removed_node);
  }

  for (auto& tensor_name : removed_initializers) {
    if (graph.GetConsumerNodes(tensor_name).empty()) {
      graph.RemoveInitializedTensor(tensor_name);
    }
  }

  if (!removed_nodes.empty() || !removed_initializers.empty()) {
=======
                                    {}, {}, kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    gelu_node.SetExecutionProviderType(div.GetExecutionProviderType());

    // move input edges to div (first in list) across to the gelu_node.
    // move output definitions and output edges from mul_node (last in list) to gelu_node.
    // remove all the other nodes.
    graph_utils::FinalizeNodeFusion(graph, {div, erf_node, add_node, mul2_node, mul_node}, gelu_node);

>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
