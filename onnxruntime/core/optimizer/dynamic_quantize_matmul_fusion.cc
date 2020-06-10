// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dynamic_quantize_matmul_fusion.h"

#include "core/optimizer/initializer.h"
#include "core/graph/graph_utils.h"
#include <deque>

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;
namespace onnxruntime {

/**
DynamicQuantizeMatMulFusion will fuse subgraph like below into DynamicQuantizeMatMul:
    (input)
       |
       v
DynamicQuantizeLinear --------+
       |                      |
       v                      v
MatMulInteger (B const)      Mul (B const)
       |                      |
       v                      v
     Cast ------------------>Mul
                              |
                              v
                           (output)
*/
Status DynamicQuantizeMatMulFusion::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  std::vector<std::reference_wrapper<Node>> nodes_to_remove;

  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (nullptr == node_ptr)
      continue;  // node was removed

    auto& mul_node = *node_ptr;

    ORT_RETURN_IF_ERROR(Recurse(mul_node, modified, graph_level, logger));

    if (!graph_utils::IsSupportedOptypeVersionAndDomain(mul_node, "Mul", {7}) ||
        !graph_utils::IsSupportedProvider(mul_node, GetCompatibleExecutionProviders())) {
      continue;
    }

    // Left Parents path
    std::vector<graph_utils::EdgeEndToMatch> left_parent_path{
        {0, 0, "Cast", {6, 9}, kOnnxDomain},
        {0, 0, "MatMulInteger", {10}, kOnnxDomain},
        {0, 0, "DynamicQuantizeLinear", {11}, kOnnxDomain}};

    std::vector<graph_utils::EdgeEndToMatch> right_parent_path{
        {0, 1, "Mul", {7}, kOnnxDomain},
        {1, 0, "DynamicQuantizeLinear", {11}, kOnnxDomain}};

    std::vector<const Node::EdgeEnd*> left_edges;
    std::vector<const Node::EdgeEnd*> right_edges;
    if (!graph_utils::FindPath(mul_node, true, left_parent_path, left_edges, logger) ||
        !graph_utils::FindPath(mul_node, true, right_parent_path, right_edges, logger)) {
      continue;
    }

    Node& cast_node = const_cast<Node&>(left_edges[0]->GetNode());
    Node& matmulinteger_node = const_cast<Node&>(left_edges[1]->GetNode());
    Node& dql_node_left = const_cast<Node&>(left_edges[2]->GetNode());

    Node& mul_node_right = const_cast<Node&>(right_edges[0]->GetNode());
    Node& dql_node_right = const_cast<Node&>(right_edges[1]->GetNode());

    // Check if left DynamicQuantizeLinear is same as right DynamicQuantizeLinear
    if (dql_node_left.Index() != dql_node_right.Index()) {
      continue;
    }

    // Check Nodes' Edges count and Nodes' outputs are not in Graph output
    if (cast_node.GetOutputEdgesCount() != 1 ||
        matmulinteger_node.GetOutputEdgesCount() != 1 ||
        mul_node_right.GetOutputEdgesCount() != 1 ||
        dql_node_left.GetOutputEdgesCount() != 3 ||
        !graph.GetNodeOutputsInGraphOutputs(cast_node).empty() ||
        !graph.GetNodeOutputsInGraphOutputs(matmulinteger_node).empty() ||
        !graph.GetNodeOutputsInGraphOutputs(mul_node_right).empty() ||
        !graph.GetNodeOutputsInGraphOutputs(dql_node_left).empty()) {
      continue;
    }

    const NodeArg& matmulinteger_B = *(matmulinteger_node.InputDefs()[1]);
    if (!graph_utils::IsInitializer(graph, matmulinteger_B.Name(), true)) {
      continue;
    }

    const NodeArg& mul_right_B = *(mul_node_right.InputDefs()[1]);
    if (!graph_utils::IsInitializer(graph, mul_right_B.Name(), true)) {
      continue;
    }

    std::vector<NodeArg*> input_defs{dql_node_left.MutableInputDefs()[0],
                                     matmulinteger_node.MutableInputDefs()[1],
                                     mul_node_right.MutableInputDefs()[1]};

    if (matmulinteger_node.InputDefs().size() == 4) {
      const NodeArg& matmulinteger_B_zp = *(matmulinteger_node.InputDefs()[3]);
      if (!graph_utils::IsInitializer(graph, matmulinteger_B_zp.Name(), true)) {
        continue;
      }
      input_defs.push_back(matmulinteger_node.MutableInputDefs()[3]);
    }

    Node& fused_node = graph.AddNode(graph.GenerateNodeName("DynamicQuantizeMatMul"),
                                     "DynamicQuantizeMatMul",
                                     "fused DynamicQuantizeMatMul",
                                     input_defs,
                                     mul_node.MutableOutputDefs(),
                                     nullptr,
                                     kMSDomain);

    // Assign provider to this new node. Provider should be same as the provider for old node.
    fused_node.SetExecutionProviderType(mul_node.GetExecutionProviderType());

    nodes_to_remove.push_back(dql_node_left);
    nodes_to_remove.push_back(matmulinteger_node);
    nodes_to_remove.push_back(cast_node);
    nodes_to_remove.push_back(mul_node_right);
    nodes_to_remove.push_back(mul_node);
  }

  for (const auto& node : nodes_to_remove) {
    graph_utils::RemoveNodeOutputEdges(graph, node);
    graph.RemoveNode(node.get().Index());
  }

  modified = true;

  return Status::OK();
}
}  // namespace onnxruntime
