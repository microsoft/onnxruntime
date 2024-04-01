// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/shape_input_merge.h"

#include "core/graph/graph_utils.h"

namespace onnxruntime {

namespace {
std::string GetShapeString(const NodeArg* input_arg) {
  auto shape = input_arg->Shape();
  if (!shape) return "";
  std::stringstream ss;
  ss << "[";
  for (int i = 0; i < shape->dim_size(); ++i) {
    if (i != 0) ss << ",";
    auto dim = shape->dim(i);
    if (dim.has_dim_value()) {
      ss << std::to_string(dim.dim_value());
    } else if (dim.has_dim_param()) {
      ss << "'" << dim.dim_param() << "'";
    } else {
      return "";
    }
  }
  ss << "]";
  return ss.str();
}

}  // namespace

Status ShapeInputMerge::ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();
  InlinedHashMap<std::string, InlinedVector<Node*>> input_hash_to_nodes;
  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (!p_node) continue;  // we removed the node as part of an earlier fusion
    ORT_RETURN_IF_ERROR(Recurse(*p_node, modified, graph_level, logger));
    if (!graph_utils::IsSupportedOptypeVersionAndDomain(*p_node, "Shape", {1, 13, 15, 19, 21}) ||
        !graph_utils::IsSupportedProvider(*p_node, GetCompatibleExecutionProviders())) {
      continue;
    }
    std::string shape_str = GetShapeString(p_node->InputDefs()[0]);
    if (shape_str.empty()) continue;
    if (input_hash_to_nodes.find(shape_str) == input_hash_to_nodes.end()) {
      input_hash_to_nodes[shape_str] = InlinedVector<Node*>();
    }
    input_hash_to_nodes[shape_str].emplace_back(p_node);
  }

  // All Shape nodes are processed in topological order, so we can safely merge the inputs to the first node's input.
  for (auto& kv : input_hash_to_nodes) {
    if (kv.second.size() < 2) continue;
    NodeArg* first_input_arg = kv.second[0]->MutableInputDefs()[0];
    bool is_first_input_arg_graph_input = graph.IsInputsIncludingInitializers(first_input_arg);
    for (size_t i = 1; i < kv.second.size(); ++i) {
      Node* p_node = kv.second[i];
      const NodeArg* input_arg = p_node->InputDefs()[0];
      if (p_node->InputDefs()[0]->Name() == first_input_arg->Name()) continue;
      if (!graph.IsInputsIncludingInitializers(input_arg)) {
        const Node::EdgeEnd& input_edge = *p_node->InputEdgesBegin();
        graph.RemoveEdge(input_edge.GetNode().Index(), p_node->Index(), input_edge.GetSrcArgIndex(), 0);
      }
      graph_utils::ReplaceNodeInput(*p_node, 0, *first_input_arg);
      if (!is_first_input_arg_graph_input) {
        const Node::EdgeEnd& first_input_edge = *kv.second[0]->InputEdgesBegin();
        graph.AddEdge(first_input_edge.GetNode().Index(), p_node->Index(), first_input_edge.GetSrcArgIndex(), 0);
      }
      modified = true;
    }
  }

  return Status::OK();
}

}  // namespace onnxruntime
