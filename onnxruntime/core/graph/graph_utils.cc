// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {

namespace graph_utils {
// fusion is only done for ONNX domain ops
bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       const std::string& op_type,
                                       ONNX_NAMESPACE::OperatorSetVersion version,
                                       const std::string& domain) {
  if (node.OpType() != op_type ||
      node.Op()->Deprecated() || node.Op()->SinceVersion() != version ||
      (!node.Domain().empty() && node.Domain() != domain)) {
    return false;
  }
  return true;
}

Status ForAllMutableSubgraphs(Graph& graph, std::function<Status(Graph&)> func) {
  Status status = Status::OK();

  for (auto& node : graph.Nodes()) {
    for (auto& attribute : node.GetAttributes()) {
      auto& name = attribute.first;
      auto& proto = attribute.second;

      // check if it has a subgraph
      if (proto.has_g()) {
        Graph* subgraph = node.GetMutableGraphAttribute(name);
        ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

        status = func(*subgraph);
        ORT_RETURN_IF_ERROR(status);

        // recurse
        status = ForAllMutableSubgraphs(*subgraph, func);
        ORT_RETURN_IF_ERROR(status);
      }
    }
  }

  return status;
}

Status ForAllSubgraphs(const Graph& graph, std::function<Status(const Graph&)> func) {
  Status status = Status::OK();

  for (auto& node : graph.Nodes()) {
    for (auto& attribute : node.GetAttributes()) {
      auto& name = attribute.first;
      auto& proto = attribute.second;

      // check if it has a subgraph
      if (proto.has_g()) {
        const Graph* subgraph = node.GetGraphAttribute(name);
        ORT_ENFORCE(subgraph, "Main Graph instance should have populated all subgraphs when being resolved.");

        status = func(*subgraph);
        ORT_RETURN_IF_ERROR(status);

        // recurse
        status = ForAllSubgraphs(*subgraph, func);
        ORT_RETURN_IF_ERROR(status);
      }
    }
  }

  return status;
}

bool IsSingleInSingleOutNode(const Node& node) {
  return node.GetInputEdgesCount() == 1 && node.GetOutputEdgesCount() == 1;
}

const ONNX_NAMESPACE::AttributeProto* GetNodeAttribute(
    const Node& node, const std::string& attr_name) {
  const auto& attrs = node.GetAttributes();
  const auto iter = attrs.find(attr_name);
  return iter == attrs.end() ? nullptr : &iter->second;
}

bool RemoveSingleInSingleOutNode(Graph& graph, Node& node) {
  if (!IsSingleInSingleOutNode(node)) {
    return false;
  }
  // Get input/output edges, nodes, and node args.
  const Node::EdgeEnd& input_edge = *node.InputEdgesBegin();
  const NodeIndex input_edge_node = input_edge.GetNode().Index();
  const int input_edge_dst_arg = input_edge.GetSrcArgIndex();
  const Node::EdgeEnd& output_edge = *node.OutputEdgesBegin();
  const NodeIndex output_edge_node = output_edge.GetNode().Index();
  const int output_edge_dst_arg = output_edge.GetDstArgIndex();

  // Remove output edge.
  graph.RemoveEdge(node.Index(), output_edge.GetNode().Index(),
                   output_edge.GetSrcArgIndex(), output_edge.GetDstArgIndex());

  // Remove node (this will remove the input edge too).
  graph.RemoveNode(node.Index());

  // Add new edge connecting the input with the output nodes directly.
  graph.AddEdge(input_edge_node, output_edge_node,
                input_edge_dst_arg, output_edge_dst_arg);

  return true;
}

bool HasGraphInput(const Graph& graph, const NodeArg* input) {
  const std::vector<const NodeArg*>& graph_inputs = graph.GetInputsIncludingInitializers();
  return std::find(graph_inputs.begin(), graph_inputs.end(), input) != graph_inputs.end();
}

bool IsConstantInputsNode(const Graph& graph, const Node& node) {
  if (node.GetInputEdgesCount() > 0) {
    return false;
  }
  const onnx::TensorProto* initializer = nullptr;
  for (const auto* input_def : node.InputDefs()) {
    // Important note: when an initializer appears in the graph's input, this input will not be considered constant,
    // because it can be overriden by the user at runtime. For constant folding to be applied, the initializer should not
    // appear in the graph's inputs (that is the only way to guarantee it will always be constant).
    if (!graph.GetInitializedTensor(input_def->Name(), initializer) || HasGraphInput(graph, input_def)) {
      return false;
    }
  }
  return true;
}

size_t RemoveNodeOutputEdges(Graph& graph, Node& node) {
  std::vector<std::tuple<NodeIndex, int, int>> edges_to_remove;
  for (auto it = node.OutputEdgesBegin(); it != node.OutputEdgesEnd(); ++it) {
    edges_to_remove.emplace_back(std::make_tuple(it->GetNode().Index(),
                                                 it->GetSrcArgIndex(),
                                                 it->GetDstArgIndex()));
  }
  for (auto& edge_to_remove : edges_to_remove) {
    graph.RemoveEdge(node.Index(),
                     std::get<0>(edge_to_remove),
                     std::get<1>(edge_to_remove),
                     std::get<2>(edge_to_remove));
  }

  return edges_to_remove.size();
}

}  // namespace graph_utils

}  // namespace onnxruntime
