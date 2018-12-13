// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_utils.h"
#include "core/graph/graph.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {

namespace graph_edit_utils {
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

bool IsConstantInputsNode(const Graph& graph, const Node& node) {
  if (node.GetInputEdgesCount() > 0) {
    return false;
  }
  const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
  for (const auto* input_def : node.InputDefs()) {
    if (!graph.GetInitializedTensor(input_def->Name(), initializer)) {
      return false;
    }
  }
  return true;
}

Status BuildSubgraph(const Graph& graph,
                     const std::vector<onnxruntime::NodeIndex>& subgraph_nodes,
                     Graph& subgraph) {
  // Add nodes and initializers to subgraph.
  // TODO Can we directly copy the node instead of re-creating it?
  for (auto& node_index : subgraph_nodes) {
    auto node = graph.GetNode(node_index);
    std::vector<onnxruntime::NodeArg*> inputs, outputs;
    const ONNX_NAMESPACE::TensorProto* initializer = nullptr;
    for (auto input : node->InputDefs()) {
      auto& n_input = subgraph.GetOrCreateNodeArg(input->Name(), input->TypeAsProto());
      inputs.push_back(&n_input);
      if (graph.GetInitializedTensor(input->Name(), initializer)) {
        subgraph.AddInitializedTensor(*initializer);
      }
    }
    for (auto output : node->OutputDefs()) {
      auto& n_output = subgraph.GetOrCreateNodeArg(output->Name(), output->TypeAsProto());
      outputs.push_back(&n_output);
    }
    subgraph.AddNode(node->Name(), node->OpType(), node->Description(),
                     inputs, outputs, &node->GetAttributes(), node->Domain());
  }

  //TODO Check if we don't need to resolve the graph.
  ORT_ENFORCE(subgraph.Resolve().IsOK());

  return Status::OK();
}

int RemoveNodeOutputEdges(const Graph& graph,
                          const Node& node) {
  (void)graph, node;
  // Implement the method.
  //// Remove the output edges of the constant node.
  //std::vector<onnxruntime::NodeIndex> edge_nodes_to_remove;
  //for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
  //  edge_nodes_to_remove.push_back((*it).Index());
  //}

  //const auto* node_out_arg = node.OutputDefs()[fetch_idx];
  //for (auto& edge_node_idx : edge_nodes_to_remove) {
  //  (void)edge_node_idx, node_out_arg;
  //  //graph.RemoveEdge(node.Index(), edge_node_idx, *node_out_arg);
  //}
  return 0;
}

}  // namespace graph_edit_utils

}  // namespace onnxruntime
