// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph.h"

namespace onnxruntime {

namespace graph_utils {

bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       const std::string& op_type,
                                       ONNX_NAMESPACE::OperatorSetVersion version,
                                       const std::string& domain = kOnnxDomainAlias);

/* Returns true if the execution provider assigned to current node is present in the compatible providers list
 * or if the compatible_providers list is empty
 */
bool IsSupportedProvider(const Node& node,
                         const std::unordered_set<std::string>& compatible_providers);

/** Check whether the node has a single input and a single output. */
bool IsSingleInSingleOutNode(const Node& node);

/** Returns true if the graph has the given input.*/
bool IsGraphInput(const Graph& graph, const NodeArg* input);

/** Checks if the given node has only constant inputs (initializers). */
bool AllNodeInputsAreConstant(const Graph& graph, const Node& node);

/** Get the name of the incoming NodeArg with the specified index for the given node. */
const std::string& GetNodeInputName(const Node& node, int index);

/** Get the name of the outgoing NodeArg with the specified index for the given node. */
const std::string& GetNodeOutputName(const Node& node, int index);

/** Return the attribute of a Node with a given name. */
const ONNX_NAMESPACE::AttributeProto* GetNodeAttribute(const Node& node, const std::string& attr_name);

/** Retrieve the values for a repeated attribute of a node and place them to the values vector. */
template <typename T>
bool GetRepeatedNodeAttributeValues(const Node& node,
                                    const std::string& attr_name,
                                    std::vector<T>& values) {
  const auto* attr = graph_utils::GetNodeAttribute(node, attr_name);
  if (attr) {
    values = ONNX_NAMESPACE::RetrieveValues<T>(*attr);
    return true;
  } else {
    return false;
  }
}

Status ForAllMutableSubgraphs(Graph& main_graph, std::function<Status(Graph&)> func);
Status ForAllSubgraphs(const Graph& main_graph, std::function<Status(const Graph&)> func);

/** Remove the given single-input Node from the Graph. The single input might be either
    another node or an initializer.*/
bool RemoveSingleInputNode(Graph& graph, Node& node);

/** Remove all output edges from the given Node of the Graph. 
    This should probably be elevated to the Graph API eventually. */
size_t RemoveNodeOutputEdges(Graph& graph, Node& node);

struct GraphEdge {
  NodeIndex src_node;
  NodeIndex dst_node;
  int src_arg_index;
  int dst_arg_index;
  std::string arg_name;

  GraphEdge(NodeIndex src_node, NodeIndex dst_node,
            int src_arg_index, int dst_arg_index, const std::string& arg_name) : src_node(src_node),
                                                                                 dst_node(dst_node),
                                                                                 src_arg_index(src_arg_index),
                                                                                 dst_arg_index(dst_arg_index),
                                                                                 arg_name(arg_name) {}

  // Constructs a GraphEdge given a node, an edge_end, and a boolean for the edge direction.
  GraphEdge(const Node& node, const Node::EdgeEnd& edge_end, bool is_input_edge) {
    is_input_edge
        ? init(edge_end.GetNode().Index(), node.Index(), edge_end.GetSrcArgIndex(),
               edge_end.GetDstArgIndex(), GetNodeInputName(node, edge_end.GetDstArgIndex()))
        : init(node.Index(), edge_end.GetNode().Index(), edge_end.GetSrcArgIndex(),
               edge_end.GetDstArgIndex(), GetNodeOutputName(node, edge_end.GetSrcArgIndex()));
  }

  void init(NodeIndex src_node_idx, NodeIndex dst_node_idx,
            int src_arg_idx, int dst_arg_idx, const std::string& name) {
    src_node = src_node_idx;
    dst_node = dst_node_idx;
    src_arg_index = src_arg_idx;
    dst_arg_index = dst_arg_idx;
    arg_name = name;
  }
};

}  // namespace graph_utils

}  // namespace onnxruntime
