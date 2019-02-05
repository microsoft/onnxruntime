// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph.h"

namespace onnxruntime {

namespace graph_edit_utils {
bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       const std::string& op_type,
                                       ONNX_NAMESPACE::OperatorSetVersion version,
                                       const std::string& domain = kOnnxDomainAlias);

Status ForAllMutableSubgraphs(Graph& main_graph, std::function<Status(Graph&)> func);
Status ForAllSubgraphs(Graph& main_graph, std::function<Status(Graph&)> func);

/** Check whether the node has a single input and a single output. */
bool IsSingleInSingleOutNode(const Node& node);

/** Return the attribute of a Node with a given name. */
const ONNX_NAMESPACE::AttributeProto* GetNodeAttribute(const Node& node, const std::string& attr_name);

/** Retrieve the values for a repeated attribute of a node and place them to the values vector. */
template <typename T>
bool GetRepeatedNodeAttributeValues(const Node& node,
                                    const std::string& attr_name,
                                    std::vector<T>& values) {
  const auto* attr = graph_edit_utils::GetNodeAttribute(node, attr_name);
  if (attr) {
    values = ONNX_NAMESPACE::RetrieveValues<T>(*attr);
    return true;
  } else {
    return false;
  }
}

/** Remove the given single-input-single-output Node from the Graph. */
bool RemoveSingleInSingleOutNode(Graph& graph, Node& node);

/** Checks if the given node has only constant inputs (initializers). */
bool IsConstantInputsNode(const Graph& graph, const Node& node);

/** Build a subgraph given a Graph and the indices of the nodes of the Graph that will
    be added to the subgraph. */
Status BuildSubgraph(const Graph& graph,
                     const std::vector<NodeIndex>& subgraph_nodes,
                     Graph& subgraph);

/** Remove all output edges from the given Node of the Graph. */
size_t RemoveNodeOutputEdges(Graph& graph, Node& node);

}  // namespace graph_edit_utils

}  // namespace onnxruntime
