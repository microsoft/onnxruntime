// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/onnx_protobuf.h"
#include "core/graph/graph.h"

namespace onnxruntime {

namespace utils {
bool IsSupportedOptypeVersionAndDomain(const Node& node,
                                       const std::string& op_type,
                                       ONNX_NAMESPACE::OperatorSetVersion version,
                                       const std::string& domain = kOnnxDomainAlias);

Status ForAllMutableSubgraphs(Graph& main_graph, std::function<Status(Graph&)> func);
Status ForAllSubgraphs(Graph& main_graph, std::function<Status(Graph&)> func);

/** Check whether the node has a single input and a single output. */
bool IsSingleInSingleOutNode(const Node& node);

/** Return the attribute of a Node with a given name. */
const onnx::AttributeProto* GetNodeAttribute(const Node& node, const std::string& attr_name);

/** Retrieve the values for a repeated attribute of a node and place them to the values vector. */
template <typename T>
bool GetRepeatedNodeAttributeValues(const Node& node,
                                    const std::string& attr_name,
                                    std::vector<T>& values) {
  const auto* attr = utils::GetNodeAttribute(node, attr_name);
  if (attr) {
    values = onnx::RetrieveValues<T>(*attr);
    return true;
  } else {
    return false;
  }
}

/** Remove the given single-input-single-output Node from the Graph. */
bool RemoveNodeFromPath(Graph& graph, Node& node);

}  // namespace utils
}  // namespace onnxruntime
