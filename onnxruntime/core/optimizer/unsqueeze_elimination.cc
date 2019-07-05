// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/unsqueeze_elimination.h"
#include "core/common/logging/logging.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::common;

namespace onnxruntime {

Status UnsqueezeElimination::Apply(Graph& graph, Node& node, RewriteRuleEffect& rule_effect) const {
  // Get "axes" attribute.
  const ONNX_NAMESPACE::AttributeProto* attr = graph_utils::GetNodeAttribute(node, "axes");
  if (attr == nullptr || attr->type() != AttributeProto_AttributeType_INTS) {
    return Status::OK();
  }

  std::vector<int64_t> axes;
  axes.reserve(attr->ints_size());
  for (int i = 0; i < attr->ints_size(); i++) {
    axes.push_back(static_cast<int64_t>(attr->ints(i)));
  }

  // Generate new dims.
  NodeArg& input_def = *node.MutableInputDefs()[0];
  const auto* tensor_proto = graph_utils::GetConstantInitializer(graph, input_def.Name());
  ORT_ENFORCE(tensor_proto);

  std::vector<int64_t> new_dims(axes.size() + tensor_proto->dims().size(), 0);
  if (new_dims.size() >= std::numeric_limits<int>::max()) {
    return Status(ONNXRUNTIME, FAIL, "index out of range");
  }

  // for simplicity we will add the replacement initializer in this graph, even if the original came from an ancestor
  // so generate a replacement node arg name for this graph. use a prefix to minimize the chance of any naming
  // clash with values in a subgraph
  auto new_name = graph.GenerateNodeArgName("UnsqueezeElimination_" + input_def.Name());
  if (!graph_utils::CanRemoveNode(graph, node, &new_name)) {
    LOGS_DEFAULT(WARNING) << "UnsqueezeElimination cannot remove node " << node.Name();
    return Status::OK();
  }

  for (int64_t axis : axes) {
    new_dims[axis] = 1;
  }

  auto begin = tensor_proto->dims().cbegin();
  for (auto& axis : new_dims) {
    if (axis == 0) {
      axis = *begin++;
    }
  }

  // Update shape of tensor proto.
  ONNX_NAMESPACE::TensorProto new_tensor_proto(*tensor_proto);
  new_tensor_proto.set_name(new_name);

  for (int i = 0; i < static_cast<int>(new_dims.size()); i++) {
    if (i < tensor_proto->dims().size()) {
      new_tensor_proto.set_dims(i, new_dims[i]);
    } else {
      new_tensor_proto.add_dims(new_dims[i]);
    }
  }

  auto& new_node_arg = graph_utils::AddReplacementInitializer(graph, new_tensor_proto);

  // Remove Unsqueeze node.
  if (graph_utils::RemoveNode(graph, node, &new_node_arg)) {
    rule_effect = RewriteRuleEffect::kRemovedCurrentNode;
  }

  return Status::OK();
}

bool UnsqueezeElimination::SatisfyCondition(const Graph& graph, const Node& node) const {
  // Attempt to remove an Unsqueeze operator only if it gets a constant initializer as input.
  return graph_utils::IsConstantInitializer(graph, node.InputDefs()[0]->Name()) &&
         !graph.IsNodeOutputsInGraphOutputs(node);
}

}  // namespace onnxruntime
