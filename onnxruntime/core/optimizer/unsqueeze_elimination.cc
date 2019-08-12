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
  // Get "axes" attribute. It's a required attribute so can't be null (model loading would fail if it was).
  const ONNX_NAMESPACE::AttributeProto& attr = *graph_utils::GetNodeAttribute(node, "axes");

  std::vector<int64_t> axes;
  axes.reserve(attr.ints_size());
  for (int i = 0; i < attr.ints_size(); i++) {
    axes.push_back(static_cast<int64_t>(attr.ints(i)));
  }

  // Generate new dims.
  NodeArg& input_def = *node.MutableInputDefs()[0];
  const auto& tensor_proto = *graph_utils::GetConstantInitializer(graph, input_def.Name());

  auto new_name = graph.GenerateNodeArgName("UnsqueezeElimination_" + input_def.Name());
  if (!graph_utils::CanReplaceNodeWithInitializer(graph, node, new_name)) {
    LOGS_DEFAULT(WARNING) << "UnsqueezeElimination cannot remove node " << node.Name();
    return Status::OK();
  }

  std::vector<int64_t> new_dims(axes.size() + tensor_proto.dims().size(), 0);

  for (int64_t axis : axes) {
    new_dims[axis] = 1;
  }

  auto begin = tensor_proto.dims().cbegin();
  for (auto& axis : new_dims) {
    if (axis == 0) {
      axis = *begin++;
    }
  }

  // Update shape of tensor proto.
  ONNX_NAMESPACE::TensorProto new_tensor_proto(tensor_proto);
  new_tensor_proto.set_name(new_name);
  new_tensor_proto.clear_dims();

  for (const auto& dim : new_dims) {
    new_tensor_proto.add_dims(dim);
  }

  auto& new_node_arg = graph_utils::AddInitializer(graph, new_tensor_proto);
  // Remove the Unsqueeze node and replace it with the initializer.
  graph_utils::ReplaceNodeWithInitializer(graph, node, new_node_arg);

  rule_effect = RewriteRuleEffect::kRemovedCurrentNode;

  return Status::OK();
}

bool UnsqueezeElimination::SatisfyCondition(const Graph& graph, const Node& node) const {
  // Attempt to remove an Unsqueeze operator only if it gets a constant initializer as input.
  return graph_utils::IsConstantInitializer(graph, node.InputDefs()[0]->Name());
}

}  // namespace onnxruntime
