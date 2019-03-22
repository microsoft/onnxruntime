// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/unsqueeze_elimination.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph.h"

using namespace ONNX_NAMESPACE;
using namespace ::onnxruntime::common;

namespace onnxruntime {

Status UnsqueezeElimination::Apply(Graph& graph, Node& node, bool& modified, bool& removed) {
  // Get "axes" attribute
  const ONNX_NAMESPACE::AttributeProto* attr = graph_utils::GetNodeAttribute(node, "axes");
  if (attr == nullptr || attr->type() != AttributeProto_AttributeType_INTS) {
    return Status::OK();
  }

  std::vector<int64_t> axes;
  for (int i = 0; i < attr->ints_size(); i++) {
    axes.push_back(static_cast<int64_t>(attr->ints(i)));
  }

  // Generate new dims
  NodeArg* input_def = node.MutableInputDefs()[0];
  const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
  graph.GetInitializedTensor(input_def->Name(), tensor_proto);
  if (tensor_proto == nullptr) {
    return Status::OK();
  }
  std::vector<int64_t> new_dims(axes.size() + tensor_proto->dims().size(), 0);
  if (new_dims.size() >= std::numeric_limits<int>::max()) {
    return Status(ONNXRUNTIME, FAIL, "index out of range");
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

  // Update shape of tensor proto
  ONNX_NAMESPACE::TensorProto new_tensor_proto(*tensor_proto);

  for (int i = 0; i < static_cast<int>(new_dims.size()); i++) {
    if (i < tensor_proto->dims().size()) {
      new_tensor_proto.set_dims(i, new_dims[i]);
    } else {
      new_tensor_proto.add_dims(new_dims[i]);
    }
  }
  graph.RemoveInitializedTensor(input_def->Name());
  graph.AddInitializedTensor(new_tensor_proto);

  // Update shape of NodeArg
  TensorShapeProto shape;
  for (auto dim : new_dims) {
    shape.add_dim()->set_dim_value(dim);
  }
  input_def->SetShape(shape);

  // Keep all output nodes in a list (after removing all edges below, there will be no other way to access them; 
  // I cannot first update the output defs, because edge removal will not work afterwards).
  std::vector<NodeIndex> output_nodes_idx;
  for (auto it = node.OutputNodesBegin(), end = node.OutputNodesEnd(); it != end; ++it) {
    auto output_node_idx = (*it).Index();
    output_nodes_idx.push_back(output_node_idx);
  }

  // Remove output edges of the Unsqueeze node.
  graph_utils::RemoveNodeOutputEdges(graph, node);

  // Replace the input of the nodes following the Unsqueeze node.
  const NodeArg* output_def = node.OutputDefs()[0];
  for (auto idx : output_nodes_idx) {
    auto output_node = graph.GetNode(idx);
    if (!output_node) {
      return Status(ONNXRUNTIME, INVALID_ARGUMENT);
    }
    auto& input_defs = output_node->MutableInputDefs();
    for (auto& def : input_defs) {
      if (def == output_def) {
        def = input_def;
      }
    }
  }

  // Remove the Unsqueeze node.
  graph.RemoveNode(node.Index());
  removed = modified = true;

  return Status::OK();
}  // namespace onnxruntime

bool UnsqueezeElimination::SatisfyCondition(const Graph& graph, const Node& node) {
  return OpTypeCondition(node) && node.GetInputEdgesCount() == 0 && !graph.IsNodeOutputsInGraphOutputs(node);
}

bool UnsqueezeElimination::OpTypeCondition(const Node& node) {
  return node.OpType() == included_op_type_;
}

}  // namespace onnxruntime
