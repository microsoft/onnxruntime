// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/unsqueeze_elimination.h"

using namespace onnx;
using namespace ::onnxruntime::common;

namespace onnxruntime {

Status UnsqueezeElimination::ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level) const {
  std::vector<onnxruntime::NodeIndex> removed_nodes;

  for (auto& node : graph.Nodes()) {
    // recurse first as there are early exits in the processing here
    ORT_RETURN_IF_ERROR(Recurse(node, modified, graph_level));

    if (node.OpType() != "Unsqueeze" || node.GetInputEdgesCount() != 0 || graph.IsNodeOutputsInGraphOutputs(node)) {
      continue;
    }

    const onnxruntime::NodeAttributes& attributes = node.GetAttributes();
    const onnx::AttributeProto* attr = &attributes.find("axes")->second;
    if (attr == nullptr || attr->type() != AttributeProto_AttributeType_INTS) {
      continue;
    }

    // Get attribute of "axes"
    std::vector<int64_t> axes;
    for (int i = 0; i < attr->ints_size(); i++) {
      axes.push_back(static_cast<int64_t>(attr->ints(i)));
    }

    // Generate new dims
    NodeArg* input_def = node.MutableInputDefs()[0];
    const ONNX_NAMESPACE::TensorProto* tensor_proto = nullptr;
    graph.GetInitializedTensor(input_def->Name(), tensor_proto);
    if (tensor_proto == nullptr) {
      continue;
    }
    std::vector<int64_t> new_dims(axes.size() + tensor_proto->dims().size(), 0);
    if (new_dims.size() >= std::numeric_limits<int>::max())
      return Status(ONNXRUNTIME, FAIL, "index out of range");

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

    // Replace the input of the nodes following unsqueeze node
    const NodeArg* output_def = node.OutputDefs()[0];
    for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
      auto output_node = graph.GetNode((*it).Index());
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

    removed_nodes.push_back(node.Index());
  }

  for (auto i : removed_nodes) {
    graph.RemoveNode(i);
  }

  if (!removed_nodes.empty()) {
    modified = true;
  }

  return Status::OK();
}
}  // namespace onnxruntime
