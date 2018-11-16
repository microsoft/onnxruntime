// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/initializer.h"
#include "core/graph/conv_add_fusion.h"

using namespace onnx;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status ConvAddFusion::Apply(onnxruntime::Graph& graph, bool& modified) const {
  std::vector<onnxruntime::NodeIndex> removed_nodes;

  for (auto& node : graph.Nodes()) {
    if (node.OpType() != "Add" ||
        node.GetInputEdgesCount() != 1 ||
        (*node.InputEdgesBegin()).GetNode().OpType() != "Conv" ||
        graph.IsNodeOutputsInGraphOutputs(node)) {
      continue;
    }

    const auto& conv_node = (*node.InputEdgesBegin()).GetNode();
    const auto& conv_inputs = conv_node.InputDefs();
    // For now, fusion is only done when conv has bias.
    if (conv_inputs.size() != 3) {
      continue;
    }
    const ONNX_NAMESPACE::TensorProto* conv_B_tensor_proto = nullptr;
    graph.GetInitializedTensor(conv_inputs[2]->Name(), conv_B_tensor_proto);

    const auto& add_inputs = node.InputDefs();
    const ONNX_NAMESPACE::TensorProto* add_B_tensor_proto = nullptr;
    graph.GetInitializedTensor(add_inputs[1]->Name(), add_B_tensor_proto);

    // Currently, fusion is only supported for float or double data type.
    if (!Initializer::IsSupportedDataType(conv_B_tensor_proto) ||
        !Initializer::IsSupportedDataType(add_B_tensor_proto)) {
      continue;
    }

    auto conv_B = std::make_unique<Initializer>(conv_B_tensor_proto);
    auto add_B = std::make_unique<Initializer>(add_B_tensor_proto);

    // Don't fuse if size or data type is different.
    if (conv_B->size() != add_B->size() || conv_B->data_type() != add_B->data_type()) {
      continue;
    }

    // Caculate new value of initializers of conv node
    conv_B->add(*add_B);

    // Create new initializers of conv
    ONNX_NAMESPACE::TensorProto new_conv_B_tensor_proto(*conv_B_tensor_proto);
    conv_B->ToProto(&new_conv_B_tensor_proto);

    // Replace initializers of conv node
    graph.RemoveInitializedTensor(conv_inputs[2]->Name());
    graph.AddInitializedTensor(new_conv_B_tensor_proto);

    // Replace the input of the node following add node
    const NodeArg* add_output_def = node.OutputDefs()[0];
    const NodeArg* conv_output_def = conv_node.OutputDefs()[0];
    for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
      auto output_node = graph.GetNode((*it)->Index());
      if (!output_node) {
        return Status(ONNXRUNTIME, INVALID_ARGUMENT);
      }

      auto& input_defs = output_node->MutableInputDefs();
      for (auto& def : input_defs) {
        if (def == add_output_def) {
            def = const_cast<NodeArg*>(conv_output_def);
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
    ONNXRUNTIME_RETURN_IF_ERROR(graph.Resolve());
  }

  return Status::OK();
}
}  // namespace onnxruntime