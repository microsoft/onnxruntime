// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/initializer.h"
#include "core/graph/conv_mul_fusion.h"

using namespace onnx;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status ConvMulFusion::Apply(onnxruntime::Graph& graph, bool& modified) const {
  std::vector<onnxruntime::NodeIndex> removed_nodes;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() != "Mul" ||
        node.GetInputEdgesCount() != 1 ||
        (*node.InputEdgesBegin())->GetNode().OpType() != "Conv" ||
        graph.IsNodeOutputsInGraphOutputs(node)) {
      continue;
    }

    const auto& conv_node = (*node.InputEdgesBegin())->GetNode();
    const auto& conv_inputs = conv_node.InputDefs();
    // For now, fusion is only done when conv has bias.
    if (conv_inputs.size() != 3) {
      continue;
    }

    const ONNX_NAMESPACE::TensorProto* conv_W_tensor_proto = nullptr;
    graph.GetInitializedTensor(conv_inputs[1]->Name(), conv_W_tensor_proto);

    const ONNX_NAMESPACE::TensorProto* conv_B_tensor_proto = nullptr;
    graph.GetInitializedTensor(conv_inputs[2]->Name(), conv_B_tensor_proto);

    const auto& mul_inputs = node.InputDefs();
    const ONNX_NAMESPACE::TensorProto* mul_B_tensor_proto = nullptr;
    graph.GetInitializedTensor(mul_inputs[1]->Name(), mul_B_tensor_proto);

    // Currently, fusion is only supported for float or double data type.
    if (!Initializer::IsSupportedDataType(conv_W_tensor_proto) ||
        !Initializer::IsSupportedDataType(conv_B_tensor_proto) ||
        !Initializer::IsSupportedDataType(mul_B_tensor_proto)) {
      continue;
    }

    auto conv_W = std::make_unique<Initializer>(conv_W_tensor_proto);
    auto conv_B = std::make_unique<Initializer>(conv_B_tensor_proto);
    auto mul_B = std::make_unique<Initializer>(mul_B_tensor_proto);

    if (conv_B->size() != mul_B->size() ||
        !(conv_W->dims().size() > 2 && conv_W->dims()[0] == mul_B->dims()[0]) ||
        conv_W->data_type() != mul_B->data_type() || conv_B->data_type() != mul_B->data_type()) {
      continue;
    }

    // Caculate new value of initializers of conv node
    conv_W->scale_by_axis(*mul_B, 1);
    conv_B->mul(*mul_B);

    // Create new initializers of conv
    ONNX_NAMESPACE::TensorProto new_conv_W_tensor_proto(*conv_W_tensor_proto);
    conv_W->ToProto(&new_conv_W_tensor_proto);
    ONNX_NAMESPACE::TensorProto new_conv_B_tensor_proto(*conv_B_tensor_proto);
    conv_B->ToProto(&new_conv_B_tensor_proto);

    // Replace initializers of conv node
    graph.RemoveInitializedTensor(conv_inputs[1]->Name());
    graph.RemoveInitializedTensor(conv_inputs[2]->Name());
    graph.AddInitializedTensor(new_conv_W_tensor_proto);
    graph.AddInitializedTensor(new_conv_B_tensor_proto);

    // Replace the input of the node following mul node
    const NodeArg* mul_output_def = node.OutputDefs()[0];
    const NodeArg* conv_output_def = conv_node.OutputDefs()[0];
    for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
      auto output_node = graph.GetNode((*it)->Index());
      if (!output_node) {
        return Status(ONNXRUNTIME, INVALID_ARGUMENT);
      }

      auto& input_defs = output_node->MutableInputDefs();
      for (auto& def : input_defs) {
        if (def == mul_output_def) {
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