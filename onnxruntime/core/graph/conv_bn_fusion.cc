// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/initializer.h"
#include "core/graph/conv_bn_fusion.h"

using namespace onnx;
using namespace ::onnxruntime::common;
namespace onnxruntime {

Status ConvBNFusion::Apply(onnxruntime::Graph& graph, bool& modified) const {
  std::vector<onnxruntime::NodeIndex> removed_nodes;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() != "BatchNormalization" ||
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

    // Get value of attribute epsilon
    const onnxruntime::NodeAttributes& attributes = node.GetAttributes();
    const onnx::AttributeProto* attr = &(attributes.find("epsilon")->second);
    if (attr == nullptr || attr->type() != AttributeProto_AttributeType_FLOAT) {
      continue;
    }
    float epsilon = static_cast<float>(attr->f());

    // Get initializers of BatchNormalization
    const auto& bn_inputs = node.InputDefs();
    const ONNX_NAMESPACE::TensorProto* bn_scale_tensor_proto = nullptr;
    graph.GetInitializedTensor(bn_inputs[1]->Name(), bn_scale_tensor_proto);

    const ONNX_NAMESPACE::TensorProto* bn_B_tensor_proto = nullptr;
    graph.GetInitializedTensor(bn_inputs[2]->Name(), bn_B_tensor_proto);

    const ONNX_NAMESPACE::TensorProto* bn_mean_tensor_proto = nullptr;
    graph.GetInitializedTensor(bn_inputs[3]->Name(), bn_mean_tensor_proto);

    const ONNX_NAMESPACE::TensorProto* bn_var_tensor_proto = nullptr;
    graph.GetInitializedTensor(bn_inputs[4]->Name(), bn_var_tensor_proto);

    const ONNX_NAMESPACE::TensorProto* conv_W_tensor_proto = nullptr;
    graph.GetInitializedTensor(conv_inputs[1]->Name(), conv_W_tensor_proto);

    const ONNX_NAMESPACE::TensorProto* conv_B_tensor_proto = nullptr;
    graph.GetInitializedTensor(conv_inputs[2]->Name(), conv_B_tensor_proto);

    // Currently, fusion is only supported for float or double data type.
    if (!Initializer::IsSupportedDataType(bn_scale_tensor_proto) ||
        !Initializer::IsSupportedDataType(bn_B_tensor_proto) ||
        !Initializer::IsSupportedDataType(bn_mean_tensor_proto) ||
        !Initializer::IsSupportedDataType(bn_var_tensor_proto) ||
        !Initializer::IsSupportedDataType(conv_W_tensor_proto) ||
        !Initializer::IsSupportedDataType(conv_B_tensor_proto)) {
      continue;
    }
    auto bn_scale = std::make_unique<Initializer>(bn_scale_tensor_proto);
    auto bn_B = std::make_unique<Initializer>(bn_B_tensor_proto);
    auto bn_mean = std::make_unique<Initializer>(bn_mean_tensor_proto);
    auto bn_var = std::make_unique<Initializer>(bn_var_tensor_proto);
    auto conv_W = std::make_unique<Initializer>(conv_W_tensor_proto);
    auto conv_B = std::make_unique<Initializer>(conv_B_tensor_proto);

    if (bn_scale->size() != bn_var->size() || bn_scale->data_type() != bn_var->data_type() ||
      !(conv_W->dims().size() > 2 && conv_W->dims()[0] == bn_scale->dims()[0]) ||
      conv_B->size() != bn_mean->size() || conv_B->data_type() != bn_mean->data_type() ||
      conv_B->size() != bn_scale->size() || conv_B->data_type() != bn_scale->data_type() ||
      conv_B->size() != bn_B->size() || conv_B->data_type() != bn_B->data_type()) {
      continue;
    }

    // Caculate new value of initializers of conv node
    bn_var->add(epsilon);
    bn_var->sqrt();
    bn_scale->div(*bn_var);
    conv_W->scale_by_axis(*bn_scale, 1);
    conv_B->sub(*bn_mean);
    conv_B->mul(*bn_scale);
    conv_B->add(*bn_B);

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

    // Replace the input of the nodes following batch normalization node
    const NodeArg* bn_output_def = node.OutputDefs()[0];
    const NodeArg* conv_output_def = conv_node.OutputDefs()[0];
    for (auto it = node.OutputNodesBegin(); it != node.OutputNodesEnd(); ++it) {
      auto output_node = graph.GetNode((*it)->Index());
      if (!output_node) {
        return Status(ONNXRUNTIME, INVALID_ARGUMENT);
      }

      auto& input_defs = output_node->MutableInputDefs();
      for (auto& def : input_defs) {
        if (def == bn_output_def) {
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

}
