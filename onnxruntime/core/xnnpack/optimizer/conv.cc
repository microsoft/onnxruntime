// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/xnnpack/optimizer/conv.h"

#include "core/framework/op_node_proto_helper.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/graph/schema_registry.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/nhwc_transformer.h"
#include "core/optimizer/selectors_actions/helpers.h"
#include "core/optimizer/transpose_optimizer/optimizer_utils.h"
#include "core/optimizer/utils.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/xnnpack/optimizer/common.h"
#include "core/xnnpack/optimizer/conv_helper.h"
#include "core/xnnpack/optimizer/layout_helper.h"
#include "core/xnnpack/optimizer/trival_subgraph.h"
namespace onnxruntime {

Status AddBiasInitializer(Graph& main_graph, int64_t bias_size, const std::string& bias_tensor_name, NodeArg** out) {
  if (bias_size < 0 || static_cast<uint64_t>(bias_size) >= std::numeric_limits<size_t>::max()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "size is too large");
  }
  // Create a bias tensor and set all elements to zero
  ::ONNX_NAMESPACE::TensorProto bias_tensor;
  std::vector<float> bias_data(bias_size, 0.0f);
  bias_tensor.mutable_float_data()->Add(bias_data.begin(), bias_data.end());
  bias_tensor.mutable_dims()->Add(bias_size);
  bias_tensor.set_name(bias_tensor_name);
  bias_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  *out = &graph_utils::AddInitializer(main_graph, bias_tensor);
  return Status::OK();
}

Status ReplaceConv(const Node& nodeRef, const std::unordered_set<const NodeArg*>& graph_const_values,
                   std::unique_ptr<::ONNX_NAMESPACE::GraphProto>& output_graph) {
  {
    if (nodeRef.OpType() != "Conv") return Status(common::ONNXRUNTIME, common::FAIL);
    constexpr bool input_is_nchw = true;
    // Conv has either 2 or 3 inputs.
    auto input_defs = nodeRef.InputDefs();
    if (input_defs.size() != 2 && input_defs.size() != 3) return Status(common::ONNXRUNTIME, common::FAIL);
    if (graph_const_values.find(input_defs[1]) == graph_const_values.end()) {
      // Weight is not const, we can't run it.
      return Status(common::ONNXRUNTIME, common::FAIL);
    }
    // The two or three inputs are: X, W, B
    const NodeArg* weight_node_arg = input_defs[1];
    if (weight_node_arg == nullptr) return Status(common::ONNXRUNTIME, common::FAIL);
    // Weight must be a const and all dims are known
    bool is_weight_shape_known = optimizer_utils::IsShapeKnownOnAllDims(*weight_node_arg, 4);
    if (!is_weight_shape_known) return Status(common::ONNXRUNTIME, common::FAIL);

    ProtoHelperNodeContext nc(nodeRef);
    OpNodeProtoHelper info(&nc);
    auto X_input = input_defs[0]->TypeAsProto();
    if (X_input == nullptr || !X_input->has_tensor_type() || !X_input->tensor_type().has_shape() ||
        X_input->tensor_type().elem_type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      return Status(common::ONNXRUNTIME, common::FAIL);
    std::string auto_pad_str;
    ORT_RETURN_IF_ERROR(info.GetAttr<std::string>("auto_pad", &auto_pad_str));
    AutoPadType padding_type = StringToAutoPadType(auto_pad_str);
    if (!IsPaddingTypeSupportedByXNNPack(padding_type)) return Status(common::ONNXRUNTIME, common::FAIL);
    // The "auto_pad_str" string must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID
    // TF2ONNX converter doesn't use SAME_LOWER.
    // SAME_UPPER maps to TF SAME padding
    if (padding_type == AutoPadType::SAME_UPPER) {
      std::vector<int64_t> dilations;
      Status st1 = info.GetAttrs<int64_t>("dilations", dilations);
      if (dilations.size() != 2) return Status(common::ONNXRUNTIME, common::FAIL);
      // Don't know how to handle dilation!=1 cases yet. TF doesn't have it.
      if (dilations[0] != 1 || dilations[1] != 1) return Status(common::ONNXRUNTIME, common::FAIL);
    }

    auto& input_shape = X_input->tensor_type().shape();
    if (input_shape.dim_size() != 4) return Status(common::ONNXRUNTIME, common::FAIL);
    auto& channel_dim = input_is_nchw ? input_shape.dim(1) : input_shape.dim(3);
    if (!channel_dim.has_dim_value()) return Status(common::ONNXRUNTIME, common::FAIL);

    auto weight_input = weight_node_arg->TypeAsProto();
    TensorShape weight_shape = utils::GetTensorShapeFromTensorShapeProto(weight_input->tensor_type().shape());
    int64_t group = 1;
    ORT_RETURN_IF_ERROR(info.GetAttr<int64_t>("group", &group));
    int64_t input_channels = input_is_nchw ? input_shape.dim(1).dim_value() : input_shape.dim(3).dim_value();
    if (group != 1 && group != input_channels) return Status(common::ONNXRUNTIME, common::FAIL);

    std::vector<int64_t> pads;
    Status st = info.GetAttrs<int64_t>("pads", pads);
    if (st.IsOK()) {
      if (pads.size() != 4) return Status(common::ONNXRUNTIME, common::FAIL);
    }
  }

  ProtoHelperNodeContext nc(nodeRef);
  OpNodeProtoHelper info(&nc);
  int64_t group = 1;
  ORT_RETURN_IF_ERROR(info.GetAttr<int64_t>("group", &group));
  auto X_input = info.GetInputType(0);
  auto weight_input = info.GetInputType(1);
  TensorShape weight_shape = utils::GetTensorShapeFromTensorShapeProto(weight_input->tensor_type().shape());
  TensorShape X_shape = utils::GetTensorShapeFromTensorShapeProto(X_input->tensor_type().shape());
  if (weight_shape.NumDimensions() != 4 || X_shape.NumDimensions() != 4) return Status::OK();

  // Now the input shape is still NCHW
  int64_t input_channels = X_shape[1];

  if (group != 1 && group != input_channels) return Status::OK();
  for (size_t i = 0; i != weight_shape.NumDimensions(); ++i) {
    if (weight_shape[i] <= 0) return Status::OK();
  }

  // const_cast
  const bool has_bias = nodeRef.InputDefs().size() >= 3;
  std::string auto_pad_str;
  ORT_RETURN_IF_ERROR(info.GetAttr<std::string>("auto_pad", &auto_pad_str));
  // group == 1 || group  == input / output channel count
  // For now we assume input channel count isn't 1, so that group count != input/output channel count
  bool is_depthwise = input_channels != 1 && group == input_channels;

  if (nodeRef.InputDefs().size() < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Expect at least 2 inputs, got ", nodeRef.InputDefs().size());
  }

  std::vector<int64_t> weight_perm = is_depthwise ? std::vector<int64_t>{1, 2, 3, 0} : std::vector<int64_t>{0, 2, 3, 1};

  std::vector<int64_t> strides, dilations, pads;
  Status st = info.GetAttrs<int64_t>("strides", strides);
  if (!st.IsOK()) {
    // ONNX spec says: "If not present, the stride defaults is 1 along each spatial axis."
    strides.assign(4, 1);
  }
  st = info.GetAttrs<int64_t>("dilations", dilations);
  if (!st.IsOK()) {
    // ONNX spec says: "If not present, the dilation defaults is 1 along each spatial axis."
    dilations.assign(4, 1);
  }
  st = info.GetAttrs<int64_t>("pads", pads);
  if (!st.IsOK()) {
    // ONNX spec says: "If not present, the padding defaults to 0 along start and end of each spatial axis."
    pads.resize(4);
  }

  std::string node_name = nodeRef.Name();
  std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto> attrs;

  AddAttribute(attrs, "input_padding_top", pads[0]);
  AddAttribute(attrs, "input_padding_right", pads[3]);
  AddAttribute(attrs, "input_padding_bottom", pads[2]);
  AddAttribute(attrs, "input_padding_left", pads[1]);

  AddAttribute(attrs, "subsampling_height", strides[0]);
  AddAttribute(attrs, "subsampling_width", strides[1]);

  AddAttribute(attrs, "dilation_height", dilations[0]);
  AddAttribute(attrs, "dilation_width", dilations[1]);

  if (!is_depthwise) AddAttribute(attrs, "groups", group);

  if (auto_pad_str == "NOTSET") {
    AddAttribute(attrs, "padding_mode", static_cast<int64_t>(0));
  } else if (auto_pad_str == "VALID") {
    AddAttribute(attrs, "padding_mode", static_cast<int64_t>(0));
  } else if (auto_pad_str == "SAME_UPPER") {
    AddAttribute(attrs, "padding_mode", static_cast<int64_t>(1));
  } else {
    // This line of code should not be reached because in IsConvSupportedByXNNPack function we already checked
    // auto_pad_str
    return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
  }

  ::ONNX_NAMESPACE::GraphProto& g = *(output_graph = std::make_unique<::ONNX_NAMESPACE::GraphProto>());
  const ::ONNX_NAMESPACE::ValueInfoProto& subgraph_input0 = *g.add_input() = nodeRef.InputDefs()[0]->ToProto();
  const ::ONNX_NAMESPACE::ValueInfoProto& subgraph_input1 = *g.add_input() = nodeRef.InputDefs()[1]->ToProto();
  ::ONNX_NAMESPACE::ValueInfoProto* subgraph_input2 = nullptr;
  if (has_bias) {
    subgraph_input2 = g.add_input();
    *subgraph_input2 = nodeRef.InputDefs()[2]->ToProto();
  }
  const ::ONNX_NAMESPACE::ValueInfoProto& subgraph_output = *g.add_output() = nodeRef.OutputDefs()[0]->ToProto();
  std::string trans0_output = subgraph_input0.name() + "_x";
  std::string trans1_output = subgraph_input1.name() + "_y";

  // Transpose input
  constexpr int rank = 4;

  ORT_RETURN_IF_ERROR(CreateTransposeNode(*g.add_node(), "0", subgraph_input0.name(), trans0_output,
                                          onnx_layout_transformation::ChannelFirstToLastPerm(rank)));
  ORT_RETURN_IF_ERROR(CreateTransposeNode(*g.add_node(), "1", subgraph_input1.name(), trans1_output, weight_perm));
  std::string conv_output;
  std::string bias_tensor_name;
  if (!has_bias) {
    int64_t bias_size = weight_shape[0];
    bias_tensor_name = nodeRef.Name() + "_bias";
    if (bias_size < 0 || static_cast<uint64_t>(bias_size) >= std::numeric_limits<size_t>::max()) {
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "size is too large");
    }

    ::ONNX_NAMESPACE::TensorProto* t = g.add_initializer();
    TensorShape shape({bias_size});
    std::vector<float> data(static_cast<size_t>(bias_size), 0.0f);

    *t = utils::TensorToTensorProto(
        *Tensor::Create(DataTypeImpl::GetType<float>(), shape, data.data(),
                        OrtMemoryInfo(onnxruntime::CPU, OrtAllocatorType::OrtDeviceAllocator)),
        bias_tensor_name);

  } else {
    bias_tensor_name = subgraph_input2->name();
  }
  {
    ::ONNX_NAMESPACE::NodeProto* xnnPackConv2d = g.add_node();
    xnnPackConv2d->set_name("2");
    for (const auto& attr : attrs) {
      *xnnPackConv2d->add_attribute() = attr.second;
    }
    xnnPackConv2d->set_domain("com.microsoft");
    xnnPackConv2d->set_op_type(is_depthwise ? "XnnPackDepthwiseConvolution2d" : "XnnPackConvolution2d");
    xnnPackConv2d->add_input(trans0_output);
    xnnPackConv2d->add_input(trans1_output);
    xnnPackConv2d->add_input(bias_tensor_name);
    conv_output = subgraph_input0.name() + "_y";
    xnnPackConv2d->add_output(conv_output);
  }
  ORT_RETURN_IF_ERROR(CreateTransposeNode(*g.add_node(), "3", conv_output, subgraph_output.name(),
                                          onnx_layout_transformation::ChannelLastToFirstPerm(rank)));
  return Status::OK();
}
}  // namespace onnxruntime