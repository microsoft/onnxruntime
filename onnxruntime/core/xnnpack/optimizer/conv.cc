// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/xnnpack/optimizer/conv.h"
#include <onnx/onnx_pb.h>
#include "core/framework/op_node_proto_helper.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/graph_utils.h"
#include "core/graph/node_attr_utils.h"
#include "core/graph/schema_registry.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/xnnpack/optimizer/common.h"
#include "core/xnnpack/optimizer/conv_helper.h"
#include "core/xnnpack/optimizer/layout_helper.h"
#include "core/optimizer/transpose_optimizer/optimizer_utils.h"

#include "core/optimizer/utils.h"

namespace onnxruntime {
namespace xnnpack {
Status ConvNodeProcessor::Generate(std::unique_ptr<::ONNX_NAMESPACE::GraphProto>& output_graph) {
  {
    if (node_.OpType() != "Conv") return Status(common::ONNXRUNTIME, common::FAIL);
    // Conv has either 2 or 3 inputs.
    auto input_defs = node_.InputDefs();
    if (input_defs.size() != 2 && input_defs.size() != 3) return Status(common::ONNXRUNTIME, common::FAIL);
    if (graph_const_values_.find(input_defs[1]) == graph_const_values_.end()) {
      // Weight is not const, we can't run it.
      return Status(common::ONNXRUNTIME, common::FAIL, "weight is not const");
    }
    // The two or three inputs are: X, W, B
    const NodeArg* weight_node_arg = input_defs[1];
    if (weight_node_arg == nullptr) return Status(common::ONNXRUNTIME, common::FAIL);
    // Weight must be a const and all dims are known
    bool is_weight_shape_known = optimizer_utils::IsShapeKnownOnAllDims(*weight_node_arg, 4);
    if (!is_weight_shape_known) return Status(common::ONNXRUNTIME, common::FAIL);

    auto X_input = input_defs[0]->TypeAsProto();
    if (X_input == nullptr || !X_input->has_tensor_type() || !X_input->tensor_type().has_shape() ||
        X_input->tensor_type().elem_type() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      return Status(common::ONNXRUNTIME, common::FAIL);
  }

  std::string auto_pad_str;
  ORT_RETURN_IF_ERROR(info_.GetAttr<std::string>("auto_pad", &auto_pad_str));
  AutoPadType padding_type = StringToAutoPadType(auto_pad_str);
  // The "auto_pad_str" string must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID
  // TF2ONNX converter doesn't use SAME_LOWER.
  // SAME_UPPER maps to TF SAME padding
  std::vector<int64_t> strides, dilations, pads;
  Status st = info_.GetAttrs<int64_t>("dilations", dilations);
  if (!st.IsOK()) {
    // ONNX spec says: "If not present, the dilation defaults is 1 along each spatial axis."
    dilations.assign(4, 1);
  }
  if (padding_type == AutoPadType::SAME_UPPER) {
    if (dilations.size() != 2) return Status(common::ONNXRUNTIME, common::FAIL);
    // Don't know how to handle dilation!=1 cases yet. It seems TF doesn't have it.
    if (dilations[0] != 1 || dilations[1] != 1) return Status(common::ONNXRUNTIME, common::FAIL);
  }

  int64_t group = 1;
  ORT_RETURN_IF_ERROR(info_.GetAttr<int64_t>("group", &group));
  auto X_input = info_.GetInputType(0);
  auto weight_input = info_.GetInputType(1);
  TensorShape weight_shape = utils::GetTensorShapeFromTensorShapeProto(weight_input->tensor_type().shape());
  TensorShape X_shape = utils::GetTensorShapeFromTensorShapeProto(X_input->tensor_type().shape());
  if (weight_shape.NumDimensions() != 4 || X_shape.NumDimensions() != 4) return Status::OK();
  auto& input_shape = X_input->tensor_type().shape();
  if (input_shape.dim_size() != 4) return Status(common::ONNXRUNTIME, common::FAIL);
  auto& channel_dim = input_shape.dim(1);
  if (!channel_dim.has_dim_value()) return Status(common::ONNXRUNTIME, common::FAIL);
  // Now the input shape is still NCHW
  int64_t input_channels = X_shape[1];

  if (group != 1 && group != input_channels) return Status::OK();
  for (size_t i = 0; i != weight_shape.NumDimensions(); ++i) {
    if (weight_shape[i] <= 0) return Status::OK();
  }

  // const_cast
  const bool has_bias = node_.InputDefs().size() >= 3;
  // group == 1 || group  == input / output channel count
  // For now we assume input channel count isn't 1, so that group count != input/output channel count
  bool is_depthwise = input_channels != 1 && group == input_channels;

  if (node_.InputDefs().size() < 2) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Expect at least 2 inputs, got ", node_.InputDefs().size());
  }

  std::vector<int64_t> weight_perm = is_depthwise ? std::vector<int64_t>{1, 2, 3, 0} : std::vector<int64_t>{0, 2, 3, 1};

  st = info_.GetAttrs<int64_t>("strides", strides);
  if (!st.IsOK()) {
    // ONNX spec says: "If not present, the stride defaults is 1 along each spatial axis."
    strides.assign(4, 1);
  }

  st = info_.GetAttrs<int64_t>("pads", pads);
  if (!st.IsOK()) {
    // ONNX spec says: "If not present, the padding defaults to 0 along start and end of each spatial axis."
    pads.resize(4);
  } else if (pads.size() != 4)
    return Status(common::ONNXRUNTIME, common::FAIL);

  std::string node_name = node_.Name();
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

  if (padding_type == AutoPadType::NOTSET) {
    AddAttribute(attrs, "padding_mode", static_cast<int64_t>(0));
  } else if (padding_type == AutoPadType::VALID) {
    AddAttribute(attrs, "padding_mode", static_cast<int64_t>(0));
  } else if (padding_type == AutoPadType::SAME_UPPER) {
    AddAttribute(attrs, "padding_mode", static_cast<int64_t>(1));
  } else {
    return Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
  }

  ::ONNX_NAMESPACE::GraphProto& g = *(output_graph = std::make_unique<::ONNX_NAMESPACE::GraphProto>());
  const ::ONNX_NAMESPACE::ValueInfoProto& subgraph_input0 = *g.add_input() = node_.InputDefs()[0]->ToProto();
  const ::ONNX_NAMESPACE::ValueInfoProto& subgraph_input1 = *g.add_input() = node_.InputDefs()[1]->ToProto();
  ::ONNX_NAMESPACE::ValueInfoProto* subgraph_input2 = nullptr;
  if (has_bias) {
    subgraph_input2 = g.add_input();
    *subgraph_input2 = node_.InputDefs()[2]->ToProto();
  }
  const ::ONNX_NAMESPACE::ValueInfoProto& subgraph_output = *g.add_output() = node_.OutputDefs()[0]->ToProto();
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
    bias_tensor_name = node_.Name() + "_bias";
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

}  // namespace xnnpack
}  // namespace onnxruntime