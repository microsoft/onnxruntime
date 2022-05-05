// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/xnnpack/optimizer/maxpool.h"
#include <onnx/onnx_pb.h>

#include "core/framework/op_node_proto_helper.h"
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
#include "core/xnnpack/optimizer/maxpool.h"
namespace onnxruntime {
namespace xnnpack {
using namespace onnxruntime::common;
Status MaxPoolNodeProcessor::Generate(std::unique_ptr<::ONNX_NAMESPACE::GraphProto>& output_graph) {
  if (node_.OpType() != "MaxPool") return Status(ONNXRUNTIME, FAIL);

  {
    auto input_defs = node_.InputDefs();
    if (input_defs.size() != 1) return Status(ONNXRUNTIME, FAIL);
    if (!input_defs[0]->HasTensorOrScalarShape()) return Status(ONNXRUNTIME, FAIL);
    auto X_input = input_defs[0]->TypeAsProto();
    assert(X_input != nullptr);
    int32_t etype = X_input->tensor_type().elem_type();
    // Currently we only implemented a f32 kernel
    // if (etype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT && etype != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8 && etype !=
    // ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8) return Status(ONNXRUNTIME, FAIL);
    if (etype != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) return Status(ONNXRUNTIME, FAIL);

    auto& input_shape = X_input->tensor_type().shape();
    if (input_shape.dim_size() != 4) return Status(ONNXRUNTIME, FAIL);
    auto& channel_dim = input_shape.dim(1);
    if (!channel_dim.has_dim_value()) return Status(ONNXRUNTIME, FAIL);
  }
  PoolAttributes pool_attrs(info_, "MaxPool", node_.SinceVersion());
  if (!IsPaddingTypeSupportedByXNNPack(pool_attrs.auto_pad)) return Status(ONNXRUNTIME, FAIL);
  if (pool_attrs.kernel_shape.size() != 2) return Status(ONNXRUNTIME, FAIL);
  if (pool_attrs.kernel_shape[0] <= 0 || pool_attrs.kernel_shape[1] <= 0) return Status(ONNXRUNTIME, FAIL);
  if (pool_attrs.kernel_shape[0] == 1 && pool_attrs.kernel_shape[1] == 1) {
    // XNNPack doesn't like to support 1x1 maxpool.
    return Status(ONNXRUNTIME, FAIL);
  }
  auto inputdefs = node_.InputDefs();
  if (inputdefs.size() != 1) return Status::OK();

  // Skip if unknown rank
  auto shape = inputdefs[0]->Shape();
  if (shape == nullptr || shape->dim_size() != 4) {
    return Status::OK();
  }

  ::ONNX_NAMESPACE::GraphProto& g = *(output_graph = std::make_unique<::ONNX_NAMESPACE::GraphProto>());
  const ::ONNX_NAMESPACE::ValueInfoProto& subgraph_input = *g.add_input() = node_.InputDefs()[0]->ToProto();
  const ::ONNX_NAMESPACE::ValueInfoProto& subgraph_output = *g.add_output() = node_.OutputDefs()[0]->ToProto();
  constexpr int rank = 4;

  std::string trans_output = subgraph_input.name() + "_x";

  ORT_RETURN_IF_ERROR(CreateTransposeNode(*g.add_node(), "0", subgraph_input.name(), trans_output,
                                          onnx_layout_transformation::ChannelFirstToLastPerm(rank)));

  std::string conv_output;
  {
    ::ONNX_NAMESPACE::NodeProto* xnnPackMaxPooling2d = g.add_node();
    xnnPackMaxPooling2d->set_name("1");
    std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto> attrs;
    AddAttribute(attrs, "input_padding_top", pool_attrs.pads[0]);
    AddAttribute(attrs, "input_padding_right", pool_attrs.pads[3]);
    AddAttribute(attrs, "input_padding_bottom", pool_attrs.pads[2]);
    AddAttribute(attrs, "input_padding_left", pool_attrs.pads[1]);

    AddAttribute(attrs, "pooling_height", pool_attrs.kernel_shape[0]);
    AddAttribute(attrs, "pooling_width", pool_attrs.kernel_shape[1]);

    AddAttribute(attrs, "stride_height", pool_attrs.strides[0]);
    AddAttribute(attrs, "stride_width", pool_attrs.strides[1]);

    AddAttribute(attrs, "dilation_height", pool_attrs.dilations[0]);
    AddAttribute(attrs, "dilation_width", pool_attrs.dilations[1]);
    if (pool_attrs.auto_pad == AutoPadType::SAME_UPPER) {
      AddAttribute(attrs, "padding_mode", static_cast<int64_t>(1));
    } else {
      AddAttribute(attrs, "padding_mode", static_cast<int64_t>(0));
    }
    for (const auto& attr : attrs) {
      *xnnPackMaxPooling2d->add_attribute() = attr.second;
    }
    xnnPackMaxPooling2d->set_domain("com.microsoft");
    xnnPackMaxPooling2d->set_op_type("XnnPackMaxPooling2d");
    xnnPackMaxPooling2d->add_input(trans_output);
    conv_output = subgraph_input.name() + "_y";
    xnnPackMaxPooling2d->add_output(conv_output);
  }

  ORT_RETURN_IF_ERROR(CreateTransposeNode(*g.add_node(), "2", conv_output, subgraph_output.name(),
                                          onnx_layout_transformation::ChannelLastToFirstPerm(rank)));
  return Status();
}
}  // namespace xnnpack
}  // namespace onnxruntime
