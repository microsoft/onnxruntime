// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/nn/channel_shuffle.h"

#include <cstdint>
#include <utility>

#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/graph/graph.h"

#include "xnnpack.h"

namespace onnxruntime {
namespace xnnpack {

bool ChanneShuffle::IsOnnxNodeSupported(const NodeUnit& node_unit,
                                        const GraphViewer& graph) {
  bool support = false;
  static const std::string node_to_be_fuse = "Transpose";
  const onnxruntime::Node& node = node_unit.GetNode();
  do {
    //  reshape-->transpose-->reshape ---> ChannelShuffle
    // shape must be constant
    const auto* shape_vec_second = graph.GetConstantInitializer(node.InputDefs()[1]->Name(), true);
    if (shape_vec_second == nullptr) {
      break;
    }

    const onnxruntime::Node& node_transpose = *node.InputNodesBegin();
    if (node_transpose.OpType() != node_to_be_fuse ||
        node_transpose.GetOutputEdgesCount() != 1) {
      break;
    }

    const onnxruntime::Node& node_reshape = *node_transpose.InputNodesBegin();
    if (node_reshape.OpType() != node.OpType() ||
        node_transpose.GetOutputEdgesCount() != 1) {
      break;
    }

    // shape must be constant
    const auto* shape_vec_first = graph.GetConstantInitializer(node_reshape.InputDefs()[1]->Name(), true);
    if (shape_vec_first == nullptr) {
      break;
    }

    // channel shuffle would be decomposed as following:
    // --------reshape-first--------------transpose---------------reshape-second--------
    // n c h w --> n group c/group h w --> n c/group group h w --> n c h w
    Initializer shape_in_val(*shape_vec_first, node_unit.ModelPath());
    Initializer shape_out_val(*shape_vec_second, node_unit.ModelPath());
    auto shape_in_span = shape_in_val.DataAsSpan<int64_t>();
    auto shape_out_span = shape_out_val.DataAsSpan<int64_t>();
    if (shape_in_span.size() != shape_out_span.size() + 1 ||
        shape_in_span.size() != 5 ||
        shape_in_span[3] != shape_out_span[2] ||
        shape_in_span[4] != shape_out_span[3] ||
        (shape_out_span[1] != -1 && shape_out_span[1] != shape_in_span[1]*shape_in_span[2])) {
      break;
    }

    support = true;
  } while (false);

  return support;
}

// n h w c
ChanneShuffle::ChanneShuffle(const OpKernelInfo& info) : OpKernel{info} {
  const auto& node = info.node();
  auto input_defs = node.InputDefs();
  int x_dtype = 0;
  ORT_ENFORCE(GetType(*input_defs[0], x_dtype));
  if (x_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    op_type_ = OpComputeType::op_compute_type_fp32;
  } else if (x_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
             x_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    op_type_ = OpComputeType::op_compute_type_qu8;  // use to represent 8bit quantized data
  } else {
    auto stype = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*input_defs[0]->TypeAsProto()));
    ORT_THROW("unsupported dtype in ChanneShuffle, we have FLOAT|UINT8, but got ", stype);
  }
  const auto* x_shape = input_defs[0]->Shape();
  auto input_shape = utils::GetTensorShapeFromTensorShapeProto(*x_shape);

  int64_t group = 1;
  info.GetAttrOrDefault<int64_t>("group", &group, group);
  size_t channel = input_shape[input_shape.NumDimensions() - 1];
  size_t group_channels = channel / group;
  xnn_status xstatus = xnn_status_invalid_state;
  struct xnn_operator* p = nullptr;
  if (op_type_ == OpComputeType::op_compute_type_fp32) {
    xstatus = xnn_create_channel_shuffle_nc_x32(group, group_channels, channel, channel, 0, &p);
  } else if (op_type_ == OpComputeType::op_compute_type_qu8) {
    xstatus = xnn_create_channel_shuffle_nc_x8(group, group_channels, channel, channel, 0, &p);
  }
  ORT_ENFORCE(xstatus == xnn_status_success, "xnn_create_Transpose_nc_",
              OpTypeToString(op_type_), " failed. Status:", xstatus);
  op0_.reset(p);
}

// compute method of ChanneShuffle
Status ChanneShuffle::Compute(OpKernelContext* ctx) const {
  const auto& X = *ctx->Input<Tensor>(0);
  size_t rank = X.Shape().NumDimensions();
  InlinedVector<size_t> input1_shape(rank);
  std::transform(X.Shape().GetDims().begin(), X.Shape().GetDims().end(), input1_shape.begin(),
                 [](int64_t dim) { return gsl::narrow_cast<size_t>(dim); });
  size_t batch_size = input1_shape[0];

  TensorShape output_shape{X.Shape()};
  Tensor& Y = *ctx->Output(0, output_shape);
  if (output_shape.Size() == 0)
    return Status::OK();

  xnn_status xstatus = xnn_status_invalid_state;
  if (op_type_ == OpComputeType::op_compute_type_fp32) {
    xstatus = xnn_setup_channel_shuffle_nc_x32(
        op0_.get(), batch_size, X.DataRaw(), Y.MutableDataRaw(), nullptr);
  } else if (op_type_ == OpComputeType::op_compute_type_qu8) {
    xstatus = xnn_setup_channel_shuffle_nc_x8(
        op0_.get(), batch_size, X.DataRaw(), Y.MutableDataRaw(), nullptr);
  }
  if (xstatus != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_channel_shuffle_nc_",
                           OpTypeToString(op_type_), " returned ", xstatus);
  }
  xstatus = xnn_run_operator(op0_.get(), nullptr);
  if (xstatus != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", xstatus);
  }

  return Status::OK();
}

// {DataTypeImpl::GetTensorType<float>(),DataTypeImpl::GetTensorType<uint8_t>(), DataTypeImpl::GetTensorType<int8_t>()}),
ONNX_OPERATOR_KERNEL_EX(ChanneShuffle, kDynamicDomainByCreate, 1, kXnnpackExecutionProvider,
                        KernelDefBuilder(),  // dynamic create schema
                        ChanneShuffle);

}  // namespace xnnpack
}  // namespace onnxruntime
