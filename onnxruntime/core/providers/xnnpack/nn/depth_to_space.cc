// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/nn/depth_to_space.h"

#include <string>
#include <utility>

#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/graph/constants.h"
#include "xnnpack.h"

namespace onnxruntime {
namespace xnnpack {

bool DepthToSpace::IsOnnxNodeSupported(const NodeUnit& node_unit,
                                       const GraphViewer&) {
  bool support = false;
  const auto& node = node_unit.GetNode();
  do {
    // AveragePool has 1 input.
    const auto& x_arg = node_unit.Inputs()[0].node_arg;

    // we only support 2D (4 dims with batch and channel)
    const auto* x_shape = x_arg.Shape();
    if (!x_shape || x_shape->dim_size() != 4) {
      break;
    }
    // we only support float and u8|s8 currently
    const auto* x_type = x_arg.TypeAsProto();
    if (x_type == nullptr ||
        (x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
         x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8 &&
         x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8)) {
      break;
    }
    ProtoHelperNodeContext nc(node);
    OpNodeProtoHelper info(&nc);
    std::string mode = info.GetAttrOrDefault<std::string>("mode", "DCR");
    if (mode != "DCR") {
      break;
    }
    int64_t blocksize = 1;
    if (!info.GetAttr("blocksize", &blocksize).IsOK() && blocksize <= 1) {
      break;
    }
    support = true;
  } while (false);
  return support;
}

DepthToSpace::DepthToSpace(const OpKernelInfo& info) : OpKernel{info}, SpaceDepthBase(info) {
  ORT_ENFORCE(info.GetAttr("blocksize", &blocksize_).IsOK(),
              "Attribute blocksize is not set.");

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
    ORT_THROW("unsupported dtype in DepthToSpace, we have FLOAT|UINT8, but got ", stype);
  }

  auto input_shape = utils::GetTensorShapeFromTensorShapeProto(*input_defs[0]->Shape());
  auto inferred_output_shape = utils::GetTensorShapeFromTensorShapeProto(*Node().OutputDefs()[0]->Shape());
  ORT_ENFORCE(inferred_output_shape[1] * inferred_output_shape[2] /
                      blocksize_ / blocksize_ ==
                  input_shape[1] * input_shape[2],
              "Shape mismatch between inferred value and calculated value.");
  size_t input_channel_stride = gsl::narrow_cast<size_t>(input_shape[3]);
  size_t output_channels = gsl::narrow_cast<size_t>(input_channel_stride / blocksize_ / blocksize_);
  size_t output_channel_stride = output_channels;
  uint32_t block_size = gsl::narrow_cast<uint32_t>(blocksize_);

  xnn_status xstatus = xnn_status_invalid_state;
  struct xnn_operator* p = nullptr;
  if (op_type_ == OpComputeType::op_compute_type_fp32) {
    xstatus = xnn_create_depth_to_space_nhwc_x32(
        output_channels, input_channel_stride, output_channel_stride, block_size, 0, &p);
  } else if (op_type_ == OpComputeType::op_compute_type_qu8) {
    xstatus = xnn_create_depth_to_space_nhwc_x8(
        output_channels, input_channel_stride, output_channel_stride, block_size, 0, &p);
  }
  ORT_ENFORCE(xstatus == xnn_status_success, "xnn_create_depth_to_space_nhwc_",
              OpTypeToString(op_type_), " failed. Status:", xstatus);
  op0_.reset(p);
}

// compute method of DepthToSpace
Status DepthToSpace::Compute(OpKernelContext* ctx) const {
  const auto& X = *ctx->Input<Tensor>(0);
  const TensorShape& input_shape = X.Shape();

  int64_t batch_size = input_shape[0];
  int64_t input_depth = input_shape[3];
  int64_t input_height = input_shape[1];
  int64_t input_width = input_shape[2];

  int64_t output_depth = -1;
  int64_t output_height = -1;
  int64_t output_width = -1;

  ORT_RETURN_IF_ERROR(ComputeOutputShape(
      input_depth, input_height, input_width,
      output_depth, output_height, output_width,
      false));
  TensorShape output_shape{batch_size, output_height, output_width, output_depth};
  Tensor& Y = *ctx->Output(0, output_shape);

  xnn_status xstatus = xnn_status_invalid_state;
  if (op_type_ == OpComputeType::op_compute_type_fp32) {
    xstatus = xnn_setup_depth_to_space_nhwc_x32(
        op0_.get(), batch_size, input_height, input_width,
        X.Data<float>(), Y.MutableData<float>(), nullptr);
  } else if (op_type_ == OpComputeType::op_compute_type_qu8) {
    xstatus = xnn_setup_depth_to_space_nhwc_x8(
        op0_.get(), batch_size, input_height, input_width,
        X.DataRaw(), Y.MutableDataRaw(), nullptr);
  }

  if (xstatus != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_depth_to_space_nhwc_",
                           OpTypeToString(op_type_), " returned ", xstatus);
  }
  xstatus = xnn_run_operator(op0_.get(), nullptr);
  if (xstatus != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", xstatus);
  }

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(DepthToSpace, kMSInternalNHWCDomain, 1, 10, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                          DataTypeImpl::GetTensorType<uint8_t>(),
                                                                          DataTypeImpl::GetTensorType<int8_t>()}),
                                  DepthToSpace);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(DepthToSpace, kMSInternalNHWCDomain, 11, 12, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                          DataTypeImpl::GetTensorType<uint8_t>(),
                                                                          DataTypeImpl::GetTensorType<int8_t>()}),
                                  DepthToSpace);
ONNX_OPERATOR_KERNEL_EX(DepthToSpace, kMSInternalNHWCDomain, 13, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                DataTypeImpl::GetTensorType<uint8_t>(),
                                                                DataTypeImpl::GetTensorType<int8_t>()}),
                        DepthToSpace);

}  // namespace xnnpack
}  // namespace onnxruntime
