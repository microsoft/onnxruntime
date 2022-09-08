// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/nn/transpose.h"

#include <utility>

#include "core/framework/op_kernel.h"
#include "xnnpack.h"

namespace onnxruntime {
namespace xnnpack {


bool Transpose::IsOnnxNodeSupported(const NodeUnit&,
                                    const GraphViewer&) {
  return true;
}

Transpose::Transpose(const OpKernelInfo& info) : OpKernel{info}, TransposeBase(info) {
  const auto& node = info.node();
  auto input_defs = node.InputDefs();
  int x_dtype = 0;
  ORT_ENFORCE(GetType(*input_defs[0], x_dtype));
  if (x_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    op_type_ = OpComputeType::op_compute_type_fp32;
  } else if (x_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
             x_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    op_type_ = OpComputeType::op_compute_type_qu8; // use to represent 8bit quantized data
  } else {
    auto stype = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*input_defs[0]->TypeAsProto()));
    ORT_THROW("unsupported Conv in Transpose, we have FLOAT|UINT8, but got ", stype);
  }


  xnn_status xstatus = xnn_status_invalid_state;
  struct xnn_operator* p = nullptr;
  if (op_type_ == OpComputeType::op_compute_type_qu8) {
    xstatus = xnn_create_transpose_nd_x8(0, &p);
  } else if (op_type_ == OpComputeType::op_compute_type_fp32) {
    xstatus = xnn_create_transpose_nd_x32(0, &p);
  }
  ORT_ENFORCE(xstatus == xnn_status_success, "xnn_create_Transpose_nc_",
              OpTypeToString(op_type_), " failed. Status:", xstatus);
  op0_.reset(p);
}

// compute method of Transpose
Status Transpose::Compute(OpKernelContext* ctx) const {
  const auto& X = *ctx->Input<Tensor>(0);
  size_t rank = X.Shape().NumDimensions();
  InlinedVector<size_t> input1_shape(rank);
  std::transform(X.Shape().GetDims().begin(), X.Shape().GetDims().end(), input1_shape.begin(),
                 [](int64_t dim) { return gsl::narrow_cast<size_t>(dim); });

  TensorShapeVector output_dims(rank);
  const InlinedVector<size_t>* p_perm;
  InlinedVector<size_t> default_perm(rank);
  Status status = ComputeOutputShape(X, output_dims, default_perm, p_perm);
  if (!status.IsOK())
    return status;

  TensorShape output_shape{output_dims};
  Tensor& Y = *ctx->Output(0, output_shape);
  if (output_shape.Size() == 0)
    return Status::OK();

  xnn_status xstatus = xnn_status_invalid_state;
  if (op_type_ == OpComputeType::op_compute_type_qu8) {
    xstatus = xnn_setup_transpose_nd_x8(
        op0_.get(), X.DataRaw(), Y.MutableDataRaw(),
        rank, input1_shape.data(), p_perm->data(), nullptr);
  }
  else if (op_type_ == OpComputeType::op_compute_type_fp32) {
    xstatus = xnn_setup_transpose_nd_x32(op0_.get(), X.Data<float>(), Y.MutableData<float>(),
                                         rank, input1_shape.data(), p_perm->data(), nullptr);
  }
  if (xstatus != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_convolution2d_nhwc_",
                           OpTypeToString(op_type_), " returned ", status);
  }
  xstatus = xnn_run_operator(op0_.get(), nullptr);
  if (xstatus != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Transpose, kOnnxDomain, 1, 12, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                          DataTypeImpl::GetTensorType<uint8_t>(),
                                                                          DataTypeImpl::GetTensorType<int8_t>()}),
                                  Transpose);
ONNX_OPERATOR_KERNEL_EX(Transpose, kOnnxDomain, 13, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                DataTypeImpl::GetTensorType<uint8_t>(),
                                                                DataTypeImpl::GetTensorType<int8_t>()}),
                        Transpose);

}  // namespace xnnpack
}  // namespace onnxruntime
