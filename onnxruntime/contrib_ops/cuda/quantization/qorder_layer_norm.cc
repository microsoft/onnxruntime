#pragma warning(push)
#pragma warning(disable : 4244)

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/layer_norm.h"
#include "contrib_ops/cuda/layer_norm_impl.h"

#include "core/providers/cuda/tensor/quantize_linear.cuh"
#include "qorder_layer_norm.h"
#include "qorder_common_impl.h"
#include "qorder_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    QOrderedAddBiasResidualLayerNorm,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("F", DataTypeImpl::GetTensorType<MLFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)   // scale_X
        .InputMemoryType(OrtMemTypeCPUInput, 3)   // scale_R
        .InputMemoryType(OrtMemTypeCPUInput, 5),  // scale_Y
    QOrderedAddBiasResidualLayerNorm);

QOrderedAddBiasResidualLayerNorm::QOrderedAddBiasResidualLayerNorm(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = tmp_epsilon;
  const cublasLtOrder_t COL32orROW[] = {CUBLASLT_ORDER_COL32, CUBLASLT_ORDER_ROW};
  order_X_ = GetCublasLtOrderAttr(op_kernel_info, "order_X", 2, COL32orROW,
                                  "Only CUBLASLT_ORDER_COL32 or CUBLASLT_ORDER_ROW is supported for order_Y");
  order_R_ = GetCublasLtOrderAttr(op_kernel_info, "order_R", 2, COL32orROW,
                                  "Only CUBLASLT_ORDER_COL32 or CUBLASLT_ORDER_ROW is supported for order_Y");
  order_Y_ = GetCublasLtOrderAttr(op_kernel_info, "order_Y", 2, COL32orROW,
                                  "Only CUBLASLT_ORDER_COL32 or CUBLASLT_ORDER_ROW is supported for order_Y");
  ORT_ENFORCE(order_X_ == order_Y_);
  ORT_ENFORCE(order_X_ == order_R_);
}

Status QOrderedAddBiasResidualLayerNorm::ComputeInternal(OpKernelContext* ctx) const {
  LOCATE_ERROR_IF_ENABLED_USING_CUDA_SYNC();

  typedef typename ToCudaType<int8_t>::MappedType CudaQ;
  typedef typename ToCudaType<MLFloat16>::MappedType CudaF;

  // Inputs
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* R = ctx->Input<Tensor>(2);
  const Tensor* B = ctx->Input<Tensor>(4);
  const Tensor* gamma = ctx->Input<Tensor>(6);
  const Tensor* beta = ctx->Input<Tensor>(7);

  const auto* X_data = reinterpret_cast<const CudaQ*>(X->Data<int8_t>());
  const auto* R_data = (nullptr == R) ? nullptr : reinterpret_cast<const CudaQ*>(R->Data<int8_t>());
  const auto* B_data = (nullptr == B) ? nullptr : reinterpret_cast<const CudaF*>(B->Data<MLFloat16>());
  const auto* gamma_data = reinterpret_cast<const CudaF*>(gamma->Data<MLFloat16>());
  const auto* beta_data = (nullptr == beta) ? nullptr : reinterpret_cast<const CudaF*>(beta->Data<MLFloat16>());

  const TensorShape& x_shape = X->Shape();
  ORT_ENFORCE(x_shape.GetDims().size() == 3, "input shape must be {batch, rows, cols}");
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
  ORT_ENFORCE(axis == 2, "Currently only support on last axis}");

  int batch = gsl::narrow<int>(x_shape.GetDims()[0]);
  int64_t rows = gsl::narrow<int>(x_shape.GetDims()[1]);
  int64_t cols = gsl::narrow<int>(x_shape.GetDims()[2]);
  ORT_ENFORCE(cols != 1, "cols should not be 1");

  if (order_X_ == CUBLASLT_ORDER_COL32) {
    ORT_ENFORCE((cols & 31) == 0, "cols should be a multiple of 32");
  }

  ORT_ENFORCE((R_data && B_data) || (!R_data && !B_data));

  if (R_data && B_data) {
    ORT_ENFORCE(order_X_ == CUBLASLT_ORDER_COL32, "Residual and bias addition are only supported in COL32 ordering");
  }

  // Outputs
  Tensor* Y = ctx->Output(0, x_shape);
  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  auto* Y_data = reinterpret_cast<CudaQ*>(Y->MutableData<int8_t>());

  const float* scale_x = ctx->Input<Tensor>(1)->Data<float>();

  // Use dummy scale for R in case R is missing (We won't use it anyway)
  const auto* scale_r_tensor = ctx->Input<Tensor>(3);
  const float scale_r = (nullptr == scale_r_tensor) ? 1 : *ctx->Input<Tensor>(3)->Data<float>();

  const float* scale_y = ctx->Input<Tensor>(5)->Data<float>();

  QOrderAddBiasResidualLayerNorm(Stream(), GetDeviceProp(), (cublasLtOrder_t)order_X_,
                                 X_data, *scale_x, R_data, scale_r, B_data,
                                 Y_data, *scale_y, gamma_data, beta_data, epsilon_,
                                 batch, rows, cols);

  LOCATE_ERROR_IF_ENABLED_USING_CUDA_SYNC();
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#pragma warning(pop)
