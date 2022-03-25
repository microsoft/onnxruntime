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
    QOrderedLayerNormalization,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("F", DataTypeImpl::GetTensorType<MLFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)   // scale_X
        .InputMemoryType(OrtMemTypeCPUInput, 4),  // scale_Y
    QOrderedLayerNormalization);

QOrderedLayerNormalization::QOrderedLayerNormalization(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = tmp_epsilon;
  const cublasLtOrder_t COL32 = CUBLASLT_ORDER_COL32;
  GetCublasLtOrderAttr(op_kernel_info, "order_X", 1, &COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_Y");
  GetCublasLtOrderAttr(op_kernel_info, "order_Y", 1, &COL32, "Only CUBLASLT_ORDER_COL32 is supported for order_Y");
}

Status QOrderedLayerNormalization::ComputeInternal(OpKernelContext* ctx) const {
  DUBUG_PERF_CUDA_SYNC();

  typedef typename ToCudaType<int8_t>::MappedType CudaQ;
  typedef typename ToCudaType<MLFloat16>::MappedType CudaF;

  // Inputs
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* scale = ctx->Input<Tensor>(2);
  const Tensor* bias = ctx->Input<Tensor>(3);

  auto X_data = reinterpret_cast<const CudaQ*>(X->Data<int8_t>());
  auto scale_data = reinterpret_cast<const CudaF*>(scale->Data<MLFloat16>());
  auto bias_data = (nullptr == bias) ? nullptr : reinterpret_cast<const CudaF*>(bias->Data<MLFloat16>());

  const TensorShape& x_shape = X->Shape();
  ORT_ENFORCE(x_shape.GetDims().size() == 3, "input shape must be {batch, rows, cols}");
  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
  ORT_ENFORCE(axis == 2, "Currently only support on last axis}");

  int batch = gsl::narrow<int>(x_shape.GetDims()[0]);
  int64_t rows = gsl::narrow<int>(x_shape.GetDims()[1]);
  int64_t cols = gsl::narrow<int>(x_shape.GetDims()[2]);
  ORT_ENFORCE(cols != 1, "cols should not be 1");

  // Outputs
  Tensor* Y = ctx->Output(0, x_shape);
  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  auto Y_data = reinterpret_cast<CudaQ*>(Y->MutableData<int8_t>());
  const float* scale_x = ctx->Input<Tensor>(1)->Data<float>();
  const float* scale_y = ctx->Input<Tensor>(4)->Data<float>();

  QOrderLayerNorm(Stream(), GetDeviceProp(),
                  X_data, *scale_x, Y_data, *scale_y, scale_data, bias_data, epsilon_,
                  batch, rows, cols);

  DUBUG_PERF_CUDA_SYNC();
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
