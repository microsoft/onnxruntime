// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/hip/math/binary_elementwise_ops.h"
#include "core/providers/hip/reduction/reduction_functions.h"
#include "core/providers/hip/hip_allocator.h"
#include "gradient_control.h"

namespace onnxruntime {
namespace hip {

#define REGISTER_GRADIENT_ACCUMULATOR_TYPED(T, T_GRAD)                      \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                            \
      GradientAccumulator,                                                  \
      kOnnxDomain,                                                          \
      9,                                                                    \
      T##_##T_GRAD,                                                         \
      kHipExecutionProvider,                                               \
      KernelDefBuilder()                                                    \
          .Alias(0, 0) /* Accumulate gradients in-place */                  \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())            \
          .TypeConstraint("T_GRAD", DataTypeImpl::GetTensorType<T_GRAD>()), \
      AccumulateGradient<T, T_GRAD>);

REGISTER_GRADIENT_ACCUMULATOR_TYPED(float, float)
REGISTER_GRADIENT_ACCUMULATOR_TYPED(float, MLFloat16)

template <typename T>
Status ZeroGradient<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& old_gradient = *ctx->Input<Tensor>(0);
  Tensor& zero_gradient = *ctx->Output(0, old_gradient.Shape());

  HIP_RETURN_IF_ERROR(hipMemsetAsync(
    zero_gradient.template MutableData<T>(),
    0,
    zero_gradient.Shape().Size() * sizeof(T)));

  return Status::OK();
}

#define REGISTER_ZERO_GRADIENT_TYPED(T)                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ZeroGradient,                                               \
      kOnnxDomain,                                                \
      9,                                                          \
      T,                                                          \
      kHipExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .Alias(0, 0) /* Zero out gradients in-place */          \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", DataTypeImpl::AllTensorTypes()),  \
      ZeroGradient<T>);
REGISTER_ZERO_GRADIENT_TYPED(float)
REGISTER_ZERO_GRADIENT_TYPED(MLFloat16)

template <typename T, typename T_GRAD>
Status AccumulateGradient<T, T_GRAD>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToHipType<T>::MappedType HipT;
  typedef typename ToHipType<T_GRAD>::MappedType HipT_GRAD;

  const Tensor& gradient_buffer = *ctx->Input<Tensor>(0);
  const Tensor& gradient = *ctx->Input<Tensor>(1);
  Tensor& accumulated_gradient = *ctx->Output(0, gradient_buffer.Shape());

  AccumulateGradientImpl(
      reinterpret_cast<const HipT*>(gradient_buffer.template Data<T>()),
      reinterpret_cast<const HipT_GRAD*>(gradient.template Data<T_GRAD>()),
      reinterpret_cast<HipT*>(accumulated_gradient.template MutableData<T>()),
      gradient.Shape().Size());

  return Status::OK();
}

}  // namespace hip
}  // namespace onnxruntime
