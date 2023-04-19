// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gradient_control.h"
#include "gradient_control_impl.h"

#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/reduction/reduction_functions.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "common.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(T, T_GRAD)                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                    \
      InPlaceAccumulator,                                                           \
      kMSDomain,                                                                    \
      1,                                                                            \
      T##_##T_GRAD,                                                                 \
      kCudaExecutionProvider,                                                       \
      (*KernelDefBuilder::Create())                                                 \
          .Alias(0, 0)                            /* Accumulate tensors in-place */ \
          .InputMemoryType(OrtMemTypeCPUInput, 2) /* Keep do_update in CPU */       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                    \
          .TypeConstraint("T_GRAD", DataTypeImpl::GetTensorType<T_GRAD>()),         \
      InPlaceAccumulator<T, T_GRAD>);

REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(float, float)
REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(float, MLFloat16)
REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(MLFloat16, MLFloat16)
REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(MLFloat16, float)
REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(float, BFloat16)
REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(BFloat16, BFloat16)
REGISTER_IN_PLACE_TENSOR_ACCUMULATOR_TYPED(BFloat16, float)

template <typename T>
Status ZeroGradient<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& old_gradient = *ctx->Input<Tensor>(0);
  Tensor& zero_gradient = *ctx->Output(0, old_gradient.Shape());

  CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
      zero_gradient.template MutableData<T>(),
      0,
      zero_gradient.Shape().Size() * sizeof(T), Stream(ctx)));

  return Status::OK();
}

#define REGISTER_ZERO_GRADIENT_TYPED(T)                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      ZeroGradient,                                               \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      (*KernelDefBuilder::Create())                               \
          .Alias(0, 0) /* Zero out gradients in-place */          \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()) \
          .TypeConstraint("T2", DataTypeImpl::AllTensorTypes()),  \
      ZeroGradient<T>);
REGISTER_ZERO_GRADIENT_TYPED(float)
REGISTER_ZERO_GRADIENT_TYPED(MLFloat16)

template <typename T, typename T_GRAD>
Status InPlaceAccumulator<T, T_GRAD>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<T_GRAD>::MappedType CudaT_GRAD;

  const Tensor& left_addee_buffer = *ctx->Input<Tensor>(0);
  const Tensor& right_addee_buffer = *ctx->Input<Tensor>(1);
  const Tensor* do_update_tensor = ctx->Input<Tensor>(2);
  Tensor& accumulation_output = *ctx->Output(0, left_addee_buffer.Shape());

  if (do_update_tensor) {
    const bool do_update = *(do_update_tensor->template Data<bool>());
    if (!do_update) {
      ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T>(Stream(ctx), left_addee_buffer, accumulation_output));
      return Status::OK();
    }
  }

  InPlaceAccumulatorImpl(
      Stream(ctx),
      reinterpret_cast<const CudaT*>(left_addee_buffer.template Data<T>()),
      reinterpret_cast<const CudaT_GRAD*>(right_addee_buffer.template Data<T_GRAD>()),
      reinterpret_cast<CudaT*>(accumulation_output.template MutableData<T>()),
      right_addee_buffer.Shape().Size());

  return Status::OK();
}

#define REGISTER_IN_PLACE_TENSOR_ACCUMULATORV2_TYPED(T, T_GRAD)                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                      \
      InPlaceAccumulatorV2,                                                           \
      kMSDomain,                                                                      \
      1,                                                                              \
      T##_##T_GRAD,                                                                   \
      kCudaExecutionProvider,                                                         \
      (*KernelDefBuilder::Create())                                                   \
          .Alias(0, 1)                              /* Accumulate tensors in-place */ \
          .InputMemoryType(OrtMemTypeCPUInput, 2)   /* overwrite_flag is on CPU*/     \
          .OutputMemoryType(OrtMemTypeCPUOutput, 0) /* updated_flag is on CPU*/       \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())                      \
          .TypeConstraint("T_GRAD", DataTypeImpl::GetTensorType<T_GRAD>()),           \
      InPlaceAccumulatorV2<T, T_GRAD>);

REGISTER_IN_PLACE_TENSOR_ACCUMULATORV2_TYPED(float, float)
REGISTER_IN_PLACE_TENSOR_ACCUMULATORV2_TYPED(float, MLFloat16)

template <typename T, typename T_GRAD>
Status InPlaceAccumulatorV2<T, T_GRAD>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<T_GRAD>::MappedType CudaT_GRAD;

  Tensor& left_addee_buffer = *const_cast<Tensor*>(ctx->Input<Tensor>(0));
  const Tensor& right_addee_buffer = *ctx->Input<Tensor>(1);
  const Tensor* overwrite_tensor = ctx->Input<Tensor>(2);
  const bool overwrite = overwrite_tensor != nullptr ? *(overwrite_tensor->template Data<bool>()) : false;

  if (overwrite) {
    const T_GRAD* source = right_addee_buffer.template Data<T_GRAD>();
    T* target = left_addee_buffer.template MutableData<T>();
    if (std::is_same<T, T_GRAD>::value) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target, source, right_addee_buffer.SizeInBytes(), cudaMemcpyDeviceToDevice,
                                           Stream(ctx)));
    } else {
      Impl_Cast<CudaT_GRAD, CudaT>(
          Stream(ctx),
          reinterpret_cast<const CudaT_GRAD*>(source),
          reinterpret_cast<CudaT*>(target),
          right_addee_buffer.Shape().Size());
    }
  } else {
    InPlaceAccumulatorImpl(
        Stream(ctx),
        reinterpret_cast<const CudaT*>(left_addee_buffer.template Data<T>()),
        reinterpret_cast<const CudaT_GRAD*>(right_addee_buffer.template Data<T_GRAD>()),
        reinterpret_cast<CudaT*>(left_addee_buffer.template MutableData<T>()),
        right_addee_buffer.Shape().Size());
  }

  Tensor& updated_output = *ctx->Output(0, {1});
  bool* updated_output_ptr = updated_output.MutableData<bool>();
  *updated_output_ptr = true;

  Tensor* accumulation_output = ctx->Output(1, left_addee_buffer.Shape());
  if (nullptr != accumulation_output) {
    ORT_RETURN_IF_ERROR(CopyIfNotSameBuffer<T>(Stream(ctx), left_addee_buffer, *accumulation_output));
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
