// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/optimizer/accumulate_gradient.h"
#include "orttraining/training_ops/cuda/optimizer/gradient_control_impl.h"
#include "core/providers/cuda/math/unary_elementwise_ops_impl.h"
#include "core/providers/cuda/cuda_allocator.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_ACCUMULATE_GRADIENT_TYPED(T1, T2)                       \
  ONNX_OPERATOR_KERNEL_EX(                                               \
      AccumulateGradient,                                                \
      kMSDomain,                                                         \
      1,                                                                 \
      kCudaExecutionProvider,                                            \
      (*KernelDefBuilder::Create())                                      \
          .OutputMemoryType(OrtMemTypeCPUOutput, 0)                      \
          .MayStridedInput(0)                                            \
          .Alias(0, 1) /* Accumulate tensors in-place */                 \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T1>())       \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T2>())       \
          .TypeConstraint("T3", DataTypeImpl::GetTensorType<int64_t>()), \
      AccumulateGradient<T1, T2>);

REGISTER_ACCUMULATE_GRADIENT_TYPED(float, float);

template <typename T1, typename T2>
Status AccumulateGradient<T1, T2>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T1>::MappedType CudaT1;
  typedef typename ToCudaType<T2>::MappedType CudaT2;

  const Tensor* gradient_buffer = ctx->Input<Tensor>(0);
  const Tensor* gradient_to_add = ctx->Input<Tensor>(1);
  Tensor* updated_flag = ctx->Output(0, {});
  Tensor* accumulation_output = ctx->Output(1, gradient_buffer->Shape());
  T1* gradient_buffer_ptr = const_cast<T1*>(gradient_buffer->template Data<T1>());

  int64_t* updated_flag_ptr = updated_flag->template MutableData<int64_t>();
  *updated_flag_ptr = 1;

  size_t count = gradient_to_add->Shape().Size();
  if (gradient_buffer->IsContiguous()) {
    InPlaceAccumulatorImpl(
        Stream(),
        reinterpret_cast<CudaT1*>(gradient_buffer_ptr),
        reinterpret_cast<const CudaT2*>(gradient_to_add->template Data<T2>()),
        reinterpret_cast<CudaT1*>(gradient_buffer_ptr),
        count);

  } else {
    Impl_Cast<CudaT2, CudaT1>(
        Stream(),
        reinterpret_cast<const CudaT2*>(gradient_to_add->template Data<T2>()),
        reinterpret_cast<CudaT1*>(gradient_buffer_ptr),
        count);
  }

  if (accumulation_output) {
    const T1* source = gradient_buffer->template Data<T1>();
    T1* target = accumulation_output->template MutableData<T1>();
    if (target != source) {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target, source, gradient_buffer->SizeInBytes(),
                                           cudaMemcpyDeviceToDevice, Stream()));
    }
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
