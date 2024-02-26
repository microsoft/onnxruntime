// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/tensor/gelu.h"
#include "core/providers/cuda/tensor/gelu_impl.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      Gelu,                                                      \
      kOnnxDomain,                                               \
      20,                                                        \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      (*KernelDefBuilder::Create())                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .MayInplace(0, 0),                                     \
      Gelu<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(double)

template <typename T>
Status Gelu<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const auto& input_dims = input->Shape().GetDims();
  if (input_dims.size() < 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Input 0 is expected to have 1 or more dimensions, got ", input_dims.size());
  }

  Tensor* output = context->Output(0, input->Shape());

  int64_t input_length = input->Shape().Size();
  if (input_length == 0) {
    return Status::OK();
  }

  typedef typename ToCudaType<T>::MappedType CudaT;

  if (approximation_algorithm_ == "tanh") {
    return LaunchFastGeluKernel<CudaT>(GetDeviceProp(),
                                       Stream(context),
                                       static_cast<int>(input_length),
                                       0 /* no bias */,
                                       reinterpret_cast<const CudaT*>(input->Data<T>()),
                                       nullptr /* no bias */,
                                       reinterpret_cast<CudaT*>(output->MutableData<T>()),
                                       use_half2_);
  } else if (approximation_algorithm_ == "none") {
    return LaunchGeluKernel<CudaT>(Stream(context),
                                   reinterpret_cast<const CudaT*>(input->Data<T>()),
                                   reinterpret_cast<CudaT*>(output->MutableData<T>()),
                                   static_cast<size_t>(input_length));
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Unsupported approximation_algorithm: ", approximation_algorithm_);
}

}  // namespace cuda

#ifndef DISABLE_CONTRIB_OPS
namespace contrib::cuda {
#define REGISTER_CONTRIB_KERNEL_TYPED(T)                         \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                 \
      Gelu,                                                      \
      kMSDomain,                                                 \
      1,                                                         \
      T,                                                         \
      kCudaExecutionProvider,                                    \
      (*KernelDefBuilder::Create())                              \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()) \
          .MayInplace(0, 0),                                     \
      onnxruntime::cuda::Gelu<T>);

REGISTER_CONTRIB_KERNEL_TYPED(float)
REGISTER_CONTRIB_KERNEL_TYPED(MLFloat16)
REGISTER_CONTRIB_KERNEL_TYPED(double)

#undef REGISTER_CONTRIB_KERNEL_TYPED
}  // namespace contrib::cuda
#endif

}  // namespace onnxruntime
