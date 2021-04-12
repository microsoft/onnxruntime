// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/reduction/all.h"

namespace onnxruntime {
namespace cuda {

#define REGISTER_ALL_KERNEL_TYPED(T)                                                       \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      All,                                                                                 \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      All<T>);

template <typename T>
Status All<T>::ComputeInternal(OpKernelContext* ctx) const {
  const Tensor& input = *ctx->Input<Tensor>(0);
  Tensor& output = *ctx->Output(0, {1});

  const auto size = input.Shape().Size();
  ORT_ENFORCE(size <= std::numeric_limits<int>::max(), "Number of reduced elements (",
              size, ") exceeds the max allowed value (", std::numeric_limits<int>::max(), ").");

  // TODO: LaunchAllKernel is implemented with thrust, which always uses default CUDA stream.
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(Stream()));
  LaunchAllKernel(
      Stream(),
      input.Data<T>(),
      static_cast<int>(size),
      output.MutableData<bool>());
  CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(0));
  return Status::OK();
}

REGISTER_ALL_KERNEL_TYPED(bool)

}  // namespace cuda
}  // namespace onnxruntime
