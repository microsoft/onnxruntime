// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "fft_ops.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
#define REGISTER_KERNEL_TYPED(T)                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                  \
      FFT,                                                        \
      kMSDomain,                                                  \
      1,                                                          \
      T,                                                          \
      kCudaExecutionProvider,                                     \
      KernelDefBuilder()                                          \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      FFT<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(double)
REGISTER_KERNEL_TYPED(MLFloat16)

template <typename T>
Status FFT<T>::ComputeInternal(OpKernelContext* context) const {
  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
