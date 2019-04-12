// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv_transpose_with_dynamic_pads.h"

namespace onnxruntime {
namespace cuda {
namespace contrib {
#define REGISTER_KERNEL_TYPED(T)                                                \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                \
      ConvTransposeWithDynamicPads,                                             \
      kMSDomain,                                                                \
      1,                                                                        \
      T,                                                                        \
      kCudaExecutionProvider,                                                   \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      ConvTransposeWithDynamicPads<T>);

REGISTER_KERNEL_TYPED(float)
}  // namespace contrib
}  // namespace cuda
}  // namespace onnxruntime
