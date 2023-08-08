// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/span_utils.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/fpgeneric.h"
#include "core/providers/cuda/tensor/slice.h"
#include "core/providers/cuda/nn/conv.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                                           \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                           \
      NhwcConv,                                                                            \
      kMSDomain,                                                                           \
      1,                                                                                   \
      T,                                                                                   \
      kCudaExecutionProvider,                                                              \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), \
      Conv<T, true>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
