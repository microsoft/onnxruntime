// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/cpu/tensor/col2im.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                              \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                              \
      Col2Im,                                                                 \
      kMSDomain,                                                              \
      1,                                                                      \
      T,                                                                      \
      kCpuExecutionProvider,                                                  \
      KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()), \
      Col2Im<T>);

REGISTER_KERNEL_TYPED(float)

}  // namespace contrib
}  // namespace onnxruntime
