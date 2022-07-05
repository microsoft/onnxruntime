// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/cpu/tensor/grid_sample.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL_TYPED(T)                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                    \
      GridSample,                                                   \
      kMSDomain,                                                    \
      1,                                                            \
      T,                                                            \
      kCpuExecutionProvider,                                        \
      KernelDefBuilder()                                            \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<T>())   \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),  \
      GridSample<T>);

REGISTER_KERNEL_TYPED(float)

}  // namespace contrib
}  // namespace onnxruntime
