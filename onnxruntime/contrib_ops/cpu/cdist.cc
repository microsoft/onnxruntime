// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cdist.h"

namespace onnxruntime {
namespace contrib {
#define DEFINE_KERNEL(data_type)                                                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(CDist, kMSDomain, 1, data_type, kCpuExecutionProvider,                            \
                                KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
                                CDist<data_type>);
DEFINE_KERNEL(float);
DEFINE_KERNEL(double);

}  // namespace contrib
}  // namespace onnxruntime