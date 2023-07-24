// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "instance_norm.h"

namespace onnxruntime {
namespace js {

#define INSTANCE_NORM_KERNEL(op_name, domain, data_type, since_version)                            \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                                   \
      op_name,                                                                                     \
      domain,                                                                                      \
      since_version,                                                                               \
      data_type,                                                                                   \
      kJsExecutionProvider,                                                                        \
      (*KernelDefBuilder::Create()).TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
      InstanceNorm);

INSTANCE_NORM_KERNEL(InstanceNormalization, kOnnxDomain, float, 6)
INSTANCE_NORM_KERNEL(InstanceNormalization, kMSInternalNHWCDomain, float, 6)

}  // namespace js
}  // namespace onnxruntime
