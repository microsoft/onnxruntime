// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "instance_norm.h"

namespace onnxruntime {
namespace js {

#define INSTANCE_NORM_KERNEL(op_name, domain, since_version, is_channels_last)      \
  ONNX_OPERATOR_KERNEL_EX(                                                          \
      op_name,                                                                      \
      domain,                                                                       \
      since_version,                                                                \
      kJsExecutionProvider,                                                         \
      (*KernelDefBuilder::Create()).TypeConstraint("T", JsepSupportedFloatTypes()), \
      InstanceNorm<is_channels_last>);

INSTANCE_NORM_KERNEL(InstanceNormalization, kOnnxDomain, 6, false)
INSTANCE_NORM_KERNEL(InstanceNormalization, kMSInternalNHWCDomain, 6, true)

}  // namespace js
}  // namespace onnxruntime
