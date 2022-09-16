// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "relu.h"

namespace onnxruntime {
namespace vulkan {

ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(
    Relu,
    kOnnxDomain,
    6,
    12,
    float,
    kVulkanExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", BuildKernelDefConstraints<float>()),
    Relu);

Status Relu::Compute(OpKernelContext* /*ctx*/) const {
  int a = 2;
  ORT_IGNORE_RETURN_VALUE(a);
  return Status::OK();
}

}  // namespace vulkan
}  // namespace onnxruntime
