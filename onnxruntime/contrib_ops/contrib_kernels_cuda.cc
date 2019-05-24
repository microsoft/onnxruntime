// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/contrib_kernels.h"
#include "core/graph/constants.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ConvTransposeWithDynamicPads);
class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, Resize);

void RegisterCudaContribKernels(KernelRegistry& kernel_registry) {
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, ConvTransposeWithDynamicPads)>());
  kernel_registry.Register(BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(kCudaExecutionProvider, kMSDomain, 1, float, Resize)>());
}
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
