// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/webgpu/div_ceil.h"
#include "contrib_ops/webgpu/webgpu_contrib_kernels.h"

#include "core/providers/webgpu/webgpu_supported_types.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

using namespace onnxruntime::webgpu;

ONNX_OPERATOR_KERNEL_EX(
    DivCeil,
    kMSDomain,
    1,
    kWebGpuExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", WebGpuSupportedFloatTypes()),
    DivCeil);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
