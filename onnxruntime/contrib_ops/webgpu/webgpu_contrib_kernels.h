// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/kernel_registry.h"

namespace onnxruntime {
namespace contrib {
namespace webgpu {

// forward declaration for this EP's namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

Status RegisterWebGpuContribKernels(KernelRegistry& kernel_registry);

}  // namespace webgpu
}  // namespace contrib
}  // namespace onnxruntime
