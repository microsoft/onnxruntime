// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {
namespace contrib {
void RegisterContribKernels(KernelRegistry& kernel_registry);
namespace cuda {
void RegisterCudaContribKernels(KernelRegistry& kernel_registry);
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
