// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/kernel_registry.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

Status RegisterCpuTrainingKernels(KernelRegistry& kernel_registry);

}  // namespace contrib
}  // namespace onnxruntime
