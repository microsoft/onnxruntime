// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {
namespace featurizers {

Status RegisterCpuMSFeaturizersKernels(KernelRegistry& kernel_registry);

}  // namespace featurizers
}  // namespace onnxruntime
