// Copyright (c) Microsoft Corporation. All rights reserved.
// Copyright (c) 2023 NVIDIA Corporation.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"

using namespace onnxruntime::common;

namespace onnxruntime {
namespace cuda {

Status RegisterCudaNhwcKernels(KernelRegistry& kernel_registry);

}  // namespace cuda
}  // namespace onnxruntime
