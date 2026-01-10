// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

namespace onnxruntime {
namespace contrib {
namespace js {

Status RegisterJsContribKernels(KernelRegistry& kernel_registry);

}  // namespace js
}  // namespace contrib
}  // namespace onnxruntime
