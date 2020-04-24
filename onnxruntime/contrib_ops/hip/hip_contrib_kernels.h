// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/kernel_registry.h"

namespace onnxruntime {
namespace contrib {
namespace hip {

Status RegisterHipContribKernels(KernelRegistry& kernel_registry);

}  // namespace hip
}  // namespace contrib
}  // namespace onnxruntime