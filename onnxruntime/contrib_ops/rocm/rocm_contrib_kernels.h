// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/kernel_registry.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

Status RegisterRocmContribKernels(KernelRegistry& kernel_registry);

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime