// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace cuda {
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();
}
}
