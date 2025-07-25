// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/framework/kernel_registry.h"

// Forward declarations for QMoE
#include "contrib_ops/cpu/quantization/moe_quantization_cpu.h"

namespace onnxruntime {
namespace contrib {
Status RegisterCpuContribKernels(KernelRegistry& kernel_registry);
}  // namespace contrib
}  // namespace onnxruntime
