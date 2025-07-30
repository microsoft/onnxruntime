// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cstdint>
#include "contrib_ops/cpu/moe/moe_base_cpu.h"

namespace onnxruntime {
namespace contrib {

float ApplyActivation(float x, ActivationType activation_type);
void ApplySwiGLU(const float* fc1_output, float* result, int64_t inter_size);

}  // namespace contrib
}  // namespace onnxruntime
