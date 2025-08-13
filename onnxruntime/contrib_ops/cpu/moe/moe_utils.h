// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cstdint>
#include "contrib_ops/cpu/moe/moe_base_cpu.h"

namespace onnxruntime {
namespace contrib {

float ApplyActivation(float x, ActivationType activation_type);
void ApplySwiGLUActivation(float* data, int64_t inter_size, bool is_interleaved_format,
                           float activation_alpha = 1.702f, float activation_beta = 1.0f);

}  // namespace contrib
}  // namespace onnxruntime
