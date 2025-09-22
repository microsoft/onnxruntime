// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cstdint>
#include "contrib_ops/cpu/moe/moe_base_cpu.h"

namespace onnxruntime {
namespace contrib {

float ApplyActivation(float x, ActivationType activation_type);

void ApplySwiGLUActivation(const float* input_data, float* output_data, int64_t inter_size, bool is_interleaved_format,
                           float activation_alpha, float activation_beta, float clamp_limit);

}  // namespace contrib
}  // namespace onnxruntime
