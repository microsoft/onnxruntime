// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/moe/moe_utils.h"
#include <cmath>
#include <algorithm>
#include "core/common/common.h"

namespace onnxruntime {
namespace contrib {

float ApplyActivation(float x, ActivationType activation_type) {
  switch (activation_type) {
    case ActivationType::Relu:
      return std::max(0.0f, x);
    case ActivationType::Gelu:
      return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    case ActivationType::Silu:
      return x * (1.0f / (1.0f + std::exp(-x)));
    case ActivationType::Identity:
      return x;
    case ActivationType::SwiGLU:
      // SwiGLU is a special case handled by ApplySwiGLUActivation, this is just a placeholder
      return x;
    default:
      return x;
  }
}

void ApplySwiGLUActivation(const float* input_data, float* output_data, int64_t inter_size, bool is_interleaved_format,
                           float activation_alpha, float activation_beta, float clamp_limit) {
  if (is_interleaved_format) {
    for (int64_t i = 0; i < inter_size; ++i) {
      float gate_val = input_data[2 * i];
      float linear_val = input_data[2 * i + 1];

      gate_val = std::min(gate_val, clamp_limit);
      linear_val = std::clamp(linear_val, -clamp_limit, clamp_limit);

      float sigmoid_arg = activation_alpha * gate_val;
      float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
      float swish_out = gate_val * sigmoid_out;

      output_data[i] = swish_out * (linear_val + activation_beta);
    }
  } else {
    ORT_NOT_IMPLEMENTED("Non-interleaved format not supported for SwiGLU activation");
  }
}

}  // namespace contrib
}  // namespace onnxruntime
