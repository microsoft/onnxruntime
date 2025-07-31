// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/moe/moe_utils.h"
#include <cmath>
#include <algorithm>

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
      // SwiGLU: This is handled specially as it requires gating, not applied here
      return x;
    default:
      return x;  // Default to identity
  }
}

void ApplySwiGLU(const float* fc1_output, float* result, int64_t inter_size) {
  constexpr float swiglu_alpha = 1.702f;
  for (int64_t i = 0; i < inter_size; ++i) {
    float linear_val = fc1_output[2 * i];       // Interleaved: even index
    float gate_val = fc1_output[2 * i + 1];    // Interleaved: odd index
    // SwiGLU: gate * sigmoid(alpha * gate) * (linear + 1)
    float sigmoid_arg = swiglu_alpha * gate_val;
    float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
    float swish_out = gate_val * sigmoid_out;
    result[i] = swish_out * (linear_val + 1.0f);
  }
}

}  // namespace contrib
}  // namespace onnxruntime
