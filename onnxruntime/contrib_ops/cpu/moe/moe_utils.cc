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

// Helper method for applying SwiGLU activation with different memory layouts
void ApplySwiGLUActivation(float* data, int64_t inter_size, bool is_interleaved_format) {
  constexpr float swiglu_alpha = 1.702f;
  constexpr float clamp_limit = 7.0f;  // Clamping limit as specified

  if (is_interleaved_format) {
    // Handle the interleaved format [gate, linear, gate, linear, ...]
    // The output is written to the first half of the data buffer.
    for (int64_t i = 0; i < inter_size; ++i) {
      const size_t gate_idx = 2 * i;
      const size_t linear_idx = gate_idx + 1;

      float gate_val = data[gate_idx];      // Gate is at the even index
      float linear_val = data[linear_idx];  // Linear is at the odd index

      // Apply clamping to the values
      gate_val = std::min(gate_val, clamp_limit);                              // Clamp gate max only
      linear_val = std::max(-clamp_limit, std::min(linear_val, clamp_limit));  // Clamp linear min/max

      // SwiGLU: gate * sigmoid(alpha * gate) * (linear + 1)
      float sigmoid_arg = swiglu_alpha * gate_val;
      float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
      float swish_out = gate_val * sigmoid_out;
      float result = swish_out * (linear_val + 1.0f);

      // Store the result in the first half of the buffer
      data[i] = result;
    }
  } else {
    ORT_NOT_IMPLEMENTED("SwiGLU activation is not implemented for non-interleaved format");
  }
}

}  // namespace contrib
}  // namespace onnxruntime
