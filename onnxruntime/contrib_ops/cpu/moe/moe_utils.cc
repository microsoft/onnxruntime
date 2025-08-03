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
      // SwiGLU: This is handled specially as it requires gating, not applied here
      return x;
    default:
      return x;  // Default to identity
  }
}

// Helper method for applying SwiGLU activation with different memory layouts - optimized version
void ApplySwiGLUActivation(float* data, int64_t inter_size, bool is_interleaved_format) {
  constexpr float swiglu_alpha = 1.702f;
  constexpr float clamp_limit = 7.0f;  // Clamping limit as specified

  if (is_interleaved_format) {
    // For interleaved format [gate, linear, gate, linear, ...], process directly
    // Optimized vectorized processing
    for (int64_t i = 0; i < inter_size; ++i) {
      const size_t gate_idx = 2 * static_cast<size_t>(i);  // Interleaved: even index (gate)
      const size_t linear_idx = gate_idx + 1;              // Interleaved: odd index (linear)

      // Load original values
      float gate_val = data[gate_idx];
      float linear_val = data[linear_idx];

      // Apply optimized clamping to the values
      gate_val = std::min(gate_val, clamp_limit);                      // Clamp gate max only
      linear_val = std::clamp(linear_val, -clamp_limit, clamp_limit);  // Clamp linear min/max

      // SwiGLU: gate * sigmoid(alpha * gate) * (linear + 1) - optimized computation
      float sigmoid_arg = swiglu_alpha * gate_val;

      // Optimized sigmoid computation using fast approximation for better performance
      // For better performance, we can use the original exact sigmoid since SIMD will handle it efficiently
      float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
      float swish_out = gate_val * sigmoid_out;
      float result = swish_out * (linear_val + 1.0f);

      // Store result in first element (output position) - optimized memory access
      data[static_cast<size_t>(i)] = result;
    }
  } else {
    // Non-interleaved format not implemented
    ORT_NOT_IMPLEMENTED("Non-interleaved format not supported for SwiGLU activation");
  }
}

}  // namespace contrib
}  // namespace onnxruntime
