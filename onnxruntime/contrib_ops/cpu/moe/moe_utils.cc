// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/moe/moe_utils.h"
#include <cmath>
#include <algorithm>

namespace onnxruntime {
namespace contrib {
namespace moe {

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
    case ActivationType::Swiglu:
      // Swiglu: This is handled specially as it requires gating, not applied here
      return x;
    default:
      return x;  // Default to identity
  }
}

// Helper method for applying Swiglu activation with different memory layouts
void ApplySwiGLUActivation(float* data, int64_t inter_size, bool is_interleaved_format) {
  constexpr float swiglu_alpha = 1.702f;
  constexpr float clamp_limit = 7.0f;  // Clamping limit as specified

  if (is_interleaved_format) {
    // For interleaved format [linear, gate, linear, gate, ...], process directly
    // Make a temporary copy of each pair of values before modifying them
    for (int64_t i = 0; i < inter_size; ++i) {
      const size_t idx = static_cast<size_t>(i);
      const size_t linear_idx = 2 * idx;
      const size_t gate_idx = linear_idx + 1;

      // Store original values
      float linear_val = data[linear_idx];  // Interleaved: even index
      float gate_val = data[gate_idx];      // Interleaved: odd index

      // Apply clamping to the values
      if (gate_val > clamp_limit) gate_val = clamp_limit;      // Clamp gate max only
      if (linear_val > clamp_limit) linear_val = clamp_limit;  // Clamp linear min/max
      if (linear_val < -clamp_limit) linear_val = -clamp_limit;

      // Swiglu: gate * sigmoid(alpha * gate) * (linear + 1)
      float sigmoid_arg = swiglu_alpha * gate_val;
      float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
      float swish_out = gate_val * sigmoid_out;
      float result = swish_out * (linear_val + 1.0f);

      // Store result in first element (linear position)
      data[idx] = result;
    }
  } else {
    // For chunked layout [linear..., gate...], handle separately
    // Need to work with original data in-place
    // First, store all the gate computations since they depend on original gate values
    std::vector<float> computed_gates(static_cast<size_t>(inter_size));

    for (int64_t i = 0; i < inter_size; ++i) {
      const size_t idx = static_cast<size_t>(i);
      float gate_val = data[idx + static_cast<size_t>(inter_size)];

      // Apply clamping to the gate value (max only)
      if (gate_val > clamp_limit) gate_val = clamp_limit;

      // Compute the gate part of Swiglu
      float sigmoid_arg = swiglu_alpha * gate_val;
      float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
      computed_gates[idx] = gate_val * sigmoid_out;
    }

    // Now apply the full activation with the precomputed gate values
    for (int64_t i = 0; i < inter_size; ++i) {
      const size_t idx = static_cast<size_t>(i);
      float linear_val = data[idx];

      // Apply clamping to the linear value (min/max)
      if (linear_val > clamp_limit) linear_val = clamp_limit;
      if (linear_val < -clamp_limit) linear_val = -clamp_limit;

      data[idx] = computed_gates[idx] * (linear_val + 1.0f);
    }
  }
}

}  // namespace moe
}  // namespace contrib
}  // namespace onnxruntime
