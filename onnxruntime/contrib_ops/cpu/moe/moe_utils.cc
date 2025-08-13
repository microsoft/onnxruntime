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

// SwiGLU activation aligned with CUDA implementation
void ApplySwiGLUActivation(float* data, int64_t inter_size, bool is_interleaved_format,
                           float activation_alpha, float activation_beta) {
  constexpr float clamp_limit = 7.0f;

  if (is_interleaved_format) {
    // Interleaved format: [gate0, linear0, gate1, linear1, ...]
    // Output: SwiGLU(gate, linear) = gate * sigmoid(alpha * gate) * (linear + beta)
    // This exactly matches CUDA's swiglu_kernel_interleaved implementation

    for (int64_t i = 0; i < inter_size; ++i) {
      float gate_val = data[2 * i];        // Gate value at even index
      float linear_val = data[2 * i + 1];  // Linear value at odd index

      // Store original for debug
      float orig_gate = gate_val;
      float orig_linear = linear_val;

      // Apply clamping as in CUDA version
      gate_val = std::min(gate_val, clamp_limit);
      linear_val = std::clamp(linear_val, -clamp_limit, clamp_limit);

      // SwiGLU calculation: gate * sigmoid(alpha * gate) * (linear + beta)
      // This matches CUDA: swish_out * (linear + beta)
      // Use configurable activation_alpha and activation_beta parameters
      float sigmoid_arg = activation_alpha * gate_val;
      float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
      float swish_out = gate_val * sigmoid_out;

      // Store result back to position i (first half of original array)
      // Use activation_beta instead of hardcoded 1.0f
      data[i] = swish_out * (linear_val + activation_beta);
    }
  } else {
    // Non-interleaved format not implemented
    ORT_NOT_IMPLEMENTED("Non-interleaved format not supported for SwiGLU activation");
  }
}

}  // namespace contrib
}  // namespace onnxruntime
