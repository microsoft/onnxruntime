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
  // Create a temporary buffer for the result
  auto result_buffer = std::make_unique<float[]>(static_cast<size_t>(inter_size));

  if (is_interleaved_format) {
    // For interleaved format [linear, gate, linear, gate, ...], process directly
    for (int64_t i = 0; i < inter_size; ++i) {
      float linear_val = data[2 * static_cast<size_t>(i)];    // Interleaved: even index
      float gate_val = data[2 * static_cast<size_t>(i) + 1];  // Interleaved: odd index

      // SwiGLU: gate * sigmoid(alpha * gate) * (linear + 1)
      float sigmoid_arg = swiglu_alpha * gate_val;
      float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
      float swish_out = gate_val * sigmoid_out;
      result_buffer[static_cast<size_t>(i)] = swish_out * (linear_val + 1.0f);
    }
  } else {
    // For chunked layout [linear..., gate...], handle separately
    float* linear_part = data;
    float* gate_part = data + static_cast<size_t>(inter_size);

    for (int64_t i = 0; i < inter_size; ++i) {
      float linear_val = linear_part[static_cast<size_t>(i)];
      float gate_val = gate_part[static_cast<size_t>(i)];

      // SwiGLU: gate * sigmoid(alpha * gate) * (linear + 1)
      float sigmoid_arg = swiglu_alpha * gate_val;
      float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
      float swish_out = gate_val * sigmoid_out;
      result_buffer[static_cast<size_t>(i)] = swish_out * (linear_val + 1.0f);
    }
  }

  // Copy result back to data (first inter_size elements only - rest is overwritten by GEMM)
  std::memcpy(data, result_buffer.get(), static_cast<size_t>(inter_size) * sizeof(float));
}

}  // namespace contrib
}  // namespace onnxruntime
