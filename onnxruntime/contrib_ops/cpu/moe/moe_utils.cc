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
void ApplySwiGLUActivation(float* data, int64_t inter_size, bool is_interleaved_format) {
  constexpr float swiglu_alpha = 1.702f;
  constexpr float clamp_limit = 7.0f;

  if (is_interleaved_format) {
    // Interleaved format: [gate0, linear0, gate1, linear1, ...]
    // Output: SwiGLU(gate, linear) = gate * sigmoid(alpha * gate) * (linear + 1)
    // This exactly matches CUDA's swiglu_kernel_interleaved implementation

    // Debug: Show first few inputs before SwiGLU
    // static thread_local int debug_count = 0;
    // if (debug_count < 3) {
    //   printf("    SwiGLU before: inter_size=%ld, data[0:6]=[%f, %f, %f, %f, %f, %f]\n",
    //          inter_size, data[0], data[1], data[2], data[3], data[4], data[5]);
    //   debug_count++;
    // }

    for (int64_t i = 0; i < inter_size; ++i) {
      float gate_val = data[2 * i];        // Gate value at even index
      float linear_val = data[2 * i + 1];  // Linear value at odd index

      // Store original for debug
      float orig_gate = gate_val;
      float orig_linear = linear_val;

      // Apply clamping as in CUDA version
      gate_val = std::min(gate_val, clamp_limit);
      linear_val = std::clamp(linear_val, -clamp_limit, clamp_limit);

      // SwiGLU calculation: gate * sigmoid(alpha * gate) * (linear + 1)
      // This matches CUDA: swish_out * (linear + 1.f)
      float sigmoid_arg = swiglu_alpha * gate_val;
      float sigmoid_out = 1.0f / (1.0f + std::exp(-sigmoid_arg));
      float swish_out = gate_val * sigmoid_out;

      // Store result back to position i (first half of original array)
      data[i] = swish_out * (linear_val + 1.0f);

      // Debug first few computations
      // if (debug_count <= 3 && i < 3) {
      //   printf("      SwiGLU[%ld]: gate=%f->%f, linear=%f->%f, sigmoid_arg=%f, sigmoid=%f, swish=%f, result=%f\n",
      //          i, orig_gate, gate_val, orig_linear, linear_val, sigmoid_arg, sigmoid_out, swish_out, data[i]);
      // }
    }

    // Debug: Show first few outputs after SwiGLU
    // if (debug_count <= 3) {
    //   printf("    SwiGLU after: data[0:3]=[%f, %f, %f]\n", data[0], data[1], data[2]);
    // }
  } else {
    // Non-interleaved format not implemented
    ORT_NOT_IMPLEMENTED("Non-interleaved format not supported for SwiGLU activation");
  }
}

}  // namespace contrib
}  // namespace onnxruntime
