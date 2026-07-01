/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <limits>

namespace onnxruntime::llm::kernels::cutlass_kernels {

// IMPORTANT: Keep the same order of activation functions in this enum and the activation functions in
// moe_gemm_activation_kernels.cuh::doActivationKernel().
// Note: Update moe.py to match if modifying.
enum class ActivationType {
  InvalidType = 0,
  Identity = 1,
  Gelu = 2,
  Relu = 3,
  Silu = 4,
  Swiglu = 5,
  Geglu = 6,
  SwigluBias = 7,
  Relu2 = 8,
};

// Matches TensorRT-LLM ActivationParams structure with backward compatibility.
// Per-expert pointers (swiglu_alpha, swiglu_beta, swiglu_limit) for advanced use.
// Scalar defaults (alpha, beta, limit, fusion) for existing kernel compatibility.
struct ActivationParams {
  ActivationType activation_type = ActivationType::Identity;

  // Per-expert arrays (TRT-LLM style) - nullptr means use scalar defaults
  float const* swiglu_alpha = nullptr;  // Per-expert scaling for gate
  float const* swiglu_beta = nullptr;   // Per-expert bias for linear
  float const* swiglu_limit = nullptr;  // Per-expert activation clamping

  // Scalar defaults for backward compatibility with existing kernels
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();
  int swiglu_fusion = 0;  // 0 = default, 1 = interleaved layout

  ActivationParams() = default;

  explicit ActivationParams(ActivationType type)
      : activation_type(type) {
  }

  // Constructor for per-expert arrays (TRT-LLM style)
  ActivationParams(ActivationType type, float const* per_expert_alpha, float const* per_expert_beta, float const* per_expert_limit)
      : activation_type(type), swiglu_alpha(per_expert_alpha), swiglu_beta(per_expert_beta), swiglu_limit(per_expert_limit) {
  }

  // Implicit conversion to ActivationType for convenience
  operator ActivationType() const {
    return activation_type;
  }
};

// Legacy alias for backward compatibility during transition
using ActivationParameters = ActivationParams;

}  // namespace onnxruntime::llm::kernels::cutlass_kernels
