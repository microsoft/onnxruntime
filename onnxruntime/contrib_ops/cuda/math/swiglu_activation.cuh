// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// The clamped SwiGLU of one element, shared by every kernel that fuses it:
//   G = min(gate, limit), L = clamp(up, -limit, limit)         (clamps off when limit <= 0)
//   result = G * sigmoid(alpha * G) * (L + beta)
// Evaluated in float, widened to double for the product when alpha * G is below -87 so that a negative tail is not
// lost to float exp underflow. With `round_activation`, G * sigmoid(alpha * G) is narrowed to CudaT before the
// multiply by (L + beta), exactly as a QuickGelu node followed by a Mul in CudaT would.
// Keep the arithmetic identical to the SwiGLU operator.
template <typename CudaT = float>
__device__ __forceinline__ float SwiGluActivation(float gate, float up, float limit, float alpha, float beta,
                                                  bool round_activation = false) {
  // `l + beta` with beta == +0.0 turns l == -0.0 into +0.0 and can flip the sign of a zero result, which a
  // QuickGelu followed by a Mul keeps. -0.0 is the additive identity for every l, so a zero beta adds that.
  const float bias = beta == 0.0f ? -0.0f : beta;
  float g = gate;
  float l = up;
  if (limit > 0.0f) {
    // Unlike fmin and fmax, these comparisons preserve NaNs.
    if (g > limit) g = limit;
    if (l < -limit) l = -limit;
    if (l > limit) l = limit;
  }
  const float scaled = alpha * g;
  if (scaled < -87.0f) {
    // Float exp enters its subnormal range near -87 and underflows to zero by -104, which would zero a tail that
    // the following multiply by (l + beta) can bring back into range. Keep the whole product wide.
    const double wide_g = static_cast<double>(g);
    const double e = exp(wide_g * static_cast<double>(alpha));
    const double activated = wide_g * e / (1.0 + e);
    if (round_activation) {
      // QuickGelu narrows double -> float -> CudaT.
      return static_cast<float>(static_cast<CudaT>(static_cast<float>(activated))) * (l + bias);
    }
    return static_cast<float>(activated * (static_cast<double>(l) + static_cast<double>(bias)));
  }
  // Keeps the exponent non-positive without cancelling small negative-input results.
  const float sigmoid = scaled > 0.0f ? 1.0f / (1.0f + expf(-scaled)) : expf(scaled) / (1.0f + expf(scaled));
  float activated = g * sigmoid;
  if (round_activation) activated = static_cast<float>(static_cast<CudaT>(activated));
  return activated * (l + bias);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
