// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <type_traits>

namespace onnxruntime {
namespace contrib {
namespace kernel_helper {

// Numerically stable logistic function.
inline float SigmoidFloat(float x) {
  if (x > 0.0f) {
    return 1.0f / (1.0f + std::exp(-x));
  }
  const float exp_x = std::exp(x);
  return exp_x / (1.0f + exp_x);
}

inline float SiluFloat(float x) {
  return x * SigmoidFloat(x);
}

// Euclidean modulo: the result always has the sign of `mod`, which must be positive.
template <typename T>
inline T PositiveMod(T value, T mod) {
  T result = static_cast<T>(value % mod);
  if (result < 0) {
    result = static_cast<T>(result + mod);
  }
  return result;
}

// Multiplies through the unsigned counterpart of T so that overflow wraps around instead of
// being undefined behavior.
template <typename T>
inline T WrappedMultiply(T a, T b) {
  using UnsignedT = typename std::make_unsigned<T>::type;
  return static_cast<T>(static_cast<UnsignedT>(a) * static_cast<UnsignedT>(b));
}

}  // namespace kernel_helper
}  // namespace contrib
}  // namespace onnxruntime
