// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace onnxruntime {
namespace contrib {
namespace engram_helper {

// Numerically stable logistic function.
inline float SigmoidFloat(float x) {
  if (x > 0.0f) {
    return 1.0f / (1.0f + std::exp(-x));
  }
  const float exp_x = std::exp(x);
  return exp_x / (1.0f + exp_x);
}

// Engram gate pre-activation: sign(dot) * sqrt(max(abs(dot), 1e-6)).
// std::copysign cannot be used here because it maps a zero dot product to +sqrt(1e-6) instead of
// zero, which would disagree with the schema formula and with the other execution providers.
inline float EngramGateArg(float dot) {
  if (dot == 0.0f) {
    return 0.0f;
  }
  const float magnitude = std::sqrt(std::max(std::abs(dot), 1.0e-6f));
  return dot < 0.0f ? -magnitude : magnitude;
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

// Multiplies two non-negative, attribute-derived dimensions (e.g. (max_ngram_size - 1) *
// n_head_per_ngram) with an overflow check. Both factors are user-controlled model attributes with
// only lower-bound checks, so the product must be validated before it is used to size any tensor or
// index: an unchecked signed multiplication would otherwise be free to silently wrap (undefined
// behavior in the general case) before ever reaching a tensor-shape validation. Returns false, and
// leaves `result` unset, when the product would overflow int64_t.
inline bool TryMultiplyDims(int64_t a, int64_t b, int64_t& result) {
  if (a != 0 && b > std::numeric_limits<int64_t>::max() / a) {
    return false;
  }
  result = a * b;
  return true;
}

}  // namespace engram_helper
}  // namespace contrib
}  // namespace onnxruntime
