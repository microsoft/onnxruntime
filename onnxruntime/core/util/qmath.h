// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/mlas/inc/mlas.h"

#include <cmath>

namespace onnxruntime {

inline float RoundHalfToEven(float input) {
  if (!std::isfinite(input)) {
    return input;
  }
  // std::remainder returns x - n, where n is the integral value nearest to x. When |x - n| = 0.5, n is chosen to be even
  return input - std::remainderf(input, 1.f);
}

template <typename T>
struct is_quant_type : std::false_type {};

template <>
struct is_quant_type<int8_t> : std::true_type {};

template <>
struct is_quant_type<uint8_t> : std::true_type {};

// ReduceRange and Symmetric is for test only
template <typename QType,
          bool ReduceRange = false,
          bool Symmetric = false,
          typename std::enable_if<is_quant_type<QType>::value, int>::type = 0>
void GetQuantizationParameter(const float* data, int64_t num_of_elements, float& scale, QType& zp) {
  // find input range min and max
  float min, max;
  MlasFindMinMaxElement(data, &min, &max, num_of_elements);

  // ensure the input range includes zero
  min = std::min(min, 0.0f);
  max = std::max(max, 0.0f);

  // find scale and zero point
  QType qmin = std::numeric_limits<QType>::min();
  QType qmax = std::numeric_limits<QType>::max();
  if (std::is_same<QType, int8_t>::value) {
    if (ReduceRange) {
      qmin = static_cast<QType>(-64);
      qmax = static_cast<QType>(64);
    }

    if (Symmetric) {
      zp = 0;
      float max_value = std::max(max, -min);
      scale = max_value > 0 ? max_value / qmax : 1.f;
      return;
    }
  }
  scale = max == min ? 1.0f : (max - min) / float(qmax - qmin);

  float initial_zero_point = qmin - min / scale;
  zp = static_cast<QType>(RoundHalfToEven(std::max(float(qmin), std::min(float(qmax), initial_zero_point))));
}

}  // namespace onnxruntime
