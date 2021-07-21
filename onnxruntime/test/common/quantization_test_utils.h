// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <cfenv>
#include <cmath>
#include <vector>

#include "core/quantization/quantization.h"

// Contains utility functions for quantizing test vectors inline.
namespace onnxruntime {
namespace test {

template <typename T>
inline T QuantizeTestValue(const float& value,
                           const quantization::Params<T>& params) {
  return quantization::Quantize(value, params);
}

template <typename T>
inline std::vector<T> QuantizeTestVector(const std::vector<float>& data,
                                         const quantization::Params<T>& params) {
  std::vector<T> result;
  result.resize(data.size());

  quantization::Quantize(data, result, params);
  return result;
}

template <typename T>
inline std::vector<T> QuantizeLinearTestVector(
    const std::vector<float>& data,
    quantization::Params<T>& out_params,
    bool force_symmetric = false) {
  std::vector<T> result;
  result.resize(data.size());

  out_params = quantization::QuantizeLinear(data, result, force_symmetric);
  return result;
}

}  // namespace test
}  // namespace onnxruntime

