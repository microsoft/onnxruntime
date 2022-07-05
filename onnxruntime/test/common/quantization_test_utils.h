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

template <typename T>
std::vector<T> ToVector(const int* value, int size) {
  std::vector<T> data(size);
  for (int i = 0; i < size; i++) {
    data[i] = static_cast<T>(value[i]);
  }
  return data;
}

template <typename T>
T GetMiddle(const std::vector<T>& v) {
  const auto min_max_pair = std::minmax_element(v.begin(), v.end());
  return (*(min_max_pair.first) + *(min_max_pair.second)) / 2;
}

}  // namespace test
}  // namespace onnxruntime
