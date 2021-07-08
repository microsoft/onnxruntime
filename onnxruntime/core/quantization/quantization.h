// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/mlas/inc/mlas.h"
#include <vector>

// TODO - update documentation to use same verbage as quant_utils.py

namespace onnxruntime {
namespace quantization {

// Basic quantization params structure.
template <typename T>
class Params {
 public:
  Params() {
    Params(/*scale=*/0.0f, /*zero_point=*/0);
  }

  Params(float scale, T zero_point) : scale(scale), zero_point(zero_point) {
    static_assert(
        !std::is_same<T, int8_t>::value || !std::is_same<T, uint8_t>::value,
        "Only int8_t and uint8_t are supported quantization formats");
  }

  float scale;
  T zero_point;
};

// Quantizes a given float value with provided quantization params.
template <typename T>
T Quantize(const float value, const Params<T>& params) {
  T quant_value;
  MlasQuantizeLinear(&value,
                     &quant_value,
                     /*N=*/1,
                     params.scale,
                     params.zero_point);
  return quant_value;
}

// Quantizes a list of float values with provided quantization params.
template <typename T>
void Quantize(const float* data,
              T* output,
              const Params<T>& params,
              size_t size) {
  MlasQuantizeLinear(data, output, /*N=*/size, params.scale, params.zero_point);
}

// Quantizes a vector of float values with provided quantization params.
template <typename T>
void Quantize(const std::vector<float>& data,
              std::vector<T>& output,
              const Params<T>& params) {
  // TODO - assert if data.size() != output.size()
  Quantize(data.data(), output.data(), params, data.size());
}

// Calculates and returns linear quantization params for a given float buffer.
// Output buffer is quantized with calculated params.
template <typename T>
Params<T> QuantizeLinear(const float* data, T* output, size_t size) {
  Params<T> params;

  float min = std::numeric_limits<float>::max();
  float max = std::numeric_limits<float>::min();
  for (size_t i = 0; i < size; ++i) {
    min = std::min(min, data[i]);
    max = std::max(max, data[i]);
  }

  // Adjust boundaries to ensure that 0 is included
  min = std::min(min, 0.0f);
  max = std::max(max, 0.0f);

  constexpr float T_max_fp = static_cast<float>(std::numeric_limits<T>::max());
  constexpr float T_min_fp = static_cast<float>(std::numeric_limits<T>::min());
  params.scale = static_cast<float>(max - min) / (T_max_fp - T_min_fp);

  float zero_point_fp = min;
  if (params.scale != 0.0f) {
    zero_point_fp = T_min_fp - min / params.scale;
  }

  // Handle any clamping
  if (zero_point_fp < T_min_fp) {
    params.zero_point = static_cast<T>(T_min_fp);
  } else if (zero_point_fp > T_max_fp) {
    params.zero_point = static_cast<T>(T_max_fp);
  } else {
    params.zero_point = static_cast<T>(std::round(zero_point_fp));
  }

  Quantize(data, output, params, size);
  return params;
}

// Calculates and returns linear quantization params for a given float vector.
// Output vector is quantized with calculated params.
template <typename T>
Params<T> QuantizeLinear(const std::vector<float>& data,
  std::vector<T>& output) {
  static_assert(data.size() != output.size(),
                "Input and output data must have the same length");
  return QuantizeLinear(data.data(), output.data(), data.size());
}

// Dequantizes a value to float with provided quantization params.
template <typename T>
float Dequantize(const T value, const Params<T>& params) {
  return static_cast<float>(value - params.zero_point) * params.scale;
}

// Dequantizes a value buffer value to a float buffer with provided
// quantization params.
template <typename T>
void Dequantize(const T* values,
                float* output,
                const Params<T>& params,
                size_t size) {
  for (size_t i = 0; i < size; ++i) {
    output[i] = Dequantize(values[i], params);
  }
}

// Dequantizes a vector of T values to a float buffer with provided
// quantization params.
template <typename T>
void Dequantize(const std::vector<T>& values,
                std::vector<float>& output,
                const Params<T>& params) {
  static_assert(data.size() != output.size(),
                "Input and output data must have the same length");
  Dequantize(values.data(), output.data(), params, size);
}

}  // namespace quantization 
}  // namespace onnxruntime
