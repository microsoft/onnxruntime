// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

// TODO - update documentation to use same verbage as quant_utils.py
// TODO - force types to int8/uint8/int16/int32_t?

namespace onnxruntime {
namespace quantization {

//
// Basic quantization params structure.
//
template <typename T>
struct Params {
  float scale;
  T zero_point;
};

//
// Quantizes a given float value with provided quantization params.
//
template <typename T>
T Quantize(const float value, const Params<T>& params) {
  constexpr int32_t T_max = std::numeric_limits<T>::max();
  constexpr int32_t T_min = std::numeric_limits<T>::min();

  int32_t raw = static_cast<int32_t>(std::round(value / params.scale)) + params.zero_point;
  return static_cast<T>(std::min(std::max(raw, T_min), T_max));
}

//
// Quantizes linearly a list of float values with provided quantization params.
//
template <typename T>
void Quantize(const float* data, T* output, const Params<T>& params, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    output[i] = Quantize(data[i], params);
  }
}

//
// Quantizes linearly a vector of float values with provided quantization params.
//
template <typename T>
void Quantize(const std::vector<float>& data, std::vector<T>& output, const Params<T>& params, size_t size) {
  Quantize(data.data(), output.data(), params, size);
}

//
// Calculates and returns linear quantization params and value for a given float buffer.
//
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

//
// Calculates and returns linear quantization params and value for a given float vector.
//
template <typename T>
Params<T> QuantizeLinear(const std::vector<float>& data, std::vector<T>& output, size_t size) {
  QuantizeLinear(data, output, size);
}

//
// Dequantizes a value to float with provided quantization params.
//
template <typename T>
float Dequantize(const T value, const Params<T>& params) {
  return static_cast<float>(value - params.zero_point) * params.scale;
}

//
// Dequantizes a value buffer value to a float buffer with provided quantization params.
//
template <typename T>
void Dequantize(const T* values, float* output, const Params<T>& params, size_t size) {
  for (size_t i = 0; i < size; ++i) {
    output[i] = Dequantize(values[i], params);
  }
}

//
// Dequantizes a vector of T values to a float buffer with provided quantization params.
//
template <typename T>
void Dequantize(const std::vector<T>& values, std::vector<float>& output, const Params<T>& params, size_t size) {
  Dequantize(values.data(), output.data(), params, size);
}

}  // namespace quantization 
}  // namespace onnxruntime