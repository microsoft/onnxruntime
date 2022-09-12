// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <vector>

#include "core/common/common.h"
#include "core/framework/tensor.h"
#include "core/mlas/inc/mlas.h"

// This header contains utility functions for quantizing and dequantizing
// values as outlined in the logic in
// onnxruntime/python/tools/quantization/quant_utils.py.
// These functions should be used for all quantization work inside ORT kernels
// and unit tests.

namespace onnxruntime {
namespace quantization {

#define ORT_STATIC_ASSERT_QUANTIZATION_TYPES(T)                            \
  static_assert(                                                           \
      !std::is_same<T, int8_t>::value || !std::is_same<T, uint8_t>::value, \
      "Only int8_t and uint8_t are supported quantization formats.");

// Basic quantization params structure.
template <typename T>
struct Params {
 public:
  Params() : Params(/*scale=*/0.0f, /*zero_point=*/0) {}

  Params(float scale, T zero_point) : scale(scale), zero_point(zero_point) {
    ORT_STATIC_ASSERT_QUANTIZATION_TYPES(T)
  }

  float scale;
  T zero_point;
};

// Returns quantization params from scale and zero point Tensor pointers.
// Caller is responsible for assuming that both Tensor pointers are of valid
// shape and type.
template <typename T>
Params<T> GetTensorQuantizationParams(const Tensor* scale_tensor,
                                      const Tensor* zero_point_tensor) {
  ORT_STATIC_ASSERT_QUANTIZATION_TYPES(T)
  return Params<T>(
      *(scale_tensor->Data<float>()),
      *(zero_point_tensor->Data<T>()));
}

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
  ORT_ENFORCE(data.size() == output.size(),
              "Input and output data must have the same length.");
  Quantize(data.data(), output.data(), params, data.size());
}

// Calculates and returns linear quantization params for a given float buffer.
// Output buffer is quantized with calculated params. The option to force
// symmetric quantization for signed values (zero point is forced at 0) is
// provided through the force_symmetric bool in-param.
template <typename T>
Params<T> QuantizeLinear(const float* data,
                         T* output,
                         size_t size,
                         bool force_symmetric = false) {
  Params<T> params;

  T T_min = std::numeric_limits<T>::min();
  T T_max = std::numeric_limits<T>::max();
  // NOTE: ORT currently clamps signed quantization values to -127,127 instead
  //       of -128/127. This is done to ensure that forced symmetric
  //       quantization results in a zero point of exactly 0 for signed 8 bit
  //       ints.
  // TODO(kreeger): Consider adjusting this clamping to enable more precision
  //                for signed 8 bit ints.
  //                See quant_utils.py - get_qmin_qmax_for_qType() for impl.
  if constexpr (std::is_same<T, int8_t>::value) {
    T_min = -127;
  }

  auto min_max_pair = std::minmax_element(data, data + size);
  float min = *min_max_pair.first;
  float max = *min_max_pair.second;

  // Adjust boundaries to ensure that 0 is included
  min = std::min(min, 0.0f);
  max = std::max(max, 0.0f);

  const float T_max_fp = static_cast<float>(T_max);
  const float T_min_fp = static_cast<float>(T_min);

  if (force_symmetric) {
    // Adjust min and max to ensure that zero_point will be at 0.
    // See quant_utils.py - compute_scale_zp() for more details.
    float abs_max = std::max(std::abs(min), std::abs(max));
    min = -abs_max;
    max = +abs_max;
  }

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
// Output buffer is quantized with calculated params. The option to force
// symmetric quantization for signed values (zero point is forced at 0) is
// provided through the force_symmetric bool in-param.
template <typename T>
Params<T> QuantizeLinear(const std::vector<float>& data,
                         std::vector<T>& output,
                         bool force_symmetric = false) {
  ORT_ENFORCE(data.size() == output.size(),
              "Input and output data must have the same length.");
  return QuantizeLinear(data.data(), output.data(), data.size(), force_symmetric);
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
  ORT_ENFORCE(values.size() == output.size(),
              "Input and output data must have the same length.");
  Dequantize(values.data(), output.data(), params, values.size());
}

// Transpose the input and store it to a new allocated buffer.
inline uint8_t* TransPoseInputData(const uint8_t* input,
                                   BufferUniquePtr& buffer_holder,
                                   AllocatorPtr& allocator,
                                   size_t M,
                                   size_t N) {
  uint8_t* output = static_cast<uint8_t*>(allocator->Alloc(M * N * sizeof(uint8_t)));
  MlasTranspose(input, output, M, N);
  buffer_holder.reset(output);
  return output;
}

}  // namespace quantization
}  // namespace onnxruntime
