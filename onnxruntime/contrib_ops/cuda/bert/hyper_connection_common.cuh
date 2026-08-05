// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__device__ __forceinline__ float HyperToFloat(T value) {
  return static_cast<float>(value);
}

template <>
__device__ __forceinline__ float HyperToFloat<half>(half value) {
  return __half2float(value);
}

template <typename T>
__device__ __forceinline__ T HyperFromFloat(float value) {
  return static_cast<T>(value);
}

template <>
__device__ __forceinline__ half HyperFromFloat<half>(float value) {
  return __float2half(value);
}

template <typename T>
__device__ float ProjectHyperValue(const T* hidden, const float* weight,
                                   int row, int input_size, int output, float inverse_rms) {
  float value = 0.0f;
  const T* input_row = hidden + static_cast<int64_t>(row) * input_size;
  const float* weight_row = weight + static_cast<int64_t>(output) * input_size;
  for (int input = 0; input < input_size; ++input) {
    value += HyperToFloat(input_row[input]) * inverse_rms * weight_row[input];
  }
  return value;
}

inline int HyperBlockSize(int hidden_size, int max_threads) {
  return std::min(std::max(32, ((hidden_size + 31) / 32) * 32), max_threads);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime