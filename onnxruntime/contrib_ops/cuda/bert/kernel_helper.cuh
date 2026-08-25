// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace kernel_helper {

constexpr int kThreads = 256;
constexpr int64_t kMaxGridDimX = 65535;

// Number of blocks for a grid-stride loop over `count` elements, clamped to the maximum grid size.
inline int GridSize(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  return static_cast<int>(std::min(blocks, kMaxGridDimX));
}

// Numerically stable logistic function.
__device__ __forceinline__ float SigmoidFloat(float x) {
  return x > 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}

__device__ __forceinline__ float SiluFloat(float x) {
  return x * SigmoidFloat(x);
}

// Euclidean modulo: the result always has the sign of `mod`, which must be positive.
template <typename T>
__device__ __forceinline__ T PositiveMod(T value, T mod) {
  const T result = value % mod;
  return result < 0 ? static_cast<T>(result + mod) : result;
}

// Multiplies through the unsigned counterpart of T so that overflow wraps around instead of
// being undefined behavior.
template <typename T>
__device__ __forceinline__ T WrappedMultiply(T a, T b);

template <>
__device__ __forceinline__ int32_t WrappedMultiply<int32_t>(int32_t a, int32_t b) {
  return static_cast<int32_t>(static_cast<uint32_t>(a) * static_cast<uint32_t>(b));
}

template <>
__device__ __forceinline__ int64_t WrappedMultiply<int64_t>(int64_t a, int64_t b) {
  return static_cast<int64_t>(static_cast<uint64_t>(a) * static_cast<uint64_t>(b));
}

}  // namespace kernel_helper
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
