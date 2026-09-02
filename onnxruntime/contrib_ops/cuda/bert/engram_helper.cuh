// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace engram_helper {

constexpr int kThreads = 256;
// grid.x is limited to 2^31 - 1 since compute capability 3.0 (the 65535 limit applies to grid.y and
// grid.z only). All kernels launched through GridSize() use a grid-stride loop, so the clamp only
// bounds the launch; correctness does not depend on it.
constexpr int64_t kMaxGridDimX = 2147483647;

// Number of blocks for a grid-stride loop over `count` elements, clamped to the maximum grid size.
inline int GridSize(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  return static_cast<int>(std::min(blocks, kMaxGridDimX));
}

// Sums three independent per-thread partials across the block in a single tree reduction, so a row
// that needs three reductions pays one set of barriers instead of three. `shared` must point to at
// least 3 * blockDim.x floats, and blockDim.x must be a power of two. All threads must call this.
__device__ __forceinline__ void BlockSum3(float* a, float* b, float* c, float* shared) {
  float* shared_a = shared;
  float* shared_b = shared + blockDim.x;
  float* shared_c = shared + 2 * blockDim.x;
  shared_a[threadIdx.x] = *a;
  shared_b[threadIdx.x] = *b;
  shared_c[threadIdx.x] = *c;
  __syncthreads();
  for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_a[threadIdx.x] += shared_a[threadIdx.x + stride];
      shared_b[threadIdx.x] += shared_b[threadIdx.x + stride];
      shared_c[threadIdx.x] += shared_c[threadIdx.x + stride];
    }
    __syncthreads();
  }
  *a = shared_a[0];
  *b = shared_b[0];
  *c = shared_c[0];
  __syncthreads();
}

// Numerically stable logistic function.
__device__ __forceinline__ float SigmoidFloat(float x) {
  return x > 0.0f ? 1.0f / (1.0f + expf(-x)) : expf(x) / (1.0f + expf(x));
}

// Engram gate pre-activation: sign(dot) * sqrt(max(abs(dot), 1e-6)).
// copysignf cannot be used here because it maps a zero dot product to +sqrt(1e-6) instead of zero,
// which would disagree with the schema formula and with the other execution providers.
__device__ __forceinline__ float EngramGateArg(float dot) {
  if (dot == 0.0f) {
    return 0.0f;
  }
  const float magnitude = sqrtf(fmaxf(fabsf(dot), 1.0e-6f));
  return dot < 0.0f ? -magnitude : magnitude;
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

}  // namespace engram_helper
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
