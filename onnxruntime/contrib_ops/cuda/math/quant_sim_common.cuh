// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cmath>
#include <cuda_fp16.h>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

// A model trained against simulated low-precision grids carries the quantisation arithmetic
// explicitly in its graph, and a fused kernel has to reproduce it step for step --
// these are step functions, and one rounding difference becomes a whole-step move that the
// indexer's top-k and the MoE router turn into a different answer. This header is shared so
// the compressor and the indexer cannot drift apart.

constexpr float kQuantSimLog2 = 0.69314718f;  // the graph divides by float(math.log(2.0))
constexpr float kQuantSimFp8Max = 448.0f;
constexpr float kQuantSimFp4Max = 6.0f;

template <typename CudaT>
struct QuantSimConv;

template <>
struct QuantSimConv<float> {
  static __device__ __forceinline__ float ToFloat(float v) { return v; }
  static __device__ __forceinline__ float Round(float v) { return v; }
  static __device__ __forceinline__ float FromFloat(float v) { return v; }
};

template <>
struct QuantSimConv<half> {
  static __device__ __forceinline__ float ToFloat(half v) { return __half2float(v); }
  static __device__ __forceinline__ float Round(float v) { return __half2float(__float2half_rn(v)); }
  static __device__ __forceinline__ half FromFloat(float v) { return __float2half_rn(v); }
};

template <>
struct QuantSimConv<BFloat16> {
  static __device__ __forceinline__ float ToFloat(BFloat16 v) { return static_cast<float>(v); }
  static __device__ __forceinline__ float Round(float v) { return static_cast<float>(BFloat16(v)); }
  static __device__ __forceinline__ BFloat16 FromFloat(float v) { return BFloat16(v); }
};

// A power-of-two scale whose exponent is `ceil(log2(amax / limit))`, matching the graph's
// Log/Div/Ceil/Pow chain including its clamp on the way into the logarithm.
__device__ __forceinline__ float QuantSimBlockScale(float amax, float limit, float floor_value) {
  float r = amax / limit;
  if (r < floor_value) r = floor_value;
  return amax > 0.0f ? exp2f(ceilf(logf(r) / kQuantSimLog2)) : 1.0f;
}

// Round onto the FP8-E4M3FN grid. `v` is already clipped to the finite range, so this only has
// to snap to the 3-bit mantissa of the containing binade, with 2^-6 as the subnormal floor.
__device__ __forceinline__ float QuantSimRoundE4M3(float v) {
  const float a = fabsf(v);
  if (!(a > 0.0f)) return v;
  int e;
  frexpf(a, &e);
  e -= 1;
  if (e < -6) e = -6;
  const float step = ldexpf(1.0f, e - 3);
  return copysignf(rintf(a / step) * step, v);
}

// Hadamard rotation followed by the simulated FP4-E2M1 round trip, in place over one row of
// `d` channels held in shared memory. `s_scale` needs `d / 32` floats. Every thread of the
// block must call this: it synchronises internally.
template <typename CudaT>
__device__ __forceinline__ void QuantSimRotateFp4(float* s_row, float* s_scale, int d,
                                                  int tid, int nthreads) {
  // Walsh-Hadamard butterfly: the same orthogonal rotation the graph writes as a dense
  // MatMul against a Sylvester matrix, minus the matrix.
  for (int len = 1; len < d; len <<= 1) {
    for (int i = tid; i < d / 2; i += nthreads) {
      const int lo = (i / len) * (len << 1) + (i % len);
      const float a = s_row[lo];
      const float c = s_row[lo + len];
      s_row[lo] = a + c;
      s_row[lo + len] = a - c;
    }
    __syncthreads();
  }
  const float hscale = rsqrtf(static_cast<float>(d));
  for (int c = tid; c < d; c += nthreads) {
    s_row[c] = QuantSimConv<CudaT>::Round(s_row[c] * hscale);
  }
  __syncthreads();

  const int blocks = d / 32;
  for (int i = tid; i < blocks; i += nthreads) {
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(s_row[i * 32 + j]));
    s_scale[i] = QuantSimBlockScale(amax, kQuantSimFp4Max, 1e-38f);
  }
  __syncthreads();
  for (int c = tid; c < d; c += nthreads) {
    const float scale = s_scale[c >> 5];
    const float v = fminf(fmaxf(s_row[c] / scale, -kQuantSimFp4Max), kQuantSimFp4Max);
    const float u = fabsf(v);
    // E2M1 grid {0,.5,1,1.5,2,3,4,6}: the step doubles at 2 and 4, and ties go toward zero.
    const float step = u < 2.0f ? 0.5f : (u < 4.0f ? 1.0f : 2.0f);
    const float sign = v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
    s_row[c] = sign * step * ceilf(u / step - 0.5f) * scale;
  }
  __syncthreads();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
