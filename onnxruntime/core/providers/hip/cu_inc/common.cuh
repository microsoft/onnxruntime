// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include <vector>
#include <mutex>
#include <assert.h>

#include <hip/hip_runtime.h>

#include "core/providers/hip/hip_common.h"
#include "core/providers/hip/hip_call.h"

namespace onnxruntime {
namespace hip {

template <typename T>
__device__ __inline__ T _Ceil(T a);

template <>
__device__ __inline__ float _Ceil(float a) { return ceilf(a); }

template <>
__device__ __inline__ double _Ceil(double a) { return ceil(a); }

template <>
__device__ __inline__ half _Ceil(half a) { return half(ceilf((float)a)); }

template <typename T>
__device__ __inline__ T _Floor(T a);

template <>
__device__ __inline__ float _Floor(float a) { return floorf(a); }

template <>
__device__ __inline__ double _Floor(double a) { return floor(a); }

template <>
__device__ __inline__ half _Floor(half a) { return half(floorf((float)a)); }

template <typename T>
__device__ __inline__ T _Sqrt(T a);

template <>
__device__ __inline__ float _Sqrt(float a) { return sqrtf(a); }

template <>
__device__ __inline__ double _Sqrt(double a) { return sqrt(a); }

template <>
__device__ __inline__ half _Sqrt(half a) { return half(sqrtf((float)a)); }

template <typename T>
__device__ __inline__ T _Erf(T a);

template <>
__device__ __inline__ float _Erf(float a) { return erff(a); }

template <>
__device__ __inline__ double _Erf(double a) { return erf(a); }

template <>
__device__ __inline__ half _Erf(half a) { return half(erff((float)a)); }

template <typename T>
__device__ __inline__ T _Round(T a);

template <>
__device__ __inline__ float _Round(float a) { return rintf(a); }

template <>
__device__ __inline__ double _Round(double a) { return rint(a); }

template <>
__device__ __inline__ half _Round(half a) { 
#if __HIP_ARCH__ < 530
  return half(rintf((float)a));
#else
  return hrint(a);
#endif
}

template <typename T>
__device__ __inline__ T _Exp(T a);

template <>
__device__ __inline__ float _Exp(float a) { return expf(a); }

template <>
__device__ __inline__ double _Exp(double a) { return exp(a); }

template <>
__device__ __inline__ half _Exp(half a) { return half(expf((float)a)); }

template <typename T>
__device__ __inline__ T _Log(T a);

template <>
__device__ __inline__ float _Log(float a) { return logf(a); }

template <>
__device__ __inline__ double _Log(double a) { return log(a); }

template <>
__device__ __inline__ half _Log(half a) { return half(logf((float)a)); }

template <typename T>
__device__ __inline T _Tanh(T a);

template <>
__device__ __inline__ float _Tanh(float a) { return tanhf(a); }

template <>
__device__ __inline__ double _Tanh(double a) { return tanh(a); }

template <>
__device__ __inline__ half _Tanh(half a) { return half(tanhf((float)a)); }

// template <>
// __device__ __inline__ half2 _Tanh(half2 a) {
//   float2 tmp = (__half22float2(a));
//   tmp.x = tanhf(tmp.x);
//   tmp.y = tanhf(tmp.y);
//   return __float22half2_rn(tmp);
// }

template <typename T>
__device__ __inline__ T _Pow(T a, T b);

template <>
__device__ __inline__ float _Pow(float a, float b) { return powf(a, b); }

template <>
__device__ __inline__ double _Pow(double a, double b) { return pow(a, b); }

template <>
__device__ __inline__ half _Pow(half a, half b) { return half(powf((float)a, (float)b)); }

template <typename T>
__device__ __inline__ T _Min(T a, T b) { return a < b ? a : b; }

template <typename T>
__device__ __inline__ T _Max(T a, T b) { return a > b ? a : b; }

template <typename T>
__device__ __inline__ T _Abs(T a) { return a > (T)0 ? a : -a; }

template <typename T>
__device__ __inline__ T _Normcdf(T a);

template <>
__device__ __inline__ float _Normcdf(float a) { return normcdff(a); }

template <>
__device__ __inline__ double _Normcdf(double a) { return normcdf(a); }

template <>
__device__ __inline__ half _Normcdf(half a) { return half(normcdff((float)a)); }

template <typename T>
__device__ __inline__ T _Gelu(T a) {
  return a * _Normcdf(a);
}

// We would like to use 64-bit integer to support large matrices. However, HIP seems to support only 32-bit integer
// For now, use int32_t to ensure that both Linux and Windows see this as 32 bit integer type.

#ifndef HIP_LONG
#define HIP_LONG int32_t
#endif

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))  // 0 based indexing

// ---------------------------------------------------------------------------
// GridDim -- helper to choose the HIP grid dimensions
// ---------------------------------------------------------------------------

template <class INT, class INT2>
static INT CeilDiv(INT a, INT2 b)  // ceil(a/b)
{
  return (INT)(((size_t)a + (size_t)b - 1) / (size_t)b);  // these size_t casts are necessary since b may be INT_MAX (for maxGridSize[])
}

struct GridDim {
  enum : HIP_LONG {
    maxThreadsPerBlock = 256,  // max threads per block
    maxWarpsPerBlock = 32,     // max warps per block
    maxElementsPerThread = 4,  // max element processed per thread
  };

  // use these for launching
  //   GridDim grid(NN);
  //   hipLaunchKernelGGL(kernel, dim3(grid.m_blocksPerGrid), dim3(grid.m_threadsPerBlock), ..., 0, ...)
  int blocks_per_grid_, threads_per_block_;  // (these may in the future be extended to multi-dimensional ones)
  HIP_LONG N_;

  GridDim(HIP_LONG N)  // linear grid
  {
    N_ = N;
    if (N == 0)  // HIP will fail to launch with 0 blocks
      N = 1;

    // get device information
    const auto& props = DeviceProp::GetDeviceProps();
    HIP_LONG numProcs = props.multiProcessorCount;
    HIP_LONG warpSize = props.warpSize;

    // distribute warps evenly over processors
    HIP_LONG warpsPerProc = CeilDiv(N, numProcs * warpSize);

    // if too many warps per block then reduce #warps
    // This limits the number of threads to 512.
    if (warpsPerProc > maxWarpsPerBlock) {
      HIP_LONG overBy = CeilDiv(warpsPerProc, maxWarpsPerBlock);  // we are over by this factor
      warpsPerProc = CeilDiv(warpsPerProc, overBy);
    }

    // put it back together
    threads_per_block_ = warpsPerProc * warpSize;  // =a multiple of 32 that is as close to 1024 as makes sense given NN
    blocks_per_grid_ = CeilDiv(N, threads_per_block_);
    if (blocks_per_grid_ == 1)
      threads_per_block_ = N;  // don't launch more than necessary
    assert(blocks_per_grid_ * threads_per_block_ >= N);
  }

  // compute our location on the grid
  static __device__ HIP_LONG GetLinearThreadId() {
    return blockDim.x * blockIdx.x + threadIdx.x;
  }
};

#define CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N) \
  HIP_LONG id = GridDim::GetLinearThreadId();     \
  if (id >= N)                                     \
    return;

// HIP_KERNEL_ASSERT is a macro that wraps an assert() call inside hip kernels.
// This is not supported by Apple platforms so we special case it.
// See http://docs.nvidia.com/hip/hip-c-programming-guide/#assertion
#if defined(__APPLE__) || defined(__HIP_PLATFORM_HCC__)
#define HIP_KERNEL_ASSERT(...)
#else // __APPLE__
#define HIP_KERNEL_ASSERT(...) assert(__VA_ARGS__)
#endif // __APPLE__

}  // namespace hip
}  // namespace onnxruntime
