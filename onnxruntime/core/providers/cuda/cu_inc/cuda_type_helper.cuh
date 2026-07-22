// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

// Convert half/bfloat16 to float
template <typename T>
__device__ __forceinline__ float to_float(T val) = delete;

template <>
__device__ __forceinline__ float to_float(float val) { return val; }

template <>
__device__ __forceinline__ float to_float(half val) { return __half2float(val); }

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ __forceinline__ float to_float(__nv_bfloat16 val) { return __bfloat162float(val); }
#endif

// Convert float to half/bfloat16/float
template <typename T>
__device__ __forceinline__ T from_float(float val) = delete;

template <>
__device__ __forceinline__ float from_float(float val) { return val; }

template <>
__device__ __forceinline__ half from_float(float val) { return __float2half(val); }

#if __CUDA_ARCH__ >= 800 || !defined(__CUDA_ARCH__)
template <>
__device__ __forceinline__ __nv_bfloat16 from_float(float val) { return __float2bfloat16(val); }
#endif

}  // namespace

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
