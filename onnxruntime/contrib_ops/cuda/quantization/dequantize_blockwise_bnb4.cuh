// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T>
Status SetBnbQuantMap(int quant_type, T* quant_map_buffer, cudaStream_t stream);

// templated scalar multiply function
template <class T>
__device__ inline T ScalarMul(T a, T b);

template <>
__device__ inline float ScalarMul(float a, float b) {
  return a * b;
}

template <>
__device__ inline half ScalarMul(half a, half b) {
  #if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 530
    return a * b;
  #else
    // half multiplication not supported
    return static_cast<half>(static_cast<float>(a) * static_cast<float>(b));
  #endif
}

template <>
__device__ inline BFloat16 ScalarMul(BFloat16 a, BFloat16 b) {
  return a * b;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
// will use the native bfloat16 multiply instruction on sm_80+
template <>
__device__ inline nv_bfloat16 ScalarMul(nv_bfloat16 a, nv_bfloat16 b) {
  return a * b;
}
#endif

template <class T>
Status DequantizeBnb4(
    const T* quant_map,
    T* output,
    const uint8_t* quant_data,
    const T* absmax,
    int block_size,
    int numel,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
