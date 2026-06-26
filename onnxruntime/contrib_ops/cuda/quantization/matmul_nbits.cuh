// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T>
bool TryMatMul4Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    const T* bias_data,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream);

// accuracy_level=4 (int8 activation) path. LaunchQuantizeRowwiseInt8 produces a per-row int8 activation
// (aq) and per-row float scale (ascale); TryMatMulInt8Dp4a runs the int8 dp4a batched GEMV over the same
// 4-bit weight layout. Returns false if the shape/block_size is ineligible (caller falls back).
// Max M for the int8 dp4a verify path. Through M=8 the batched dp4a GEMV stays memory-bound and beats the
// A16 paths; beyond that the per-row weight re-read loses to dequant+cuBLAS, so M>8 falls through.
constexpr int kMatMulInt8Dp4aMaxM = 8;

template <class T>
void LaunchQuantizeRowwiseInt8(const T* a, int8_t* aq, float* ascale, int m, int k, cudaStream_t stream);

template <class T>
bool TryMatMulInt8Dp4a(
    T* output,
    const int8_t* aq,
    const float* ascale,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream);

template <class T>
bool TryMatMul8Bits(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream);

template <class T>
bool TryMatMulNBits(
    int bits,
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    const T* bias_data,
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream) {
  if (bits == 8) {
    if (bias_data != nullptr) {
      return false;
    }
    return TryMatMul8Bits<T>(output, a_data, b_data_quant, scales_data, zero_points,
                             m, n, k, block_size, shared_mem_per_block, stream);
  }

  if (bits == 4) {
    return TryMatMul4Bits<T>(output, a_data, b_data_quant, scales_data, zero_points, bias_data,
                             m, n, k, block_size, shared_mem_per_block, stream);
  }

  return false;
}

// Adds a per-column bias of shape [n] to the output of shape [m, n] (row-major).
// Used as a fallback when the fused bias GEMV specialization does not apply.
template <class T>
void LaunchMatMulNBitsBiasAdd(
    T* output,
    const T* bias_data,
    int m,
    int n,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
