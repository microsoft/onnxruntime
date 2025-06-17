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
    int m,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream) {
      if (bits == 8) {
        return TryMatMul8Bits<T>(output, a_data, b_data_quant, scales_data, zero_points,
                                 m, n, k, block_size, shared_mem_per_block, stream);
      }

      if (bits == 4) {
        return TryMatMul4Bits<T>(output, a_data, b_data_quant, scales_data, zero_points,
                                 m, n, k, block_size, shared_mem_per_block, stream);
      }

      return false;
    }

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
