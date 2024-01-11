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
    int shared_mem_per_block,
    cudaStream_t stream);
namespace GPTQPacking {
void TryMatMul4BitsGidx(
    cudaStream_t stream,
    const void* input,
    const int32_t* qweight,
    void* output,
    const void* scales,
    const int32_t* qzeros,
    const int32_t* g_idx,
    const int64_t* shapes);

template <typename ZEROT>
void TryMatMul4Bits(
    cudaStream_t stream,
    const void* input_data,
    const int32_t* qweight_data,
    void* mul_out_data,
    const void* scales_data,
    const ZEROT* zeros_data,
    uint32_t matrix_M,
    uint32_t matrix_K,
    uint32_t matrix_N,
    uint32_t groupsize);
}
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
