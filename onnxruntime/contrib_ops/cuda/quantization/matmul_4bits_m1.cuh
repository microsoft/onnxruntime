// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <class T>
bool TryMatMul4BitsM1(
    T* output,
    const T* a_data,
    const uint8_t* b_data_quant,
    const T* scales_data,
    const uint8_t* zero_points,
    int n,
    int k,
    int block_size,
    size_t shared_mem_per_block,
    cudaStream_t stream);

// Opt-in (ORT_MATMUL_NBITS_4BIT_M1_WIDE=1) fp16 single-row kernel with 16-byte weight loads. Returns
// false without launching when the shape, block size or alignment is unsupported.
bool TryMatMul4BitsM1Wide(
    half* output,
    const half* a_data,
    const uint8_t* b_data_quant,
    const half* scales_data,
    const uint8_t* zero_points,
    int n,
    int k,
    int block_size,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
