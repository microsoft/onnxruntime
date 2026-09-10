// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

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
    int sm_count,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
