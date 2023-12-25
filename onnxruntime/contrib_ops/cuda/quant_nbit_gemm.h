// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cstdint>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "cuda_fp16.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

void Q4bitGemv(
    cudaStream_t stream,
    const void* vec_data,
    const int32_t* mat_data,
    void* mul_out_data,
    const void* scales_data,
    const int32_t* zeros_data,
    uint32_t MATRIX_M,
    uint32_t MATRIX_K,
    uint32_t MATRIX_N,
    uint32_t groupsize);
void vecquant4matmul_cuda(
    cudaStream_t stream,
    const void* vec,
    const int* mat,
    void* mul,
    const void* scales,
    const int* zeros,
    const int* g_idx,
    int64_t* shape);
void DequantWeightNbit_g(cudaStream_t stream,
                         const int32_t* qweight_i32_i, const void* scale_fp16,
                         const int32_t* qzeros_i32_i, const int32_t* g_dix,
                         void* b_fp16,
                         uint32_t mat_k, uint32_t mat_n, int bits,
                         int groupsize);
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
