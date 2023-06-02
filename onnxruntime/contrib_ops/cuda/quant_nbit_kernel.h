// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cstdint>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

void quant4BGEMV_cuda(
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
    const void* vec_data,
    const int* mat_data,
    void* mul_out_data,
    const void* scales_data,
    const int* zeros_data,
    int batch,
    int height,
    int width,
    int zero_width,
    int groupsize,
    int vec_height);

void vecquant4matmul_g_cuda(
    cudaStream_t stream,
    const void* vec_data,
    const int* mat_data,
    void* mul_out_data,
    const void* scales_data,
    const int* zeros_data,
    const int* g_idx_data,
    int batch,
    int height,
    int width,
    int zero_width,
    int vec_height);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
