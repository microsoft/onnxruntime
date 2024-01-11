// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
template <class T>
Status Dequantize4Bits(
    T* output,
    const uint8_t* quant_data,
    const T* scales_data,
    const uint8_t* zero_points,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);


/**
 * @brief Dequantize a block-wise quantized matrix, and store the result in a
 *        column major matrix for use in subsequent GEMM. This implementation supports
 *        columnwise and rowwise block orientation.
 * @param[out] dst           pointer to the dequantized matrix, column major: [columns, rows]
 * @param[in]  qelements     pointer to the quantized elements, column major: [columns, rows]
 * @param[in]  scales        pointer to the scales of quantized blocks, column major layout
 * @param[in]  zero_points   pointer to the zero points of quantized blocks, packed column major
 *                           scales
 * @param[in]  block_size    size of the quantized block
 * @param[in]  columnwise    whether the quantized matrix is columnwise or rowwise quantized
 * @param[in]  rows
 * @param[in]  columns
 */
template <typename T>
Status DequantizeBlockwise4b(
    T* dst,
    const uint8_t* qelements,
    const T* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    cudaStream_t stream);

namespace GPTQPacking {
/**
 * @brief Dequantize a block-wise quantized matrix, and store the result in a
 *        row major matrix for use in subsequent GEMM. This implementation supports
 *        columnwise block orientation.
 * @param[out] weight_out           pointer to the dequantized matrix, row major: [rows, columns]
 * @param[in]  qweight_i32     pointer to the quantized elements, column major: [rows/8, columns]
 * @param[in]  scales_data        pointer to the scales of quantized blocks, [rows/groupsize,columns]
 * @param[in]  zeros_data   pointer to the zero points of quantized blocks,[rows/groupsize,columns/8]
 *                           scales
 * @param[in]  groupsize    size of the quantized block ,16/32/64/128
 * @param[in]  bits   2,3,4,5,6,7
 * @param[in]  matrix_k
 * @param[in]  matrix_n
 */
template <typename ZEROT>
void DequantWeightNbit(
    cudaStream_t stream,
    const int32_t* qweight_i32,
    const void* scales_data,
    const ZEROT* zeros_data,
    void* weight_out,
    uint32_t matrix_k,
    uint32_t matrix_n,
    uint32_t bits,
    uint32_t groupsize);

// with group_idx support
void DequantWeightNbitGidx(cudaStream_t stream,
                           const int32_t* qweight_i32_i,
                           const void* scale_fp16,
                           const int32_t* qzeros_i32_i,
                           const int32_t* g_dix,
                           void* b_fp16,
                           uint32_t mat_k,
                           uint32_t mat_n,
                           int bits,
                           int groupsize);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
