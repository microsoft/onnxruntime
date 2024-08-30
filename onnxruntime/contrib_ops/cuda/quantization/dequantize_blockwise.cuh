// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
template <class T, typename ZeroT>
Status Dequantize4Bits(
    T* output,
    const uint8_t* quant_data,
    const T* scales_data,
    const ZeroT* zero_points,
    const int32_t* reorder_idx,
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

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
