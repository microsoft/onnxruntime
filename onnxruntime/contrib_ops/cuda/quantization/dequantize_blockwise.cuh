// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

///////////////////////////////////////////////////////////////////////////////
// A more general block-wise dequantization implementation that supports
// different block sizes and block orientations (row-wise/column-wise).
template <
    int Row_,    ///< rows of a matrix
    int Column_  ///< columns of a matrix
    >
struct Shape2D {
  static int const kRow = Row_;              ///< rows of a matrix
  static int const kColumn = Column_;        ///< columns of a matrix
  static int const kCount = Row_ * Column_;  ///< total number of elements in a matrix
};

/**
 * @brief Blockwise quantization constants
 * @tparam ElementT       source data type, e.g. fp32/fp16
 * @tparam block_size     number of elemenets quantized together
 * @tparam qbits          number of bits in each quantized element
 * @tparam Columnwise     true:  elements in a block come from one single column
 *                        false: elements in a block come from one single row
 */
template <
    typename ElementT,
    int32_t block_size,
    int32_t qbits,
    bool Columnwise>
struct BlkQuantTraits {
  // number of qbit elements to pack into whole bytes
  static constexpr int kPackSize = (qbits == 8) ? 1 : (qbits == 4) ? 2
                                                  : (qbits == 2)   ? 4
                                                                   : 0;
  static_assert(kPackSize != 0, "Packing to whole bytes not supported for this qbits!");

  using QuantBlk = std::conditional_t<Columnwise, Shape2D<block_size, 1>, Shape2D<1, block_size>>;

  using ThreadBlk = Shape2D<QuantBlk::kRow * kPackSize, QuantBlk::kColumn>;
};

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

template <class T, typename ZeroT>
Status Dequantize8Bits(
    T* output,
    const uint8_t* quant_data,
    const T* scales_data,
    const ZeroT* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template <class T, typename ZeroT>
Status DequantizeNBits(
    int bits,
    T* output,
    const uint8_t* quant_data,
    const T* scales_data,
    const ZeroT* zero_points,
    const int32_t* reorder_idx,
    int k,
    int n,
    int block_size,
    cudaStream_t stream) {
  if (bits == 4) {
    return Dequantize4Bits<T, ZeroT>(output, quant_data, scales_data, zero_points, reorder_idx, k, n, block_size, stream);
  } else {
    ORT_ENFORCE(bits == 8);
    return Dequantize8Bits<T, ZeroT>(output, quant_data, scales_data, zero_points, reorder_idx, k, n, block_size, stream);
  }
}

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

template <typename T>
Status DequantizeBlockwise8b(
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

// Macro to reduce repetitive fallback code
#define DECLARE_DEQUANTIZE_FALLBACK(FuncName, T, ZeroT, MIN_ARCH, TYPENAME)                                      \
  template <>                                                                                                    \
  Status FuncName<T, ZeroT>(                                                                                     \
      T*, const uint8_t*, const T*, const ZeroT*, const int32_t*, int, int, int, cudaStream_t) {                 \
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, TYPENAME " requires compute capability >= " #MIN_ARCH); \
  }
