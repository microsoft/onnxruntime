// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cub/cub.cuh>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cmath>
#include <math_constants.h>
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "dequantize_blockwise.cuh"

using namespace onnxruntime::cuda;
using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {


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
  static constexpr int kPackSize = (qbits == 8) ? 1 : (qbits == 4) ? 2 : (qbits == 2) ? 4 : 0;
  static_assert(kPackSize != 0, "Packing to whole bytes not supported for this qbits!");

  using QuantBlk = std::conditional_t<Columnwise, Shape2D<block_size, 1>, Shape2D<1, block_size>>;
  using ThreadBlk = Shape2D<QuantBlk::kRow * kPackSize, QuantBlk::kColumn>;
};

template <
  typename ElementT,
  int32_t block_size,
  int32_t qbits,
  bool Columnwise>
__global__
void dequantizeThread(ElementT* dst, const uint8_t* weights, const ElementT* scales,
  const uint8_t* zero_points, int rows,  int columns,  int thrd_col_blks) {

  using QuantBlk = typename BlkQuantTraits<ElementT, block_size, qbits, Columnwise>::QuantBlk;
  using ThreadBlk = typename BlkQuantTraits<ElementT, block_size, qbits, Columnwise>::ThreadBlk;

  const auto block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto row_blks = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;

  const auto meta_rows = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;
  const auto meta_cols = (columns + QuantBlk::kColumn - 1) / QuantBlk::kColumn;

  // quantized matrix is stored in column major, packed by column
  const auto q_rows = (meta_rows * QuantBlk::kRow * qbits + 7) / 8;
  const auto q_cols = meta_cols * QuantBlk::kColumn;

  int32_t r_blk_idx = static_cast<int32_t>(block_idx / thrd_col_blks);
  int32_t c_blk_idx = static_cast<int32_t>(block_idx % thrd_col_blks);

  int32_t r = r_blk_idx * ThreadBlk::kRow;
  int32_t c = c_blk_idx * ThreadBlk::kColumn;

  int32_t r_end = std::min(r + ThreadBlk::kRow, rows);
  int32_t c_end = std::min(c + ThreadBlk::kColumn, columns);

  for (int32_t j = c; j < c_end; ++j) {
    const int32_t meta_col = j / QuantBlk::kColumn;

    // !! 4b specific code
    // the whole loop is 4b specific due to sub 8 bit packing
    // and unpacking. We can potentially make this qbits generic
    // by wraping the packing/unpacking code like cutlass::Array
    static_assert(qbits == 4, "Only 4b block quantization is supported!");

    for (int32_t i = r; i < r_end; i += 2) {
      const int32_t meta_row = i / QuantBlk::kRow;

      const float scale0 = static_cast<float>(scales[meta_col * row_blks + meta_row]);
      const int zp_pair = (zero_points == nullptr)
            ? 0x88
            : zero_points[meta_col * ((row_blks + 1) / 2) + meta_row / 2];
      const int zp0 = (meta_row & 1) ? (zp_pair >> 4) : (zp_pair & 0xf);

      const uint8_t vi0 = weights[j * q_rows + i / 2] & 0xf;
      const float v0 = (static_cast<float>(vi0) - zp0) * scale0;

      dst[j * rows + i] = static_cast<ElementT>(v0);
      if ((i + 1) < r_end) {
        float scale1 = scale0;
        int zp1 = zp0;
        if constexpr (QuantBlk::kRow == 1) {
          scale1 = static_cast<float>(scales[meta_col * row_blks + meta_row + 1]);
          zp1 = (zp_pair >> 4) & 0xf;
        }
        const uint8_t vi1 = weights[j * q_rows + i / 2] >> 4;
        const float v1 = (static_cast<float>(vi1) - zp1) * scale1;
        dst[j * rows + (i + 1)] = static_cast<ElementT>(v1);
      }
    }
  }
}

template <
  typename ElementT,
  int32_t block_size,
  int32_t qbits,
  bool Columnwise>
static void dequantize(ElementT* dst, const uint8_t* weights, const ElementT* scales,
                        const uint8_t* zero_points, int32_t rows, int32_t columns,
                        cudaStream_t stream) {
  using QuantBlk = typename BlkQuantTraits<ElementT, block_size, qbits, Columnwise>::QuantBlk;
  using ThreadBlk = typename BlkQuantTraits<ElementT, block_size, qbits, Columnwise>::ThreadBlk;

  // Thread partitioning
  const auto thrd_row_blks = (rows + ThreadBlk::kRow - 1) / ThreadBlk::kRow;
  const auto thrd_col_blks = (columns + ThreadBlk::kColumn - 1) / ThreadBlk::kColumn;
  const auto total_thrd_blks = thrd_row_blks * thrd_col_blks;

  const auto grids = (total_thrd_blks + GridDim::maxThreadsPerBlock - 1) / GridDim::maxThreadsPerBlock;
  dequantizeThread<ElementT, block_size, qbits, Columnwise><<<grids, GridDim::maxThreadsPerBlock, 0, stream>>>(
      dst,
      weights,
      scales,
      zero_points,
      rows,
      columns,
      thrd_col_blks);
}


template <typename T>
Status
DequantizeBlockwise4b(
    T* dst,
    const uint8_t* src,
    const T* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    cudaStream_t stream) {
  switch (block_size) {
    case 16:
      if (columnwise) {
        dequantize<T, 16, 4, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize<T, 16, 4, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    case 32:
      if (columnwise) {
        dequantize<T, 32, 4, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize<T, 32, 4, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    case 64:
      if (columnwise) {
        dequantize<T, 64, 4, true>(dst, src, scales, zero_points, rows, columns, stream);
      } else {
        dequantize<T, 64, 4, false>(dst, src, scales, zero_points, rows, columns, stream);
      }
      return Status::OK();
    case 128:
      if (columnwise) {
        dequantize<T, 128, 4, true>(dst, src, scales, zero_points, rows,
                                                        columns, stream);
      } else {
        dequantize<T, 128, 4, false>(dst, src, scales, zero_points,
                                                            rows, columns, stream);
      }
      return Status::OK();
    case 256:
      if (columnwise) {
        dequantize<T, 256, 4, true>(dst, src, scales, zero_points, rows,
                                                        columns, stream);
      } else {
        dequantize<T, 256, 4, false>(dst, src, scales, zero_points,
                                                            rows, columns, stream);
      }
      return Status::OK();
    default:
      // Only block size 16, 32, 64, 128, 256 are supported.
      return Status(::onnxruntime::common::ONNXRUNTIME, ::onnxruntime::common::FAIL,
                    "Unsupported block size for blockwise quantization.");
  }
}

template
Status DequantizeBlockwise4b<float>(
    float* dst,
    const uint8_t* src,
    const float* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    cudaStream_t stream);

template
Status DequantizeBlockwise4b<half>(
    half* dst,
    const uint8_t* src,
    const half* scales,
    const uint8_t* zero_points,
    int block_size,
    bool columnwise,
    int rows,
    int columns,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
