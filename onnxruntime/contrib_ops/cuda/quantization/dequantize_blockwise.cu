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


__device__ __forceinline__ void DequantizeEightElements(uint32_t values_quant, half scale, half zp, half* output) {
  half2 scale_half2 = {scale, scale};
  half zp_adjust = -scale * __short2half_rn(zp);
  half2 zp_adjust2 = {zp_adjust, zp_adjust};

  alignas(16) half2 results[4];
  half v0 = __uint2half_rn(values_quant & 0xF);
  half v1 = __uint2half_rn((values_quant >> 4) & 0xF);
  results[0] = __halves2half2(v0, v1) * scale_half2 + zp_adjust2;

  half v2 = __uint2half_rn((values_quant >> 8) & 0xF);
  half v3 = __uint2half_rn((values_quant >> 12) & 0xF);
  results[1] = __halves2half2(v2, v3) * scale_half2 + zp_adjust2;

  half v4 = __uint2half_rn((values_quant >> 16) & 0xF);
  half v5 = __uint2half_rn((values_quant >> 20) & 0xF);
  results[2] = __halves2half2(v4, v5) * scale_half2 + zp_adjust2;

  half v6 = __uint2half_rn((values_quant >> 24) & 0xF);
  half v7 = __uint2half_rn((values_quant >> 28) & 0xF);
  results[3] = __halves2half2(v6, v7) * scale_half2 + zp_adjust2;
  *(reinterpret_cast<float4*>(output)) = *(reinterpret_cast<float4*>(results));
}

__device__ __forceinline__ void DequantizeEightElements(uint32_t values_quant, float scale, float zp, float* output) {
  float zp_adjust = -scale * zp;
  output[0] = float(values_quant & 0xF) * scale + zp_adjust;
  output[1] = float((values_quant >> 4) & 0xF) * scale + zp_adjust;
  output[2] = float((values_quant >> 8) & 0xF) * scale + zp_adjust;
  output[3] = float((values_quant >> 12) & 0xF) * scale + zp_adjust;
  output[4] = float((values_quant >> 16) & 0xF) * scale + zp_adjust;
  output[5] = float((values_quant >> 20) & 0xF) * scale + zp_adjust;
  output[6] = float((values_quant >> 24) & 0xF) * scale + zp_adjust;
  output[7] = float((values_quant >> 28) & 0xF) * scale + zp_adjust;
}

template <class T>
__global__ void Dequantize4BitsKernel(
    T* output,
    const uint8_t* quant_data,
    const T* scale_data,
    const uint8_t* zero_points,
    int block_size,
    int blocks_per_K,
    int blocks_per_threadblock,
    int shift) {
  int block_id = blockIdx.x * blocks_per_threadblock + ((threadIdx.x * 8) >> shift);
  int n_idx = block_id / blocks_per_K;
  int kb_idx = block_id % blocks_per_K;
  int element_offset = block_id * block_size + ((threadIdx.x * 8) & ((1 << shift) - 1));
  uint32_t quant_value = *(reinterpret_cast<const uint32_t*>(quant_data + element_offset / 2));
  T scale = *(scale_data + block_id);
  uint8_t zp = 8;
  if (zero_points) {
    zp = zero_points[n_idx * ((blocks_per_K + 1)/2) + kb_idx / 2];
    zp = (kb_idx & 0x01) ? (zp >> 4) : (zp & 0x0f);
  }

  output = output + element_offset;
  DequantizeEightElements(quant_value, scale, static_cast<T>(zp), output);
}

template <class T>
Status Dequantize4Bits(
    T* output,
    const uint8_t* quant_data,
    const T* scales_data,
    const uint8_t* zero_points,  // shape: [N, (block_per_K + 1)/2]
    int k,
    int n,
    int block_size,
    cudaStream_t stream) {
  // k is padded and equal to block_per_K * block_size
  ORT_ENFORCE(k % block_size == 0, "k must be a multiplier of block_size");
  constexpr int element_per_thread = 8;
  int blocks_per_threadblock = GridDim::maxThreadsPerBlock * element_per_thread / block_size;
  int blocks_per_K = k / block_size;
  int blocks_per_grid = static_cast<int>(CeilDiv(n * blocks_per_K, blocks_per_threadblock));
  int shift = static_cast<int>(log2f(float(block_size)));

  Dequantize4BitsKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      output,
      quant_data,
      scales_data,
      zero_points,
      block_size,
      blocks_per_K,
      blocks_per_threadblock,
      shift);

  return Status::OK();
}

template Status Dequantize4Bits<float>(
    float* output,
    const uint8_t* quant_data,
    const float* scales_data,
    const uint8_t* zero_points,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);

template Status Dequantize4Bits<half>(
    half* output,
    const uint8_t* quant_data,
    const half* scales_data,
    const uint8_t* zero_points,
    int k,
    int n,
    int block_size,
    cudaStream_t stream);


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
  const uint8_t* zero_points, int rows,  int columns,  int thrd_row_blks) {

  using QuantBlk = typename BlkQuantTraits<ElementT, block_size, qbits, Columnwise>::QuantBlk;
  using ThreadBlk = typename BlkQuantTraits<ElementT, block_size, qbits, Columnwise>::ThreadBlk;

  // !! 4b specific code
  static_assert(qbits == 4, "Only 4b block quantization is supported!");

  const auto block_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto row_blks = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;

  const auto meta_rows = (rows + QuantBlk::kRow - 1) / QuantBlk::kRow;

  // quantized matrix is stored in column major, packed by column
  const auto q_rows = (meta_rows * QuantBlk::kRow * qbits + 7) / 8;

  int32_t r_blk_idx = static_cast<int32_t>(block_idx % thrd_row_blks);
  int32_t c_blk_idx = static_cast<int32_t>(block_idx / thrd_row_blks);

  int32_t r = r_blk_idx * ThreadBlk::kRow;
  int32_t c = c_blk_idx * ThreadBlk::kColumn;

  int32_t r_end = std::min(r + ThreadBlk::kRow, rows);
  int32_t c_end = std::min(c + ThreadBlk::kColumn, columns);

  // for 4b quant, kPackSize = 2, so we have 2 scales and 2 offsets
  const ElementT scale_buf[2] = {
      scales[(c / QuantBlk::kColumn) * row_blks + r / QuantBlk::kRow],
      ((r/QuantBlk::kRow) < (meta_rows - 1)) ? scales[(c / QuantBlk::kColumn) * row_blks + r / QuantBlk::kRow + 1] : static_cast<ElementT>(0.0f)};
  const uint8_t zp_pair = (zero_points == nullptr)
        ? 0x88
        : zero_points[(c / QuantBlk::kColumn) * ((row_blks + 1) / 2) + (r / QuantBlk::kRow) / 2];
  const uint16_t zp_buf[2] = {(uint16_t)(zp_pair & 0x0f), (uint16_t)((zp_pair >> 4) & 0x0f)};
  const ElementT adjust_buf[2] = {(-scale_buf[0]) * static_cast<ElementT>(zp_buf[0]),
                                  (-scale_buf[1]) * static_cast<ElementT>(zp_buf[1])};

  const int32_t meta_col = c / QuantBlk::kColumn;
  for (int32_t j = c; j < c_end; ++j) {
    const uint8_t* q_ptr = weights + j * q_rows;
    for (int32_t i = r; i < (r_end - 1); i += 2) {

      const auto scale0 = scale_buf[(i - r) / QuantBlk::kRow];
      const auto adjust0 = adjust_buf[(i - r) / QuantBlk::kRow];

      const auto scale1 = scale_buf[(i + 1 - r) / QuantBlk::kRow];;
      const auto adjust1 = adjust_buf[(i + 1 - r) / QuantBlk::kRow];

      const auto vi = q_ptr[i / 2];

      if constexpr (std::is_same<ElementT, half>::value){
        half2 scale_half2 = {scale0, scale1};
        half2 zp_adjust2 = {adjust0, adjust1};

        half2 v = {__ushort2half_rn(vi & 0xF), __ushort2half_rn((vi >> 4) & 0xF)};
        half2 results = v * scale_half2 + zp_adjust2;

        dst[j * rows + i] = results.x;
        dst[j * rows + (i + 1)] = results.y;
      } else {
        static_assert(std::is_same<ElementT, float>::value, "Only float and half are supported!");
        const uint8_t vi0 = vi & 0xf;
        const uint8_t vi1 = vi >> 4;
        dst[j * rows + i] = static_cast<float>(vi0) * scale0 + adjust0;;
        dst[j * rows + (i + 1)] = static_cast<float>(vi1) * scale1 + adjust1;
      }
    }

    if ((r_end & 1) && (r_end > r)){
      const auto scale0 = scale_buf[(r_end - 1 - r) / QuantBlk::kRow];
      const auto adjust0 = adjust_buf[(r_end - 1 - r) / QuantBlk::kRow];

      const auto vi = q_ptr[(r_end - 1) / 2];
      const uint8_t vi0 = vi & 0xf;

      dst[j * rows + (r_end - 1)] = static_cast<ElementT>(vi0) * scale0 + adjust0;
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
      thrd_row_blks);
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
