// Modifications: scaling is moved from masked softmax to the gemm before that.
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
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
    int total_blks,
    int shift) {
  int block_id = blockIdx.x * blocks_per_threadblock + ((threadIdx.x * 8) >> shift);
  if (block_id >= total_blks) {
    return;
  }
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
  int total_blks = n * blocks_per_K;
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
      total_blks,
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
void dequantizeThread(ElementT* dst,
                      const uint8_t* weights,
                      const ElementT* scales,
                      const uint8_t* zero_points,
                      int rows,
                      int columns,
                      int thrd_row_blks) {
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
      ((r/QuantBlk::kRow) < (meta_rows - 1))
          ? scales[(c / QuantBlk::kColumn) * row_blks + r / QuantBlk::kRow + 1]
          : static_cast<ElementT>(0.0f)};
  const uint8_t zp_pair = (zero_points == nullptr)
        ? 0x88
        : zero_points[(c / QuantBlk::kColumn) * ((row_blks + 1) / 2) + (r / QuantBlk::kRow) / 2];
  const uint16_t zp_buf[2] = {(uint16_t)(zp_pair & 0x0f), (uint16_t)((zp_pair >> 4) & 0x0f)};
  const ElementT adjust_buf[2] = {(-scale_buf[0]) * static_cast<ElementT>(zp_buf[0]),
                                  (-scale_buf[1]) * static_cast<ElementT>(zp_buf[1])};

  for (int32_t j = c; j < c_end; ++j) {
    const uint8_t* q_ptr = weights + j * q_rows;
    for (int32_t i = r; i < (r_end - 1); i += 2) {
      const auto scale0 = scale_buf[(i - r) / QuantBlk::kRow];
      const auto adjust0 = adjust_buf[(i - r) / QuantBlk::kRow];

      const auto scale1 = scale_buf[(i + 1 - r) / QuantBlk::kRow];;
      const auto adjust1 = adjust_buf[(i + 1 - r) / QuantBlk::kRow];

      const auto vi = q_ptr[i / 2];

      if constexpr (std::is_same<ElementT, half>::value) {
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

    if ((r_end & 1) && (r_end > r)) {
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

namespace GPTQPacking {
constexpr int kBlockSize = 256;

#define FETCH_UINT2(pointer) (reinterpret_cast<const uint2*>(&(pointer))[0])
#define FETCH_HALF2(pointer) (reinterpret_cast<const half2*>(&(pointer))[0])

template <typename T, int WBITS, typename ZEROT>
__global__ void kDequantizeAndUnpackWeight248(T* out, const int32_t* qweight, const T* scale, const ZEROT* qzeros,
                                              const int group_size, const int in_features, const int n) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  // const int qweight_rows = (in_features * WBITS + 31) / 32;
  const int half_n = n / 2;

  const int compress_group_size = 32 / WBITS;
  const int max_num_in_bits = (1 << WBITS) - 1;

  uint2 weight_int2 = FETCH_UINT2(qweight[tid * 2]);
  uint32_t weight_v1 = weight_int2.x;
  uint32_t weight_v2 = weight_int2.y;

  int col_ind = (tid % half_n) * 2;
  int weight_in_row = tid / half_n * compress_group_size;
  half2 scale_v = FETCH_HALF2(scale[weight_in_row / group_size * n + col_ind]);
  half2 hzeros;
  if constexpr (std::is_same<ZEROT, uint32_t>::value) {
    uint32_t zero_v = qzeros == nullptr ? 0x88888888 : qzeros[weight_in_row / group_size * (n / compress_group_size) + (col_ind) / compress_group_size];
    int zero_ind = col_ind % compress_group_size;
    uint8_t zv1 = (zero_v >> (zero_ind * WBITS)) & max_num_in_bits;
    uint8_t zv2 = (zero_v >> (zero_ind * WBITS + WBITS)) & max_num_in_bits;
    hzeros = __halves2half2(__short2half_rn(zv1), __short2half_rn(zv2));
  } else {
    hzeros = FETCH_HALF2(qzeros[weight_in_row / group_size * n + col_ind]);
  }
  half2 scale_zeros = __hmul2(hzeros, scale_v);

  half2* out_h2 = reinterpret_cast<half2*>(out);
  // decompress weights
  int remains = in_features - weight_in_row;
  if (remains >= compress_group_size) {
#pragma unroll
    for (int i = 0; i < compress_group_size; i++) {
      uint8_t wv1 = (weight_v1 >> (i * WBITS)) & max_num_in_bits;
      uint8_t wv2 = (weight_v2 >> (i * WBITS)) & max_num_in_bits;
      half2 wv = __halves2half2(__short2half_rn(wv1), __short2half_rn(wv2));
      out_h2[((weight_in_row + i) * n + col_ind) / 2] = __hfma2(wv, scale_v, -scale_zeros);
    }
  } else {
    for (int i = 0; i < remains; i++) {
      uint8_t wv1 = (weight_v1 >> (i * WBITS)) & max_num_in_bits;
      uint8_t wv2 = (weight_v2 >> (i * WBITS)) & max_num_in_bits;
      half2 wv = __halves2half2(__short2half_rn(wv1), __short2half_rn(wv2));
      out_h2[((weight_in_row + i) * n + col_ind) / 2] = __hfma2(wv, scale_v, -scale_zeros);
    }
  }
}

template <typename T, int WBITS>
__device__ __forceinline__ uchar2 IteratorQweight(const T* ptr, int idx) {
  int start_bits = idx * WBITS;
  int first = start_bits / 32;
  int end_bits = (start_bits + WBITS);
  int second = end_bits / 32;
  start_bits = start_bits % 32;
  end_bits = end_bits % 32;
  uchar2 res;
  if (first == second) {
    res.x = (ptr[first].x >> (start_bits)) & ((1 << WBITS) - 1);
    res.y = (ptr[first].y >> (start_bits)) & ((1 << WBITS) - 1);
    return res;
  } else {
    res.x = (ptr[first].x >> (start_bits));
    res.y = (ptr[first].y >> (start_bits));

    res.x |= ((ptr[second].x) & ((1 << (end_bits)) - 1)) << (32 - start_bits);
    res.y |= ((ptr[second].y) & ((1 << (end_bits)) - 1)) << (32 - start_bits);
    return res;
  }
}

template <typename T, int WBITS, typename ZEROT>
__global__ void DequantizeAndUnpackWeight3567(T* out, const uint32_t* qweight, const T* scale, const ZEROT* qzeros,
                                              int group_size, const int in_features, const int row_n) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  const int qweight_rows = (in_features * WBITS + 31) / 32;
  __shared__ uint2 qweight_shared[WBITS * kBlockSize];
  const int half_n = row_n / 2;

  const int group_row_n = half_n * (WBITS == 6 ? 3 : WBITS);
  int total_qw = qweight_rows * half_n;

  uint2* qweight_thread = qweight_shared + WBITS * threadIdx.x;

  int qweight_start = tid / half_n * group_row_n + tid % half_n;
  const uint2* qweigh2 = (const uint2*)qweight;
#pragma unroll
  for (int j = 0; j < WBITS; j++) {
    int ind = qweight_start + half_n * j;
    qweight_thread[j] = ind < total_qw ? (qweigh2[ind]) : uint2();
  }

  const int col_ind = (tid % half_n);
  const int compress_group_size = 32;
  const int fp16_weight_in_row = tid / half_n * compress_group_size;
  half2 scale_v[4];
  const int scale_zero_from = fp16_weight_in_row / group_size;
  const int scale_zero_to = min(in_features / group_size - 1, (fp16_weight_in_row + compress_group_size) / group_size);

  // decompress scales
  const half2* scale2 = reinterpret_cast<const half2*>(scale);
  for (int i = 0, scale_zero_from_i = scale_zero_from; scale_zero_from_i <= scale_zero_to; scale_zero_from_i++, i++) {
    scale_v[i] = (scale2[scale_zero_from_i * half_n + col_ind]);
  }

  // decompress zeros
  half2 hzeros[4];
  if constexpr (std::is_same<ZEROT, uint32_t>::value) {
    const int max_num_in_bits = (1 << WBITS) - 1;
    uchar2 zv1[4];
    int half_col_ind = col_ind * 2;
    const int zero_col_from = half_col_ind * WBITS / 32;
    const int zero_col_to = ((half_col_ind + 1) * WBITS - 1) / 32;
    const int zero_col_to_2 = ((half_col_ind + 2) * WBITS - 1) / 32;
    const int qzero_width = (row_n * WBITS + 32 - 1) / 32;
    for (int i = 0, scale_zero_from_i = scale_zero_from; scale_zero_from_i <= scale_zero_to; scale_zero_from_i++, i++) {
      uint32_t zero_v = qzeros == nullptr ? 0x88888888 : qzeros[scale_zero_from_i * qzero_width + zero_col_from];
      const int zero_bits_last = (((half_col_ind)*WBITS) % 32);
      zv1[i].x = (zero_v >> zero_bits_last) & max_num_in_bits;
      if (zero_col_from != zero_col_to) {
        const int zero_bits_first = ((half_col_ind + 1) * WBITS) % 32;
        uint32_t zero_v1 = qzeros == nullptr ? 0x88888888 : qzeros[scale_zero_from * qzero_width + zero_col_to];
        zv1[i].x |= (zero_v1 & ((1 << zero_bits_first) - 1)) << (32 - zero_bits_last);

        zv1[i].y = (zero_v1 >> zero_bits_first) & max_num_in_bits;
      } else {
        zv1[i].y = (zero_v >> (zero_bits_last + WBITS)) & max_num_in_bits;
      }

      if (zero_col_to != zero_col_to_2) {
        const int zero_bits_first = ((half_col_ind + 2) * WBITS) % 32;
        uint32_t zero_v1 = qzeros == nullptr ? 0x88888888 : qzeros[scale_zero_from * qzero_width + zero_col_to_2];
        zv1[i].y |= (zero_v1 & ((1 << zero_bits_first) - 1)) << (32 - zero_bits_last - WBITS);
      }
      hzeros[i] = __halves2half2(__ushort2half_rn(zv1[i].x), __ushort2half_rn(zv1[i].y));
    }
  } else {
    const half2* qzeros2 = reinterpret_cast<const half2*>(scale);
    for (int i = 0, scale_zero_from_i = scale_zero_from; scale_zero_from_i <= scale_zero_to; scale_zero_from_i++, i++) {
      hzeros[i] = (qzeros2[scale_zero_from_i * half_n + col_ind]);
    }
  }
  half2 scale_zeros[4];
  for (int i = 0; i <= scale_zero_to - scale_zero_from; i++) {
    scale_zeros[i] = __hmul2(hzeros[i], scale_v[i]);
  }
  half2 scale_2 = scale_v[0];
  half2 scale_zeros_2 = scale_zeros[0];

  const int out_offset = ((fp16_weight_in_row)*half_n + col_ind);
  half2* out_h2 = reinterpret_cast<half2*>(out);
  // decompress weights
  int remains = in_features - fp16_weight_in_row;
  if (remains >= compress_group_size) {
#pragma unroll
    for (int i = 0; i < compress_group_size / 2; i++) {
      uchar2 wv1 = IteratorQweight<uint2, WBITS>(qweight_thread, i);
      uchar2 wv2 = IteratorQweight<uint2, WBITS>(qweight_thread, 16 + i);

      half2 wv = __halves2half2(__ushort2half_rn(wv1.x), __ushort2half_rn(wv1.y));
      if (group_size < 32) {
        half2 scale_2 = scale_v[i / group_size];
        half2 scale_zeros_2 = scale_zeros[i / group_size];
      }
      half2 res = __hfma2(wv, scale_2, -scale_zeros_2);
      out_h2[out_offset + i * half_n] = res;

      wv = __halves2half2(__ushort2half_rn(wv2.x), __ushort2half_rn(wv2.y));
      if (group_size < 32) {
        half2 scale_2 = scale_v[(i + 16) / group_size];
        half2 scale_zeros_2 = scale_zeros[(i + 16) / group_size];
      }
      res = __hfma2(wv, scale_2, -scale_zeros_2);
      out_h2[(out_offset + (i + 16) * half_n)] = res;
    }
  } else {
    // decompress weights
    for (int i = 0; i < remains; i++) {
      uchar2 wv1 = IteratorQweight<uint2, WBITS>(qweight_thread, i);

      half2 wv = __halves2half2(__ushort2half_rn(wv1.x), __ushort2half_rn(wv1.y));
      if (group_size < 32) {
        scale_2 = scale_v[i / group_size];
        scale_zeros_2 = scale_zeros[i / group_size];
      }
      half2 res = __hfma2(wv, scale_2, -scale_zeros_2);
      out_h2[out_offset + i * half_n] = res;
    }
  }
}

template <typename scalar_t, int WBITS>
__global__ void DequantizeAndUnpackWeight357Gidx(
    scalar_t* out, const uint32_t* qweight, const scalar_t* scale, const uint32_t* qzeros,
    const int32_t* g_idx, int group_size, const int in_features, const int n,
    uint8_t add_zero_bias) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  int out_x = tid % n;
  int out_y = tid / n;
  int scale_row = g_idx[out_y];

  const int max_num_in_bits = (1 << WBITS) - 1;

  const int qzero_width = (n * WBITS + 32 - 1) / 32;
  scalar_t scale_v = scale[scale_row * n + out_x];
  uint32_t zero_v1 = 0x88888888;
  uint8_t zv1 = 0;
  if (qzeros != nullptr) {
    int start_bits = out_x * WBITS;
    int first = start_bits / 32;
    int end_bits = (start_bits + WBITS);
    int second = end_bits / 32;
    start_bits = start_bits % 32;
    end_bits = end_bits % 32;

    zero_v1 = qzeros[scale_row * qzero_width + first];
    zv1 = (zero_v1 >> start_bits) & max_num_in_bits;
    if (first != second) {
      zero_v1 = qzeros[scale_row * qzero_width + second];
      zv1 |= (zero_v1 & ((1 << end_bits) - 1)) << (32 - start_bits);
    }
  }

  scalar_t scale_zeros = __hmul(scale_v, __ushort2half_rn(zv1 + add_zero_bias));

  uint32_t weight_int = 0;
  uint8_t wv1 = 0;
  {
    int start_bits = out_y * WBITS;
    int first = start_bits / 32;
    int end_bits = (start_bits + WBITS);
    int second = end_bits / 32;
    start_bits = start_bits % 32;
    end_bits = end_bits % 32;

    weight_int = qweight[first * n + out_x];
    wv1 = (weight_int >> start_bits) & max_num_in_bits;
    if (first != second) {
      weight_int = qweight[second * n + out_x];
      wv1 |= (weight_int & ((1 << end_bits) - 1)) << (32 - start_bits);
    }
  }

  scalar_t wv = __ushort2half_rn(wv1);
  out[tid] = __hfma(wv, scale_v, -scale_zeros);
}

template <typename scalar_t, int WBITS>
__global__ void DequantizeAndUnpackWeight248Gidx(scalar_t* out, const uint32_t* qweight, const scalar_t* scale, const uint32_t* qzeros, const int32_t* g_idx,
                                                 int group_size, const int in_features, const int n, uint8_t add_zero_bias) {
  int bid = blockIdx.x;
  int tid = (bid * kBlockSize + threadIdx.x);
  int out_x = tid % n;
  int out_y = tid / n;
  int scale_row = g_idx[out_y];

  const int compress_group_size = 32 / WBITS;
  const int max_num_in_bits = (1 << WBITS) - 1;

  scalar_t scale_v = scale[scale_row * n + out_x];
  uint32_t zero_v = qzeros == nullptr
                        ? 0x88888888
                        : qzeros[scale_row * (n / compress_group_size) +
                                 (out_x / compress_group_size)];
  int zero_ind = out_x % compress_group_size;
  uint8_t zv1 = (zero_v >> (zero_ind * WBITS)) & max_num_in_bits;

  scalar_t scale_zeros = __hmul(scale_v, __ushort2half_rn(zv1 + add_zero_bias));

  int weight_int = qweight[(out_y / compress_group_size) * n + out_x];
  int weight_ind = (out_y) % compress_group_size;
  uint8_t wv1 = (weight_int >> (weight_ind * WBITS)) & max_num_in_bits;
  scalar_t wv = __ushort2half_rn(wv1);
  out[tid] = __hfma(wv, scale_v, -scale_zeros);
}

void DequantWeightNbitGidx(cudaStream_t stream,
                           const int32_t* qweight_i32_i, const void* scale_fp16,
                           const int32_t* qzeros_i32_i, const int32_t* g_dix,
                           void* b_fp16,
                           uint32_t mat_k, uint32_t mat_n, int bits,
                           int groupsize) {
  using scalar_t = half;
  int add_zero_bias = 0;
  dim3 gridDim = {mat_k * mat_n / kBlockSize};
  dim3 blockDim = {kBlockSize};

  const uint32_t* qweight_i32 = reinterpret_cast<const uint32_t*>(qweight_i32_i);
  const uint32_t* qzeros_i32 = reinterpret_cast<const uint32_t*>(qzeros_i32_i);
#define CASE_EVEN(WBITS)                                                                                                       \
  case WBITS:                                                                                                                  \
    DequantizeAndUnpackWeight248Gidx<scalar_t, WBITS>                                                                          \
        <<<gridDim, blockDim, 0, stream>>>(                                                                                    \
            (scalar_t*)b_fp16, qweight_i32, (scalar_t*)scale_fp16, qzeros_i32, g_dix, groupsize, mat_k, mat_n, add_zero_bias); \
    break;
#define CASE_ODD(WBITS)                                                                                                        \
  case WBITS:                                                                                                                  \
    DequantizeAndUnpackWeight357Gidx<scalar_t, WBITS>                                                                          \
        <<<gridDim, blockDim, 0, stream>>>(                                                                                    \
            (scalar_t*)b_fp16, qweight_i32, (scalar_t*)scale_fp16, qzeros_i32, g_dix, groupsize, mat_k, mat_n, add_zero_bias); \
    break;
  switch (bits) {
    CASE_EVEN(2);
    CASE_EVEN(4);
    CASE_EVEN(8);
    CASE_ODD(3);
    CASE_ODD(5);
    CASE_ODD(6);
    CASE_ODD(7);
    default:
      printf("error bits\n");
      assert(false);
  }
#undef CASE_EVEN
#undef CASE_ODD
}

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
    uint32_t groupsize) {
  uint32_t compress_ratio = sizeof(uint32_t) / bits;
  if (bits != 2 && bits != 4 && bits != 8) {
    compress_ratio = 32;
  }
  dim3 gridDim = {(matrix_n / 2 * ((matrix_k + compress_ratio - 1) / compress_ratio) + kBlockSize - 1) / kBlockSize};
  dim3 blockDim = {kBlockSize};
#define CASE_EVEN(WBITS)                                                          \
  case WBITS:                                                                     \
    kDequantizeAndUnpackWeight248<half, WBITS><<<gridDim, blockDim, 0, stream>>>( \
        (half*)weight_out, qweight_i32, (half*)scales_data, zeros_data,           \
        groupsize, matrix_k, matrix_n);                                           \
    break;
#define CASE_ODD(WBITS)                                                           \
  case WBITS:                                                                     \
    DequantizeAndUnpackWeight3567<half, WBITS><<<gridDim, blockDim, 0, stream>>>( \
        (half*)weight_out, (const uint32_t*)qweight_i32, (half*)scales_data,      \
        zeros_data, groupsize, matrix_k, matrix_n);              \
    break;
  switch (bits) {
    CASE_EVEN(2);
    CASE_EVEN(4);
    CASE_EVEN(8);
    CASE_ODD(3);
    CASE_ODD(5);
    CASE_ODD(6);
    CASE_ODD(7);
    default:
      break;
  }
#undef CASE_EVEN
#undef CASE_ODD
}
template void DequantWeightNbit<uint32_t>(
    cudaStream_t stream,
    const int32_t* qweight_i32,
    const void* scales_data,
    const uint32_t* zeros_data,
    void* weight_out,
    uint32_t matrix_k,
    uint32_t matrix_n,
    uint32_t bits,
    uint32_t groupsize);
template void DequantWeightNbit<half>(
    cudaStream_t stream,
    const int32_t* qweight_i32,
    const void* scales_data,
    const half* zeros_data,
    void* weight_out,
    uint32_t matrix_k,
    uint32_t matrix_n,
    uint32_t bits,
    uint32_t groupsize);
}  // namespace GPTQPacking


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
