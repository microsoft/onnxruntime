// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cuda/quantization/qorder_unary_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

struct __half4 {
  __half2 xy;
  __half2 zw;
};

union U4S2 {
  unsigned u4;
  short2 s2;
};

__device__ inline char4 quantize_half4_char4(__half4 val4, const __half2& inverse_scale2) {
  val4.xy *= inverse_scale2;
  val4.zw *= inverse_scale2;
  U4S2 shortxy, shortzw;
  shortxy.s2.x = __half2short_rn(__low2half(val4.xy));
  shortzw.s2.x = __half2short_rn(__low2half(val4.zw));
  shortxy.s2.y = __half2short_rn(__high2half(val4.xy));
  shortzw.s2.y = __half2short_rn(__high2half(val4.zw));
  shortxy.u4 = __vmaxs2(__vmins2(shortxy.u4, 0x007F007F), 0xFF80FF80);
  shortzw.u4 = __vmaxs2(__vmins2(shortzw.u4, 0x007F007F), 0xFF80FF80);
  return char4{(char)shortxy.s2.x, (char)shortxy.s2.y, (char)shortzw.s2.x, (char)shortzw.s2.y};
}

__device__ inline __half4 deqantize_char4_half4(const char4& ch4, const __half2& scale2) {
  return {scale2 * __half2(__short2half_rn(ch4.x), __short2half_rn(ch4.y)),
          scale2 * __half2(__short2half_rn(ch4.z), __short2half_rn(ch4.w))};
}

// input layout
static constexpr unsigned COL_TILES_PER_BLOCK = 4;  // multiply of 4, i.e. 8, 12, ...
static constexpr unsigned COLS_PER_BLOCK = COL_TILES_PER_BLOCK * 32;
static constexpr unsigned COLS_IN4_PER_BLOCK = COLS_PER_BLOCK / 4;
// static constexpr unsigned COLS_IN2_PER_BLOCK = COLS_PER_BLOCK / 2;
static constexpr unsigned ROWS_PER_BLOCK = 16;
static constexpr unsigned QORDER_THREADS_PER_BLOCK = 256;

static constexpr unsigned BLOCK_SIZE_IN4 = ROWS_PER_BLOCK * COLS_IN4_PER_BLOCK;
// static constexpr unsigned BLOCK_SIZE_IN2 = ROWS_PER_BLOCK * COLS_IN2_PER_BLOCK;
static constexpr unsigned THREAD_ITERATE_COUNT_IN4 = (BLOCK_SIZE_IN4 + QORDER_THREADS_PER_BLOCK - 1) / QORDER_THREADS_PER_BLOCK;
// static constexpr unsigned THREAD_ITERATE_COUNT_IN2 = (BLOCK_SIZE_IN2 + QORDER_THREADS_PER_BLOCK - 1) / QORDER_THREADS_PER_BLOCK;

// quantized matrix layout constants
static constexpr unsigned QMAT_COL_TILES_PER_BLOCK = ROWS_PER_BLOCK;
static constexpr unsigned QMAT_COLS_PER_BLOCK = QMAT_COL_TILES_PER_BLOCK * 32;
static constexpr unsigned QMAT_COLS_IN4_PER_BLOCK = QMAT_COLS_PER_BLOCK / 4;
// static constexpr unsigned QMAT_COLS_IN2_PER_BLOCK = QMAT_COLS_PER_BLOCK / 2;
static constexpr unsigned QMAT_ROWS_PER_BLOCK = COL_TILES_PER_BLOCK;

__global__ void QuantizeHalfRow_S8Col32_Kernel(const half* src, int8_t* dst, const __half2 inverse_scale2, size_t rows, size_t cols) {
  __shared__ char4 shm[QMAT_ROWS_PER_BLOCK][QMAT_COLS_IN4_PER_BLOCK + 1];

  size_t batch_start = blockIdx.z * (rows * cols);  // batchSize = gridDims.z
  src += batch_start;
  dst += batch_start;

  unsigned row_start = blockIdx.x * ROWS_PER_BLOCK;
  unsigned col_start = blockIdx.y * COLS_PER_BLOCK;
  src += (size_t)row_start * cols + col_start;  // src now is the block start
  unsigned idx = threadIdx.x;
#pragma unroll
  for (unsigned i = 0; i < THREAD_ITERATE_COUNT_IN4; i++) {
    unsigned r = idx / COLS_IN4_PER_BLOCK;
    unsigned c_in_4 = idx & (COLS_IN4_PER_BLOCK - 1);
    if (r + row_start < rows && ((c_in_4 * 4) + col_start) < cols) {
      __half4 val4 = *(const __half4*)(src + (r * cols) + (c_in_4 * 4));
      char4 ch4 = quantize_half4_char4(val4, inverse_scale2);
      shm[c_in_4 / 8][r * 8 + (c_in_4 & 0x7)] = ch4;
    }
    idx += QORDER_THREADS_PER_BLOCK;
  }

  __syncthreads();

  size_t qmat_rows = cols / 32;
  size_t qmat_cols = rows * 32;
  unsigned qmat_block_upper_row = blockIdx.y * QMAT_ROWS_PER_BLOCK;
  unsigned qmat_block_left_col = blockIdx.x * QMAT_COLS_PER_BLOCK;
  dst += (size_t)qmat_block_upper_row * qmat_cols + qmat_block_left_col;  // dst now is the block start

  idx = threadIdx.x;
#pragma unroll
  for (unsigned i = 0; i < THREAD_ITERATE_COUNT_IN4; i++) {
    unsigned r = idx / QMAT_COLS_IN4_PER_BLOCK;
    unsigned c_in_4 = idx & (QMAT_COLS_IN4_PER_BLOCK - 1);
    if (r + qmat_block_upper_row < qmat_rows && ((c_in_4 * 4) + qmat_block_left_col) < qmat_cols) {
      char4* thread_out = (char4*)(dst + (r * qmat_cols) + (c_in_4 * 4));
      *thread_out = shm[r][c_in_4];
    }
    idx += QORDER_THREADS_PER_BLOCK;
  }
}

// rows and cols are the dim of the non-quantized (dst) matrix
// QORDER_THREADS_PER_BLOCK must be same as threads per block when calling this
__global__ void DequantizeS8Col32_HalfRow_Kernel(const int8_t* src, half* dst, const __half2 scale2, size_t rows, size_t cols) {
  __shared__ char4 shm[QMAT_ROWS_PER_BLOCK][QMAT_COLS_IN4_PER_BLOCK + 1];

  size_t batch_start = blockIdx.z * (rows * cols);
  src += batch_start;
  dst += batch_start;

  size_t qmat_rows = cols / 32;
  size_t qmat_cols = rows * 32;
  unsigned qmat_block_upper_row = blockIdx.y * QMAT_ROWS_PER_BLOCK;
  unsigned qmat_block_left_col = blockIdx.x * QMAT_COLS_PER_BLOCK;
  src += (size_t)qmat_block_upper_row * qmat_cols + qmat_block_left_col;  // point to block start

  unsigned idx = threadIdx.x;
#pragma unroll
  for (unsigned i = 0; i < THREAD_ITERATE_COUNT_IN4; i++) {
    unsigned r = idx / QMAT_COLS_IN4_PER_BLOCK;
    unsigned c_in_4 = idx & (QMAT_COLS_IN4_PER_BLOCK - 1);
    if ((r + qmat_block_upper_row < qmat_rows) && ((c_in_4 * 4) + qmat_block_left_col) < qmat_cols) {
      shm[r][c_in_4] = *(const char4*)(src + (r * qmat_cols) + (c_in_4 * 4));
    }
    idx += QORDER_THREADS_PER_BLOCK;
  }
  __syncthreads();

  unsigned row_start = blockIdx.x * ROWS_PER_BLOCK;
  unsigned col_start = blockIdx.y * COLS_PER_BLOCK;
  dst += (size_t)row_start * cols + col_start;  // point to the block start
  idx = threadIdx.x;
#pragma unroll
  for (unsigned i = 0; i < THREAD_ITERATE_COUNT_IN4; i++) {
    unsigned r = idx / COLS_IN4_PER_BLOCK;
    unsigned c_in_4 = idx & (COLS_IN4_PER_BLOCK - 1);
    if ((r + row_start < rows) && ((c_in_4 * 4) + col_start) < cols) {
      __half4 hf4 = deqantize_char4_half4(shm[c_in_4 / 8][r * 8 + (c_in_4 & 0x7)], scale2);
      *(__half4*)(dst + (r * cols) + (c_in_4 * 4)) = hf4;
    }
    idx += QORDER_THREADS_PER_BLOCK;
  }
}

void QOrderQuantizeHalfRow_S8Col32(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                                   const half* src, int8_t* dst, float scale, int batch, size_t rows, size_t cols) {
  __half2 inverse_scale2 = __floats2half2_rn(1.0f / scale, 1.0f / scale);
  dim3 threads(QORDER_THREADS_PER_BLOCK);
  unsigned tiles_per_row = (unsigned)(cols / 32);
  dim3 blocks((unsigned)((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK), (unsigned)((tiles_per_row + COL_TILES_PER_BLOCK - 1) / COL_TILES_PER_BLOCK), (unsigned)batch);
  QuantizeHalfRow_S8Col32_Kernel<<<blocks, threads, 0, stream>>>(src, dst, inverse_scale2, rows, cols);
}

void QOrderDequantizeS8Col32_HalfRow(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                                     const int8_t* src, half* dst, float scale, int batch, size_t rows, size_t cols) {
  __half2 scale2 = __floats2half2_rn(scale, scale);
  dim3 threads(QORDER_THREADS_PER_BLOCK);
  unsigned tiles_per_row = (unsigned)(cols / 32);
  dim3 blocks((unsigned)((rows + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK), (unsigned)((tiles_per_row + COL_TILES_PER_BLOCK - 1) / COL_TILES_PER_BLOCK), (unsigned)batch);
  DequantizeS8Col32_HalfRow_Kernel<<<blocks, threads, 0, stream>>>(src, dst, scale2, rows, cols);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
