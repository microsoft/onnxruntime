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

__device__ inline int8_t quantize_float_s8(const float& val, const float& inverse_scale) {
  float dqval = fmaxf(fminf(127.0f, val * inverse_scale), -128.0f);
  return static_cast<int8_t>(__float2int_rn(dqval));
}

__device__ inline char4 quantize_half4_char4(__half4 val4, const __half2 inverse_scale2) {
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

__device__ inline __half4 deqantize_char4_half4(const char4 ch4, const __half2 scale2) {
  return {scale2 * __half2(__short2half_rn(ch4.x), __short2half_rn(ch4.y)),
          scale2 * __half2(__short2half_rn(ch4.z), __short2half_rn(ch4.w))};
}

template <typename T>
__inline__ __device__ T
WarpReduceSum(T val) {
  val += __shfl_xor_sync(0xFFFFFFFF, val, 1);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 2);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 4);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 8);
  val += __shfl_xor_sync(0xFFFFFFFF, val, 16);
  return val;
}

// source matrix block 32 x 32, each thread handle 4 int8 items, so:
// thread block size should be (8 cols_in_4, 32 rows, 1)
// grid size ((cols + 31) / 32, (rows + 31) / 32), batch)
__global__ void
QOrderQuantizeHalfRowToCol32Kernel(const half* __restrict__ src, size_t src_batch_stride,
                                   int8_t* __restrict__ dst, size_t dst_batch_stride,
                                   const __half2 inverse_scale2, unsigned rows, unsigned cols) {
  unsigned int c = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (c < cols && r < rows) {
    const size_t src_index = (src_batch_stride * blockIdx.z) + (r * cols + c);
    const size_t dst_index = (dst_batch_stride * blockIdx.z) + ((c & 0xffffffe0) * rows + (r << 5) + (c & 0x1F));
    __half4 const src_val4 = *((const __half4*)(src + src_index));
    *(char4*)(dst + dst_index) = quantize_half4_char4(src_val4, inverse_scale2);
  }
}

// source matrix block 32 x 32, each thread handle ElementsPerThread int8 items, so:
// thread block size should be (32 cols, 32 rows, 1)
// grid size (((cols / 32) + ElementsPerThread - 1) / ElementsPerThread), (rows + 31) / 32), batch)
template <unsigned ElementsPerThread = 4>
__global__ void
QOrderQuantizeFloatRowToCol32Kernel(const float* __restrict__ src, size_t src_batch_stride,
                                    int8_t* __restrict__ dst, size_t dst_batch_stride,
                                    const float inverse_scale, unsigned rows, unsigned cols) {
  unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r < rows) {
    const unsigned kColsPerIncrement = blockDim.x << 5;
    unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
    size_t src_index = (src_batch_stride * blockIdx.z) + (r * cols + c);
    size_t dst_index = (dst_batch_stride * blockIdx.z) + ((c & 0xffffffe0) * rows + (r << 5) + (c & 0x1f));

#pragma unroll
    for (int i = 0; i < ElementsPerThread; i++) {
      if (c < cols) {
        *(dst + dst_index) = quantize_float_s8(*(src + src_index), inverse_scale);
        c += kColsPerIncrement;
        src_index += kColsPerIncrement;
        dst_index += kColsPerIncrement * rows;
      }
    }
  }
}

// target matrix block 32 x 32, each thread handle 4 int8 items, so:
// thread block size should be (8 cols_in_4, 32 rows, 1)
// grid size ((cols + 31) / 32, (rows + 31) / 32), batch)
__global__ void
QOrderDequantizeCol32ToHalfRowKernel(const int8_t* __restrict__ src, size_t src_batch_stride,
                                     __half* __restrict__ dst, size_t dst_batch_stride,
                                     const __half2 scale2, unsigned rows, unsigned cols) {
  unsigned int c = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (c < cols && r < rows) {
    const size_t dst_index = (dst_batch_stride * blockIdx.z) + (r * cols + c);
    const size_t src_index = (src_batch_stride * blockIdx.z) + ((c & 0xffffffe0) * rows + (r << 5) + (c & 0x1F));
    const char4 src_ch4 = *((const char4*)(src + src_index));
    *(__half4*)(dst + dst_index) = deqantize_char4_half4(src_ch4, scale2);
  }
}

// // input layout
// static constexpr unsigned COL_TILES_PER_BLOCK = 4;  // multiply of 4, i.e. 8, 12, ...
// static constexpr unsigned COLS_PER_BLOCK = COL_TILES_PER_BLOCK * 32;
// static constexpr unsigned COLS_IN4_PER_BLOCK = COLS_PER_BLOCK / 4;
// // static constexpr unsigned COLS_IN2_PER_BLOCK = COLS_PER_BLOCK / 2;
// static constexpr unsigned ROWS_PER_BLOCK = 16;
// static constexpr unsigned QORDER_THREADS_PER_BLOCK = 256;

// static constexpr unsigned BLOCK_SIZE_IN4 = ROWS_PER_BLOCK * COLS_IN4_PER_BLOCK;
// // static constexpr unsigned BLOCK_SIZE_IN2 = ROWS_PER_BLOCK * COLS_IN2_PER_BLOCK;
// static constexpr unsigned THREAD_ITERATE_COUNT_IN4 = (BLOCK_SIZE_IN4 + QORDER_THREADS_PER_BLOCK - 1) / QORDER_THREADS_PER_BLOCK;
// // static constexpr unsigned THREAD_ITERATE_COUNT_IN2 = (BLOCK_SIZE_IN2 + QORDER_THREADS_PER_BLOCK - 1) / QORDER_THREADS_PER_BLOCK;

// // quantized matrix layout constants
// static constexpr unsigned QMAT_COL_TILES_PER_BLOCK = ROWS_PER_BLOCK;
// static constexpr unsigned QMAT_COLS_PER_BLOCK = QMAT_COL_TILES_PER_BLOCK * 32;
// static constexpr unsigned QMAT_COLS_IN4_PER_BLOCK = QMAT_COLS_PER_BLOCK / 4;
// // static constexpr unsigned QMAT_COLS_IN2_PER_BLOCK = QMAT_COLS_PER_BLOCK / 2;
// static constexpr unsigned QMAT_ROWS_PER_BLOCK = COL_TILES_PER_BLOCK;

// __global__ void QuantizeHalfRow_S8Col32_Kernel(const half* src, int8_t* dst, const __half2 inverse_scale2, size_t rows, size_t cols) {
//   __shared__ char4 shm[QMAT_ROWS_PER_BLOCK][QMAT_COLS_IN4_PER_BLOCK + 1];

//   size_t batch_start = blockIdx.z * (rows * cols);  // batchSize = gridDims.z
//   src += batch_start;
//   dst += batch_start;

//   unsigned row_start = blockIdx.x * ROWS_PER_BLOCK;
//   unsigned col_start = blockIdx.y * COLS_PER_BLOCK;
//   src += (size_t)row_start * cols + col_start;  // src now is the block start
//   unsigned idx = threadIdx.x;
// #pragma unroll
//   for (unsigned i = 0; i < THREAD_ITERATE_COUNT_IN4; i++) {
//     unsigned r = idx / COLS_IN4_PER_BLOCK;
//     unsigned c_in_4 = idx & (COLS_IN4_PER_BLOCK - 1);
//     if (r + row_start < rows && ((c_in_4 * 4) + col_start) < cols) {
//       __half4 val4 = *(const __half4*)(src + (r * cols) + (c_in_4 * 4));
//       char4 ch4 = quantize_half4_char4(val4, inverse_scale2);
//       shm[c_in_4 / 8][r * 8 + (c_in_4 & 0x7)] = ch4;
//     }
//     idx += QORDER_THREADS_PER_BLOCK;
//   }

//   __syncthreads();

//   size_t qmat_rows = cols / 32;
//   size_t qmat_cols = rows * 32;
//   unsigned qmat_block_upper_row = blockIdx.y * QMAT_ROWS_PER_BLOCK;
//   unsigned qmat_block_left_col = blockIdx.x * QMAT_COLS_PER_BLOCK;
//   dst += (size_t)qmat_block_upper_row * qmat_cols + qmat_block_left_col;  // dst now is the block start

//   idx = threadIdx.x;
// #pragma unroll
//   for (unsigned i = 0; i < THREAD_ITERATE_COUNT_IN4; i++) {
//     unsigned r = idx / QMAT_COLS_IN4_PER_BLOCK;
//     unsigned c_in_4 = idx & (QMAT_COLS_IN4_PER_BLOCK - 1);
//     if (r + qmat_block_upper_row < qmat_rows && ((c_in_4 * 4) + qmat_block_left_col) < qmat_cols) {
//       char4* thread_out = (char4*)(dst + (r * qmat_cols) + (c_in_4 * 4));
//       *thread_out = shm[r][c_in_4];
//     }
//     idx += QORDER_THREADS_PER_BLOCK;
//   }
// }

// // rows and cols are the dim of the non-quantized (dst) matrix
// // QORDER_THREADS_PER_BLOCK must be same as threads per block when calling this
// __global__ void DequantizeS8Col32_HalfRow_Kernel(const int8_t* src, half* dst, const __half2 scale2, size_t rows, size_t cols) {
//   __shared__ char4 shm[QMAT_ROWS_PER_BLOCK][QMAT_COLS_IN4_PER_BLOCK + 1];

//   size_t batch_start = blockIdx.z * (rows * cols);
//   src += batch_start;
//   dst += batch_start;

//   size_t qmat_rows = cols / 32;
//   size_t qmat_cols = rows * 32;
//   unsigned qmat_block_upper_row = blockIdx.y * QMAT_ROWS_PER_BLOCK;
//   unsigned qmat_block_left_col = blockIdx.x * QMAT_COLS_PER_BLOCK;
//   src += (size_t)qmat_block_upper_row * qmat_cols + qmat_block_left_col;  // point to block start

//   unsigned idx = threadIdx.x;
// #pragma unroll
//   for (unsigned i = 0; i < THREAD_ITERATE_COUNT_IN4; i++) {
//     unsigned r = idx / QMAT_COLS_IN4_PER_BLOCK;
//     unsigned c_in_4 = idx & (QMAT_COLS_IN4_PER_BLOCK - 1);
//     if ((r + qmat_block_upper_row < qmat_rows) && ((c_in_4 * 4) + qmat_block_left_col) < qmat_cols) {
//       shm[r][c_in_4] = *(const char4*)(src + (r * qmat_cols) + (c_in_4 * 4));
//     }
//     idx += QORDER_THREADS_PER_BLOCK;
//   }
//   __syncthreads();

//   unsigned row_start = blockIdx.x * ROWS_PER_BLOCK;
//   unsigned col_start = blockIdx.y * COLS_PER_BLOCK;
//   dst += (size_t)row_start * cols + col_start;  // point to the block start
//   idx = threadIdx.x;
// #pragma unroll
//   for (unsigned i = 0; i < THREAD_ITERATE_COUNT_IN4; i++) {
//     unsigned r = idx / COLS_IN4_PER_BLOCK;
//     unsigned c_in_4 = idx & (COLS_IN4_PER_BLOCK - 1);
//     if ((r + row_start < rows) && ((c_in_4 * 4) + col_start) < cols) {
//       __half4 hf4 = deqantize_char4_half4(shm[c_in_4 / 8][r * 8 + (c_in_4 & 0x7)], scale2);
//       *(__half4*)(dst + (r * cols) + (c_in_4 * 4)) = hf4;
//     }
//     idx += QORDER_THREADS_PER_BLOCK;
//   }
// }

// cols could be divide by 32
void QOrderQuantizeRowToCol32(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                              const __half* src, int8_t* dst, float scale,
                              unsigned batch, unsigned rows, unsigned cols) {
  if (cols & 0x1f) {
    throw std::runtime_error("cols can not divide by 32!");
  }

  __half2 inverse_scale2 = __float2half2_rn(1.0f / scale);
  dim3 threads(8, 32);
  dim3 blocks(cols / 32, (rows + 31) / 32, batch);
  size_t stride = (size_t)rows * cols;
  QOrderQuantizeHalfRowToCol32Kernel<<<blocks, threads, 0, stream>>>(src, stride, dst, stride, inverse_scale2, rows, cols);
}

// cols could be divide by 32
void QOrderQuantizeRowToCol32(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                              const float* src, int8_t* dst, float scale,
                              unsigned batch, unsigned rows, unsigned cols) {
  if (cols & 0x1f) {
    throw std::runtime_error("cols can not divide by 32!");
  }

  constexpr unsigned kElementsPerThread = 4;
  float inverse_scale = 1.0f / scale;
  dim3 threads(32, 32);
  dim3 blocks(((cols / 32) + kElementsPerThread - 1) / kElementsPerThread, (rows + 31) / 32, batch);
  size_t stride = (size_t)rows * cols;
  QOrderQuantizeFloatRowToCol32Kernel<<<blocks, threads, 0, stream>>>(src, stride, dst, stride, inverse_scale, rows, cols);
}

// cols could be divide by 32
void QOrderDequantizeCol32ToRow(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                                const int8_t* src, __half* dst, float scale,
                                unsigned batch, unsigned rows, unsigned cols) {
  if (cols & 0x1f) {
    throw std::runtime_error("cols can not divede by 32");
  }

  __half2 scale2 = __float2half2_rn(scale);
  dim3 threads(8, 32);
  dim3 blocks(cols / 32, (rows + 31) / 32, batch);
  size_t stride = (size_t)rows * cols;
  QOrderDequantizeCol32ToHalfRowKernel<<<blocks, threads, 0, stream>>>(src, stride, dst, stride, scale2, rows, cols);
}

void QOrderDequantizeCol32ToRow(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                                const int8_t* src, float* dst, float scale,
                                unsigned batch, unsigned rows, unsigned cols) {
  if (cols & 0x1f) {
    throw std::runtime_error("cols can not divede by 32");
  }
  throw std::runtime_error("Comming soon...");
}

static constexpr unsigned QORDER_LAYERNORM_ROWS_PER_BLOCK = 8;  // 4, 8, 16, ...
// block_size = (32, QORDER_LAYERNORM_ROWS_PER_BLOCK, 1)
// grid_size = ((rows + QORDER_LAYERNORM_ROWS_PER_BLOCK - 1) / QORDER_LAYERNORM_ROWS_PER_BLOCK, batch, 1)
__global__ void
QOrderLayerNormKernel(const int8_t* __restrict__ src, const float src_scale, int8_t* __restrict__ dst, const float dst_scale,
                      const __half* __restrict__ gamma, const __half* __restrict__ beta, const float epsilon,
                      const unsigned rows, const unsigned cols) {
  int32_t sum = 0;
  int32_t square_sum = 0;
  unsigned r = blockIdx.x * QORDER_LAYERNORM_ROWS_PER_BLOCK + threadIdx.y;
  if (rows <= r) return;

  const unsigned STRIDES_PER_WARP_ROUND = rows << 7;  // * 32 * 4
  unsigned c = threadIdx.x << 2;
  const size_t batch_row_index = (size_t)blockIdx.y * (rows * cols) + ((c & 0xffffffe0) * rows + (r << 5) + (c & 31));
  src += batch_row_index;
  dst += batch_row_index;
  for (unsigned index = 0; c < cols; c += 128, index += STRIDES_PER_WARP_ROUND) {
    char4 ch4 = *((const char4*)(src + index));
    sum += ((short)ch4.x + (short)ch4.y + (short)ch4.z + (short)ch4.w);
    square_sum = __dp4a(ch4, ch4, square_sum);
  }

  sum = WarpReduceSum<int32_t>(sum);
  square_sum = WarpReduceSum<int32_t>(square_sum);

  const float mean = (src_scale * sum / cols);
  const float rvar = rsqrtf(src_scale * src_scale * ((float)square_sum - ((float)sum * sum / cols)) / cols + epsilon);
  const __half2 mean2 = __float2half2_rn(mean);
  const __half2 var2 = __float2half2_rn(rvar);
  const __half2 src_scale2 = __float2half2_rn(src_scale);
  const __half2 dst_rscale2 = __float2half2_rn(1.0f / dst_scale);
  const __half4 zero4 = {__float2half2_rn(0.0f), __float2half2_rn(0.0f)};

  for (unsigned index = 0, c = threadIdx.x * 4; c < cols; c += 128, index += STRIDES_PER_WARP_ROUND) {
    char4 ch4 = __ldg((const char4*)(src + index));
    __half4 dqval4 = deqantize_char4_half4(ch4, src_scale2);
    const __half4 g4 = *((const __half4*)(gamma + c));
    const __half4 b4 = (beta == nullptr) ? zero4 : *((const __half4*)(beta + c));
    dqval4.xy = __hfma2(__hmul2(__hsub2(dqval4.xy, mean2), var2), g4.xy, b4.xy);
    dqval4.zw = __hfma2(__hmul2(__hsub2(dqval4.zw, mean2), var2), g4.zw, b4.zw);
    *(char4*)(dst + index) = quantize_half4_char4(dqval4, dst_rscale2);
  }
}

void QOrderLayerNorm(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                     const int8_t* src, const float src_scale, int8_t* dst, const float dst_scale,
                     const __half* gamma, const __half* beta, const float epsilon,
                     const unsigned batch, const unsigned rows, const unsigned cols) {
  dim3 threads(32, QORDER_LAYERNORM_ROWS_PER_BLOCK, 1);
  dim3 blocks((unsigned)(rows + QORDER_LAYERNORM_ROWS_PER_BLOCK - 1) / QORDER_LAYERNORM_ROWS_PER_BLOCK, (unsigned)batch, 1);
  QOrderLayerNormKernel<<<blocks, threads, 0, stream>>>(src, src_scale, dst, dst_scale, gamma, beta, epsilon, rows, cols);
}

// source matrix block 32 x 32, each thread handle 4 int8_t items,
// thread block size should be (8 cols_in_4, 32 rows, 1)
// grid size ((cols + 31) / 32, (rows + 31) / 32), batch)
__global__ void
ReorderS8RowToCol32Kernel(const int8_t* __restrict__ src, int8_t* __restrict__ dst, unsigned rows, unsigned cols) {
  unsigned int c = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (c < cols && r < rows) {
    const unsigned batch_start = blockIdx.z * (rows * cols);
    const unsigned src_index = batch_start + r * cols + c;
    const unsigned dst_index = batch_start + ((c & 0xffffffe0) * rows + (r << 5) + (c & 0x1f));
    *(char4*)(dst + dst_index) = *((const char4*)(src + src_index));
  }
}

void ReorderS8RowToCol32(cudaStream_t stream, const cudaDeviceProp& /* device_prop */,
                         const int8_t* src, int8_t* dst,
                         unsigned batch, unsigned rows, unsigned cols) {
  dim3 threads(8, 32, 1);
  dim3 blocks((unsigned)(cols / 32), (unsigned)((rows + 31) / 32), batch);
  ReorderS8RowToCol32Kernel<<<blocks, threads, 0, stream>>>(src, dst, rows, cols);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
