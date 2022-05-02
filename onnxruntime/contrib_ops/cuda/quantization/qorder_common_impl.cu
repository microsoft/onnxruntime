// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cuda/quantization/qorder_unary_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cuda_common.h"

//#include "qorder_binary_op.cuh"
//#include "core/providers/cuda/cu_inc/binary_elementwise_impl.cuh"

#include <cub/cub.cuh>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;
using namespace cub;

// todo: define __dp4a when it is not available (i.e. before compute capability 6.1)

/*
 * Utilility types and functions
 */
struct __half4 {
  __half2 xy;
  __half2 zw;
};

union U1S2 {
  unsigned u1;
  short2 s2;
};

__device__ inline int8_t quantize_float_s8(const float val, const float inverse_scale) {
  float dqval = fmaxf(fminf(127.0f, val * inverse_scale), -128.0f);
  return static_cast<int8_t>(__float2int_rn(dqval));
}

__device__ inline char2 quantize_half2_char2(const __half2 xy, const __half2 inverse_scale2) {
  __half2 scaled_xy = xy * inverse_scale2;
  U1S2 s2xy;
  s2xy.s2.x = __half2short_rn(scaled_xy.x);
  s2xy.s2.y = __half2short_rn(scaled_xy.y);
  s2xy.u1 = __vmaxs2(__vmins2(s2xy.u1, 0x007F007F), 0xFF80FF80);
  return char2{(char)s2xy.s2.x, (char)s2xy.s2.y};
}

__device__ inline char4 quantize_half4_char4(const __half4 val4, const __half2 inverse_scale2) {
  __half2 val4_xy = val4.xy * inverse_scale2;
  __half2 val4_zw = val4.zw * inverse_scale2;
  U1S2 shortxy, shortzw;
  shortxy.s2.x = __half2short_rn(__low2half(val4_xy));
  shortzw.s2.x = __half2short_rn(__low2half(val4_zw));
  shortxy.s2.y = __half2short_rn(__high2half(val4_xy));
  shortzw.s2.y = __half2short_rn(__high2half(val4_zw));
  shortxy.u1 = __vmaxs2(__vmins2(shortxy.u1, 0x007F007F), 0xFF80FF80);
  shortzw.u1 = __vmaxs2(__vmins2(shortzw.u1, 0x007F007F), 0xFF80FF80);
  return char4{(char)shortxy.s2.x, (char)shortxy.s2.y, (char)shortzw.s2.x, (char)shortzw.s2.y};
}

__device__ inline __half4 deqantize_char4_half4(const char4 ch4, const __half2 scale2) {
  return {scale2 * __half2(__short2half_rn(ch4.x), __short2half_rn(ch4.y)),
          scale2 * __half2(__short2half_rn(ch4.z), __short2half_rn(ch4.w))};
}

template <typename FloatT>
struct DequantizeVec {
};

template <>
struct DequantizeVec<float> {
  typedef char QuantizedVecT;
  typedef float DequantizedScalarT;
  static __device__ inline QuantizedVecT Quantize(const float fpvals, const float inv_scale) {
    float dqval = fmaxf(fminf(127.0f, fpvals * inv_scale), -128.0f);
    return static_cast<char>(__float2int_rn(dqval));
  }

  static __device__ inline float Dequantize(const QuantizedVecT qvals, const float scale) {
    return scale * qvals;
  }
};

template <>
struct DequantizeVec<float2> {
  typedef char2 QuantizedVecT;
  typedef float DequantizedScalarT;

  static __device__ inline QuantizedVecT Quantize(const float2 fpvals, const float inv_scale) {
    float dqvalx = fmaxf(fminf(127.0f, fpvals.x * inv_scale), -128.0f);
    float dqvaly = fmaxf(fminf(127.0f, fpvals.y * inv_scale), -128.0f);
    return char2{static_cast<char>(__float2int_rn(dqvalx)), static_cast<char>(__float2int_rn(dqvaly))};
  }

  static __device__ inline float2 Dequantize(const QuantizedVecT qvals, const float scale) {
    return float2{scale * qvals.x, scale * qvals.y};
  }
};

template <>
struct DequantizeVec<__half> {
  typedef char QuantizedVecT;
  typedef __half DequantizedScalarT;

  static __device__ inline QuantizedVecT Quantize(const __half fpvals, const __half inv_scale) {
    int i = __half2int_rn(fpvals * inv_scale);
    return static_cast<char>(min(127, max(i, -128)));
  }

  static __device__ inline __half Quantize(const QuantizedVecT qvals, const __half scale) {
    return scale * __short2half_rn(static_cast<short>(qvals));
  }
};

template <>
struct DequantizeVec<__half2> {
  typedef char2 QuantizedVecT;
  typedef __half DequantizedScalarT;

  static __device__ inline QuantizedVecT Quantize(const __half2 fpvals, const __half inv_scales) {
    __half2 xy = fpvals * __half2half2(inv_scales);
    U1S2 s2xy;
    s2xy.s2.x = __half2short_rn(xy.x);
    s2xy.s2.y = __half2short_rn(xy.y);
    s2xy.u1 = __vmaxs2(__vmins2(s2xy.u1, 0x007F007F), 0xFF80FF80);
    return char2{(char)s2xy.s2.x, (char)s2xy.s2.y};
  }

  static __device__ inline __half2 Dequantize(const QuantizedVecT qvals, const __half scale) {
    return __half2{scale * __short2half_rn(qvals.x), scale * __short2half_rn(qvals.y)};
  }
};

template <>
struct DequantizeVec<__half4> {
  typedef char4 QuantizedVecT;
  typedef __half DequantizedScalarT;

  static __device__ inline QuantizedVecT Quantize(const __half4 fpvals, const __half inv_scales) {
    return quantize_half4_char4(fpvals, __half2half2(inv_scales));
  }

  static __device__ inline __half4 Dequantize(const QuantizedVecT qvals, const __half scale) {
    return __half4{__half2{scale * __short2half_rn(qvals.x), scale * __short2half_rn(qvals.y)},
                   __half2{scale * __short2half_rn(qvals.z), scale * __short2half_rn(qvals.w)}};
  }
};

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

/************************************************************************
 * Quantize Routines:
 *   - OrderRow (fp16/32) to OrderCol32 (cols % 32 == 0)
 ************************************************************************/

// source matrix block 32 x 32, each thread handle 4 int8 items, so:
// thread block size should be (8 cols_in_4, 32 rows, 1)
// grid size ((cols + 31) / 32, (rows + 31) / 32), batch)
__global__ void
QOrderQuantizeHalfRowToCol32Kernel(const __half* __restrict__ src, size_t src_batch_stride,
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

// cols could be divide by 32
void QOrderQuantizeRowToCol32(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                              const __half* src, int8_t* dst, float scale,
                              unsigned batch, unsigned rows, unsigned cols) {
  if (cols & 0x1f) {
    throw std::runtime_error("cols can not divide by 32!");
  }

  __half2 inverse_scale2 = __float2half2_rn(1.0f / scale);
  dim3 threads(8, 32, 1);
  dim3 blocks(cols / 32, (rows + 31) / 32, batch);
  size_t stride = (size_t)rows * cols;
  QOrderQuantizeHalfRowToCol32Kernel<<<blocks, threads, 0, stream>>>(src, stride, dst, stride, inverse_scale2, rows, cols);
}

// source matrix block (32 x ElementsPerThread) x 32, each thread handle ElementsPerThread elements items, so:
// thread block size should be (32 cols, 32 rows, 1)
// grid size ((cols + 32*ElementsPerThread - 1) / (32 * ElementsPerThread), (rows + 31) / 32), batch)
template <unsigned ElementsPerThread = 4>
__global__ void
QOrderQuantizeFloatRowToCol32Kernel(const float* __restrict__ src, size_t src_batch_stride,
                                    int8_t* __restrict__ dst, size_t dst_batch_stride,
                                    const float inverse_scale, unsigned rows, unsigned cols) {
  unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
  static constexpr unsigned kColsPerIncrement = 32;  // it is the blockDim.x
  if (r < rows) {
    unsigned int c = blockIdx.x * (kColsPerIncrement * ElementsPerThread) + threadIdx.x;
    size_t src_index = (src_batch_stride * blockIdx.z) + (r * cols + c);
    size_t dst_index = (dst_batch_stride * blockIdx.z) + ((c & 0xffffffe0) * rows + (r << 5) + (c & 0x1f));

#pragma unroll
    for (int i = 0; i < ElementsPerThread; i++) {
      if (c < cols) {
        *(dst + dst_index) = quantize_float_s8(*(src + src_index), inverse_scale);
        c += kColsPerIncrement;
        src_index += kColsPerIncrement;
        dst_index += rows * kColsPerIncrement;
      }
    }
  }
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
  dim3 threads(32, 32, 1);
  dim3 blocks((cols + (32 * kElementsPerThread - 1)) / (kElementsPerThread * 32), (rows + 31) / 32, batch);
  size_t stride = (size_t)rows * cols;
  QOrderQuantizeFloatRowToCol32Kernel<<<blocks, threads, 0, stream>>>(src, stride, dst, stride, inverse_scale, rows, cols);
}

/************************************************************************
 * Dequantize Routines:
 *   - Col32 to OrderRow (fp16/32) (cols % 32 == 0)
 ************************************************************************/

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

// cols could be divide by 32
void QOrderDequantizeCol32ToRow(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                                const int8_t* src, __half* dst, float scale,
                                unsigned batch, unsigned rows, unsigned cols) {
  if (cols & 0x1f) {
    throw std::runtime_error("cols can not divede by 32");
  }

  __half2 scale2 = __float2half2_rn(scale);
  dim3 threads(8, 32, 1);
  dim3 blocks(cols / 32, (rows + 31) / 32, batch);
  size_t stride = (size_t)rows * cols;
  QOrderDequantizeCol32ToHalfRowKernel<<<blocks, threads, 0, stream>>>(src, stride, dst, stride, scale2, rows, cols);
}

// target matrix block 32 x 32, each thread handle 1 items, so:
// thread block size should be (32, 32 rows, 1)
// grid size ((cols / 32), (rows + 31) / 32), batch)
__global__ void
QOrderDequantizeCol32ToFloatRowKernel(const int8_t* __restrict__ src, size_t src_batch_stride,
                                      float* __restrict__ dst, size_t dst_batch_stride,
                                      float scale, unsigned rows, unsigned cols) {
  unsigned int c = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (c < cols && r < rows) {
    const size_t dst_index = (dst_batch_stride * blockIdx.z) + (r * cols + c);
    const size_t src_index = (src_batch_stride * blockIdx.z) + ((c & 0xffffffe0) * rows + (r << 5) + (c & 0x1F));
    dst[dst_index] = scale * static_cast<float>(src[src_index]);
  }
}

void QOrderDequantizeCol32ToRow(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                                const int8_t* src, float* dst, float scale,
                                unsigned batch, unsigned rows, unsigned cols) {
  if (cols & 0x1f) {
    throw std::runtime_error("cols can not divede by 32");
  }
  dim3 threads(32, 32, 1);
  dim3 blocks(cols / 32, (rows + 31) / 32, batch);
  size_t stride = (size_t)rows * cols;
  QOrderDequantizeCol32ToFloatRowKernel<<<blocks, threads, 0, stream>>>(src, stride, dst, stride, scale, rows, cols);
}

/************************************************************************
 * Quantize Routines:
 *   - fp16/32 input, do no care order
 ************************************************************************/
// C++17 constexpr not supported, use below trick
template <typename T>
struct FloatVecSelector {};

template <>
struct FloatVecSelector<__half> {
  typedef __half4 FloatVecT;
};

template <>
struct FloatVecSelector<float> {
  typedef float2 FloatVecT;
};

// block size: 256, Lets EPB = 256 * ElementCount(FloatVecT) * ElementsPerThreads
// grid size: (N + BLOCK_SIZE * EPB - 1) / EPB
template <typename FloatVecT, unsigned ElementsPerThread = 4>
__global__ void
QOrderQuantizeKernel(const typename DequantizeVec<FloatVecT>::DequantizedScalarT* __restrict__ src,
                     int8_t* __restrict__ dst, size_t N,
                     const typename DequantizeVec<FloatVecT>::DequantizedScalarT inverse_scale) {
  typedef typename DequantizeVec<FloatVecT>::QuantizedVecT CharVecT;
  size_t index = (size_t)blockIdx.x * blockDim.x * (sizeof(CharVecT) * ElementsPerThread) + threadIdx.x * sizeof(CharVecT);
  unsigned inc_per_iter = blockDim.x * sizeof(CharVecT);
#pragma unroll
  for (int i = 0; i < ElementsPerThread; i++) {
    if (index < N) {
      FloatVecT src_vals = *(const FloatVecT*)(src + index);
      *(CharVecT*)(dst + index) = DequantizeVec<FloatVecT>::Quantize(src_vals, inverse_scale);
      index += inc_per_iter;
    }
  }
}

template <typename T>
void QOrderQuantize(cudaStream_t stream, const cudaDeviceProp& /* device_prop */,
                    const T* src, int8_t* dst, size_t N, T scale) {
  if (N & 0x1fLL) {
    throw std::runtime_error("N can not divide by 32!");
  }

  typedef typename FloatVecSelector<T>::FloatVecT FloatVecT;
  typedef typename DequantizeVec<FloatVecT>::QuantizedVecT QuantizedVecT;

  static constexpr unsigned int kElementsPerThread = 4;
  unsigned int threads = 256;
  unsigned int EPB = threads * sizeof(QuantizedVecT) * kElementsPerThread;
  T inverse_scale = (T)(1.0f / (float)scale);
  unsigned int blocks = (static_cast<unsigned int>(N) + (EPB - 1)) / EPB;
  QOrderQuantizeKernel<FloatVecT, kElementsPerThread><<<blocks, threads, 0, stream>>>(src, dst, N, inverse_scale);
}

template void QOrderQuantize<float>(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                    const float* src, int8_t* dst, size_t N, float scale);

template void QOrderQuantize<__half>(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                     const __half* src, int8_t* dst, size_t N, __half scale);

/************************************************************************
 * Dequantize Routines:
 *   - fp16/32 output, do no care order
 ************************************************************************/

// block size: 256, Lets EPB = 256 * ElementCount(FloatVecT) * ElementsPerThreads
// grid size: (N + BLOCK_SIZE * EPB - 1) / EPB
template <typename FloatVecT, unsigned ElementsPerThread = 4>
__global__ void
QOrderDequantizeKernel(const int8_t* __restrict__ src,
                       const typename DequantizeVec<FloatVecT>::DequantizedScalarT* __restrict__ dst,
                       size_t N,
                       const typename DequantizeVec<FloatVecT>::DequantizedScalarT scale) {
  typedef typename DequantizeVec<FloatVecT>::QuantizedVecT CharVecT;
  unsigned inc_per_iter = blockDim.x * sizeof(CharVecT);
  size_t index = (size_t)blockIdx.x * inc_per_iter * ElementsPerThread + threadIdx.x * sizeof(CharVecT);
#pragma unroll
  for (int i = 0; i < ElementsPerThread; i++) {
    if (index < N) {
      CharVecT src_vals = *(const CharVecT*)(src + index);
      *(FloatVecT*)(dst + index) = DequantizeVec<FloatVecT>::Dequantize(src_vals, scale);
      index += inc_per_iter;
    }
  }
}

template <typename T>
void QOrderDequantize(cudaStream_t stream, const cudaDeviceProp& /* device_prop */,
                      const int8_t* src, T* dst, size_t N, T scale) {
  if (N & 0x1fLL) {
    throw std::runtime_error("N can not divide by 32!");
  }

  typedef typename FloatVecSelector<T>::FloatVecT FloatVecT;
  typedef typename DequantizeVec<FloatVecT>::QuantizedVecT QuantizedVecT;
  static constexpr unsigned int kElementsPerThread = 2;

  unsigned int threads = 256;
  unsigned int EPB = threads * sizeof(QuantizedVecT) * kElementsPerThread;
  unsigned int blocks = (static_cast<unsigned int>(N) + (EPB - 1)) / EPB;
  QOrderDequantizeKernel<FloatVecT, kElementsPerThread><<<blocks, threads, 0, stream>>>(src, dst, N, scale);
}

template void QOrderDequantize<float>(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                      const int8_t* src, float* dst, size_t N, float scale);

template void QOrderDequantize<__half>(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                       const int8_t* src, __half* dst, size_t N, __half scale);

/************************************************************************
 * QOrdered Layernorm with compute type fp16:
 *   - input is int8 with order COL32 or ROW
 ************************************************************************/

static constexpr unsigned QORDER_LAYERNORM_ROWS_PER_BLOCK = 8;  // 4, 8, 16, ...
// block_size = (32, QORDER_LAYERNORM_ROWS_PER_BLOCK, 1)
// grid_size = ((rows + QORDER_LAYERNORM_ROWS_PER_BLOCK - 1) / QORDER_LAYERNORM_ROWS_PER_BLOCK, batch, 1)
__global__ void
QOrderLayerNormCol32Kernel(const int8_t* __restrict__ src, const float src_scale,
                           int8_t* __restrict__ dst, const float dst_scale,
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

// block_size = (32, QORDER_LAYERNORM_ROWS_PER_BLOCK, 1)
// grid_size = ((rows + QORDER_LAYERNORM_ROWS_PER_BLOCK - 1) / QORDER_LAYERNORM_ROWS_PER_BLOCK, batch, 1)
__global__ void
QOrderLayerNormRowKernel(const int8_t* __restrict__ src, const float src_scale,
                         int8_t* __restrict__ dst, const float dst_scale,
                         const __half* __restrict__ gamma, const __half* __restrict__ beta, const float epsilon,
                         const unsigned rows, const unsigned cols) {
  int32_t sum = 0;
  int32_t square_sum = 0;
  unsigned r = blockIdx.x * QORDER_LAYERNORM_ROWS_PER_BLOCK + threadIdx.y;
  if (rows <= r) return;

  const size_t batch_row_index = (size_t)blockIdx.y * (rows * cols) + r * cols;
  src += batch_row_index;
  dst += batch_row_index;
  for (unsigned c = threadIdx.x << 2; c < cols; c += 128) {
    char4 ch4 = __ldg((const char4*)(src + c));
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

  for (unsigned c = threadIdx.x << 2; c < cols; c += 128) {
    char4 ch4 = __ldg((const char4*)(src + c));
    __half4 dqval4 = deqantize_char4_half4(ch4, src_scale2);
    const __half4 g4 = *((const __half4*)(gamma + c));
    const __half4 b4 = (beta == nullptr) ? zero4 : *((const __half4*)(beta + c));
    dqval4.xy = __hfma2(__hmul2(__hsub2(dqval4.xy, mean2), var2), g4.xy, b4.xy);
    dqval4.zw = __hfma2(__hmul2(__hsub2(dqval4.zw, mean2), var2), g4.zw, b4.zw);
    *(char4*)(dst + c) = quantize_half4_char4(dqval4, dst_rscale2);
  }
}

template <typename T>
__device__ inline T Rsqrt(const T& x);

template <>
__device__ inline float Rsqrt(const float& x) {
  return rsqrtf(x);
}

template <>
__device__ inline half Rsqrt(const half& x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return hrsqrt(x);
#else
  return half(rsqrtf(float(x)));
#endif
}

__device__ inline half2 AddHalf2(const half2 a, const half2 b) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return __hadd2(a, b);
#else
  return __halves2half2(__hadd(a.x, b.x), __hadd(a.y, b.y));
#endif
}

struct KeyValuePairSum {
  __device__ inline cub::KeyValuePair<float, float> operator()(const cub::KeyValuePair<float, float>& a, const cub::KeyValuePair<float, float>& b) {
    return cub::KeyValuePair<float, float>(a.key + b.key, a.value + b.value);
  }

  __device__ inline cub::KeyValuePair<half, half> operator()(const cub::KeyValuePair<half, half>& a, const cub::KeyValuePair<half, half>& b) {
    const half2 a2 = __halves2half2(a.key, a.value);
    const half2 b2 = __halves2half2(b.key, b.value);
    const half2 res = AddHalf2(a2, b2);
    return cub::KeyValuePair<half, half>(res.x, res.y);
  }

  __device__ inline cub::KeyValuePair<half2, half2> operator()(const cub::KeyValuePair<half2, half2>& a, const cub::KeyValuePair<half2, half2>& b) {
    return cub::KeyValuePair<half2, half2>(AddHalf2(a.key, b.key), AddHalf2(a.value, b.value));
  }
};

template <unsigned TPB>
__global__ void QOrderAddResidualBiasLayerNormCol32Kernel(const int8_t* __restrict__ src, const float src_scale,
                                                          const int8_t* __restrict__ residual, const float residual_scale,
                                                          const __half* __restrict__ bias,
                                                          int8_t* __restrict__ dst, const float dst_scale,
                                                          const __half* __restrict__ gamma, const __half* __restrict__ beta, const float epsilon,
                                                          const unsigned rows, const unsigned cols, const unsigned tile_stride) {
  const int row_offset = (blockIdx.y * rows * cols) + (blockIdx.x << 5);

  const __half2 src_scale2 = __float2half2_rn(src_scale);
  const __half2 residual_scale2 = __float2half2_rn(residual_scale);
  const __half2 dst_scale2 = __float2half2_rn(1.0f / dst_scale);

  KeyValuePairSum pair_sum;

  const half2 rld = __float2half2_rn(1.f / cols);

  cub::KeyValuePair<half2, half2> thread_data(__float2half2_rn(0.f), __float2half2_rn(0.f));

  constexpr unsigned col_stride = (TPB << 1);

  for (unsigned c = (threadIdx.x << 1); c < cols; c += col_stride) {
    unsigned offset = (row_offset + ((c >> 5) * tile_stride) + (c & 31));
    const half* bias_offset = bias + c;

    const int8_t* process_src = src + offset;
    const int8_t* process_residual = residual + offset;

    half2 val = __hmul2(src_scale2, __half2(__short2half_rn(process_src[0]), __short2half_rn(process_src[1])));
    const half2 residual_val = __hmul2(residual_scale2, __half2(__short2half_rn(process_residual[0]), __short2half_rn(process_residual[1])));
    const half2 bias_val = {bias_offset[0], bias_offset[1]};
    val = __hadd2(val, __hadd2(residual_val, bias_val));

    const half2 rldval = __hmul2(rld, val);
    thread_data = pair_sum(thread_data, cub::KeyValuePair<half2, half2>(rldval, __hmul2(rldval, val)));
  }

  // Compute mean, std, and normalize here

  using BlockReduce = cub::BlockReduce<cub::KeyValuePair<half2, half2>, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  __shared__ half2 mu;
  __shared__ half2 rsigma;

  const auto sum_kv = BlockReduce(temp_storage).Reduce(thread_data, pair_sum);

  if (threadIdx.x == 0) {
    // Reduce across half2
    const half2 half2_mu_reduced = sum_kv.key;
    const half mu_half = __hadd(half2_mu_reduced.x, half2_mu_reduced.y);
    mu = __half2half2(mu_half);

    const half2 half2_sum_squared_reduced = sum_kv.value;
    const half sum_squared = __hadd(half2_sum_squared_reduced.x, half2_sum_squared_reduced.y);
    rsigma = __half2half2(Rsqrt(sum_squared - mu_half * mu_half + __float2half_rn(epsilon)));
  }

  __syncthreads();

  for (unsigned c = threadIdx.x << 1; c < cols; c += col_stride) {
    unsigned offset = (row_offset + ((c >> 5) * tile_stride) + (c & 31));

    const half* bias_offset = bias + c;

    const int8_t* process_src = src + offset;
    const int8_t* process_residual = residual + offset;

    half2 val = __hmul2(src_scale2, __half2(__short2half_rn(process_src[0]), __short2half_rn(process_src[1])));
    const half2 residual_val = __hmul2(residual_scale2, __half2(__short2half_rn(process_residual[0]), __short2half_rn(process_residual[1])));
    const half2 bias_val = {bias_offset[0], bias_offset[1]};
    val = __hadd2(val, __hadd2(residual_val, bias_val));

    const half* gamma_offset = gamma + c;
    const half2 gamma_val = {gamma_offset[0], gamma_offset[1]};

    const half* beta_offset = (nullptr == beta) ? nullptr : beta + c;
    half2 beta_val;
    if (nullptr == beta) {
      beta_val = __float2half2_rn(0.0f);
    } else {
      beta_val = {beta_offset[0], beta_offset[1]};
    }

    __half2 output_val = gamma_val * (val - mu) * rsigma + beta_val;
    output_val *= dst_scale2;

    U1S2 short_output_val;
    short_output_val.s2.x = __half2short_rn(output_val.x);
    short_output_val.s2.y = __half2short_rn(output_val.y);

    short_output_val.u1 = __vmaxs2(__vmins2(short_output_val.u1, 0x007F007F), 0xFF80FF80);

    int8_t* process_dst = dst + offset;

    process_dst[0] = static_cast<int8_t>(short_output_val.s2.x);
    process_dst[1] = static_cast<int8_t>(short_output_val.s2.y);
  }
}

void QOrderAddBiasResidualLayerNorm(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/, cublasLtOrder_t order,
                                    const int8_t* src, const float src_scale,
                                    const int8_t* residual, const float residual_scale,
                                    const __half* bias,
                                    int8_t* dst, const float dst_scale,
                                    const __half* gamma, const __half* beta, const float epsilon,
                                    const unsigned batch, const unsigned rows, const unsigned cols) {
  if (order == CUBLASLT_ORDER_COL32 && residual != nullptr && bias != nullptr) {
    constexpr int tpb = 128;
    const dim3 blocks(rows, batch, 1);
    const dim3 threads(tpb, 1, 1);
    QOrderAddResidualBiasLayerNormCol32Kernel<tpb><<<blocks, threads, 0, stream>>>(src, src_scale, residual,
                                                                                   residual_scale, bias,
                                                                                   dst, dst_scale, gamma,
                                                                                   beta, epsilon, rows, cols,
                                                                                   rows * 32);
  } else {  // No bias or residual
    dim3 threads(32, QORDER_LAYERNORM_ROWS_PER_BLOCK, 1);
    dim3 blocks((unsigned)(rows + QORDER_LAYERNORM_ROWS_PER_BLOCK - 1) / QORDER_LAYERNORM_ROWS_PER_BLOCK, (unsigned)batch, 1);
    if (order == CUBLASLT_ORDER_COL32) {
      QOrderLayerNormCol32Kernel<<<blocks, threads, 0, stream>>>(src, src_scale, dst, dst_scale,
                                                                 gamma, beta, epsilon, rows, cols);
    } else {  // order == CUBLASLT_ORDER_ROW
      QOrderLayerNormRowKernel<<<blocks, threads, 0, stream>>>(src, src_scale, dst, dst_scale,
                                                               gamma, beta, epsilon, rows, cols);
    }
  }
}

// source matrix block 32 x 32, each thread handle 4 int8_t items,
// thread block size should be (8 cols_in_4, 32 rows, 1)
// grid size ((cols + 31) / 32, (rows + 31) / 32), batch)
__global__ void
ReorderS8RowToCol32Kernel(const int8_t* __restrict__ src, int8_t* __restrict__ dst, unsigned rows, unsigned cols) {
  unsigned int c = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  unsigned int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (c < cols && r < rows) {
    const size_t batch_start = blockIdx.z * (rows * cols);
    const size_t src_index = batch_start + (r * cols + c);
    const size_t dst_index = batch_start + ((c & 0xffffffe0) * rows + (r << 5) + (c & 0x1f));
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
