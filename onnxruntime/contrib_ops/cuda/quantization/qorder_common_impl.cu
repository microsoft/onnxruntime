// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cuda/quantization/qorder_unary_ops_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cub/cub.cuh>

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

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

__device__ inline float to_float(const __half h) { return __half2float(h); }
__device__ inline float to_float(const float f) { return f; }

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

__device__ inline char4 quantize_float4_char4(const float4 val4, const float rscale) {
  return char4{quantize_float_s8(val4.x, rscale), quantize_float_s8(val4.y, rscale),
               quantize_float_s8(val4.z, rscale), quantize_float_s8(val4.w, rscale)};
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

__device__ inline char4 quantize_half4_char4_strict_fp32(const __half4 val4, const float inverse_scale) {
  U1S2 shortxy, shortzw;
  shortxy.s2.x = static_cast<short>(__float2int_rn(__half2float(val4.xy.x) * inverse_scale));
  shortxy.s2.y = static_cast<short>(__float2int_rn(__half2float(val4.xy.y) * inverse_scale));
  shortzw.s2.x = static_cast<short>(__float2int_rn(__half2float(val4.zw.x) * inverse_scale));
  shortzw.s2.y = static_cast<short>(__float2int_rn(__half2float(val4.zw.y) * inverse_scale));
  shortxy.u1 = __vmaxs2(__vmins2(shortxy.u1, 0x007F007F), 0xFF80FF80);
  shortzw.u1 = __vmaxs2(__vmins2(shortzw.u1, 0x007F007F), 0xFF80FF80);
  return char4{(char)shortxy.s2.x, (char)shortxy.s2.y, (char)shortzw.s2.x, (char)shortzw.s2.y};
}

__device__ inline __half4 deqantize_char4_half4(const char4 ch4, const __half2 scale2) {
  return {scale2 * __half2(__short2half_rn(ch4.x), __short2half_rn(ch4.y)),
          scale2 * __half2(__short2half_rn(ch4.z), __short2half_rn(ch4.w))};
}

__device__ inline __half4 deqantize_char4_half4_strict(const char4 ch4, const float scale) {
  return __half4{{__float2half_rn(scale * ch4.x), __float2half_rn(scale * ch4.y)}, {__float2half_rn(scale * ch4.z), __float2half_rn(scale * ch4.w)}};
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

template <unsigned ElementsPerThread = 4>
__global__ void
QOrderQuantizeHalfStrictKernel(const __half* __restrict__ src, int8_t* __restrict__ dst, size_t N, const float inverse_scale) {
  unsigned inc_per_iter = blockDim.x * sizeof(char4);
  size_t index = (size_t)blockIdx.x * blockDim.x * (sizeof(char4) * ElementsPerThread) + threadIdx.x * sizeof(char4);

#pragma unroll
  for (int i = 0; i < ElementsPerThread; i++) {
    if (index < N) {
      __half4 src_vals = *(const __half4*)(src + index);
      *(char4*)(dst + index) = quantize_half4_char4_strict_fp32(src_vals, inverse_scale);
      index += inc_per_iter;
    }
  }
}

template <typename T>
void QOrderQuantize(cudaStream_t stream, const cudaDeviceProp& /* device_prop */,
                    const T* src, int8_t* dst, float scale, size_t N) {
  if (N & 0x3LL) {
    throw std::runtime_error("N can not divide by 4!");
  }

  typedef typename FloatVecSelector<T>::FloatVecT FloatVecT;
  typedef typename DequantizeVec<FloatVecT>::QuantizedVecT QuantizedVecT;

  static constexpr unsigned kElementsPerThread = 4;
  unsigned int threads = 256;
  unsigned int EPB = threads * sizeof(QuantizedVecT) * kElementsPerThread;
  T inverse_scale = (T)(1.0f / (scale));
  unsigned int blocks = gsl::narrow<unsigned int>((N + (EPB - 1)) / EPB);
  QOrderQuantizeKernel<FloatVecT, kElementsPerThread><<<blocks, threads, 0, stream>>>(src, dst, N, inverse_scale);
}

void QOrderQuantize_Strict(cudaStream_t stream, const cudaDeviceProp& /* device_prop*/,
                           const __half* src, int8_t* dst, float scale, size_t N) {
  if (N & 0x3LL) {
    throw std::runtime_error("N can not divide by 4!");
  }

  static constexpr unsigned kElementsPerThread = 4;
  unsigned int threads = 256;
  unsigned int EPB = threads * sizeof(char4) * kElementsPerThread;
  float inverse_scale = 1.0f / scale;
  unsigned int blocks = gsl::narrow<unsigned int>((N + (EPB - 1)) / EPB);
  QOrderQuantizeHalfStrictKernel<kElementsPerThread><<<blocks, threads, 0, stream>>>(src, dst, N, inverse_scale);
}

template void QOrderQuantize<float>(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                    const float* src, int8_t* dst, float scale, size_t N);

template void QOrderQuantize<__half>(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                     const __half* src, int8_t* dst, float scale, size_t N);

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
                      const int8_t* src, T* dst, float scale, size_t N) {
  if (N & 0x3LL) {
    throw std::runtime_error("N can not divide by 4!");
  }

  typedef typename FloatVecSelector<T>::FloatVecT FloatVecT;
  typedef typename DequantizeVec<FloatVecT>::QuantizedVecT QuantizedVecT;
  static constexpr unsigned kElementsPerThread = 2;

  unsigned int threads = 256;
  unsigned int EPB = threads * sizeof(QuantizedVecT) * kElementsPerThread;
  T scale_as_T = (T)(scale);
  unsigned int blocks = gsl::narrow<unsigned int>((N + (EPB - 1)) / EPB);
  QOrderDequantizeKernel<FloatVecT, kElementsPerThread><<<blocks, threads, 0, stream>>>(src, dst, N, scale_as_T);
}

// block size: 256, Lets EPB = 256 * ElementCount(FloatVecT) * ElementsPerThreads
// grid size: (N + BLOCK_SIZE * EPB - 1) / EPB
template <unsigned ElementsPerThread = 4>
__global__ void
QOrderDequantizeKernel_Strict(const int8_t* __restrict__ src, const __half* __restrict__ dst, size_t N, const float scale) {
  unsigned inc_per_iter = blockDim.x * sizeof(char4);
  size_t index = (size_t)blockIdx.x * inc_per_iter * ElementsPerThread + threadIdx.x * sizeof(char4);
#pragma unroll
  for (int i = 0; i < ElementsPerThread; i++) {
    if (index < N) {
      char4 src_vals = *(const char4*)(src + index);
      *(__half4*)(dst + index) = deqantize_char4_half4_strict(src_vals, scale);
      index += inc_per_iter;
    }
  }
}

void QOrderDequantize_Strict(cudaStream_t stream, const cudaDeviceProp& device_prop,
                             const int8_t* src, __half* dst, float scale, size_t N) {
  if (N & 0x3LL) {
    throw std::runtime_error("N can not divide by 4!");
  }

  static constexpr unsigned kElementsPerThread = 2;
  unsigned int threads = 256;
  unsigned int EPB = threads * sizeof(char4) * kElementsPerThread;
  unsigned int blocks = gsl::narrow<unsigned int>((N + (EPB - 1)) / EPB);
  QOrderDequantizeKernel_Strict<kElementsPerThread><<<blocks, threads, 0, stream>>>(src, dst, N, scale);
}

template void QOrderDequantize<float>(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                      const int8_t* src, float* dst, float scale, size_t N);

template void QOrderDequantize<__half>(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                       const int8_t* src, __half* dst, float scale, size_t N);

void QOrderDequantizeToRow(cublasLtOrder_t input_order, cudaStream_t stream, const cudaDeviceProp& device_prop,
                           const int8_t* src, __half* dst, float scale, unsigned batch, unsigned rows, unsigned cols) {
  if (input_order == CUBLASLT_ORDER_ROW) {
    QOrderDequantize_Strict(stream, device_prop, src, dst, scale, (size_t)batch * rows * cols);
  } else if (input_order == CUBLASLT_ORDER_COL32) {
    QOrderDequantizeCol32ToRow(stream, device_prop, src, dst, scale, batch, rows, cols);
  } else {
    throw std::runtime_error("Currently not supported!");
  }
}

void QOrderQuantizeRowTo(cublasLtOrder_t output_order, cudaStream_t stream, const cudaDeviceProp& device_prop,
                         const __half* src, int8_t* dst, float scale, unsigned batch, unsigned rows, unsigned cols) {
  if (output_order == CUBLASLT_ORDER_ROW) {
    QOrderQuantize_Strict(stream, device_prop, src, dst, scale, (size_t)batch * rows * cols);
  } else if (output_order == CUBLASLT_ORDER_COL32) {
    QOrderQuantizeRowToCol32(stream, device_prop, src, dst, scale, batch, rows, cols);
  } else {
    throw std::runtime_error("Currently not supported!");
  }
}

/************************************************************************
 * QOrdered Layernorm with compute type fp16:
 *   - input is int8 with order COL32 or ROW
 ************************************************************************/

static constexpr unsigned QORDER_LAYERNORM_ROWS_PER_BLOCK = 8;  // 4, 8, 16, ...
// block_size = (32, QORDER_LAYERNORM_ROWS_PER_BLOCK, 1)
// grid_size = ((rows + QORDER_LAYERNORM_ROWS_PER_BLOCK - 1) / QORDER_LAYERNORM_ROWS_PER_BLOCK, batch, 1)
template <typename T>
__global__ void
QOrderLayerNormCol32Kernel(const int8_t* __restrict__ src, const float src_scale, int8_t* __restrict__ dst, const float dst_scale,
                           const T* __restrict__ gamma, const T* __restrict__ beta, const float epsilon,
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

  const float mean = src_scale * (float)sum / cols;
  const float rvar = rsqrtf(src_scale * src_scale * ((double)square_sum - ((double)sum * sum / cols)) / cols + epsilon);
  const float dst_rscale = 1.0f / dst_scale;
  float4 f4;
  for (unsigned index = 0, c = threadIdx.x * 4; c < cols; c += 128, index += STRIDES_PER_WARP_ROUND) {
    char4 ch4 = __ldg((const char4*)(src + index));
    f4.x = (src_scale * ch4.x - mean) * rvar * to_float(gamma[c]);
    f4.y = (src_scale * ch4.y - mean) * rvar * to_float(gamma[c + 1]);
    f4.z = (src_scale * ch4.z - mean) * rvar * to_float(gamma[c + 2]);
    f4.w = (src_scale * ch4.w - mean) * rvar * to_float(gamma[c + 3]);
    if (beta) {
      f4.x += to_float(beta[c]);
      f4.y += to_float(beta[c + 1]);
      f4.z += to_float(beta[c + 2]);
      f4.w += to_float(beta[c + 3]);
    }
    *(char4*)(dst + index) = quantize_float4_char4(f4, dst_rscale);
  }
}

// block_size = (32, QORDER_LAYERNORM_ROWS_PER_BLOCK, 1)
// grid_size = ((rows + QORDER_LAYERNORM_ROWS_PER_BLOCK - 1) / QORDER_LAYERNORM_ROWS_PER_BLOCK, batch, 1)
template <typename T>
__global__ void
QOrderLayerNormRowKernel(const int8_t* __restrict__ src, const float src_scale, int8_t* __restrict__ dst, const float dst_scale,
                         const T* __restrict__ gamma, const T* __restrict__ beta, const float epsilon,
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

  const float mean = __double2float_rn(src_scale * (double)sum / cols);
  const float rvar = rsqrtf(src_scale * src_scale * __double2float_rn((double)square_sum - ((double)sum * (double)sum / (double)cols)) / cols + epsilon);
  const float dst_rscale = 1.0f / dst_scale;
  float4 f4;
  for (unsigned c = threadIdx.x << 2; c < cols; c += 128) {
    char4 ch4 = __ldg((const char4*)(src + c));
    f4.x = (src_scale * ch4.x - mean) * rvar * to_float(gamma[c]);
    f4.y = (src_scale * ch4.y - mean) * rvar * to_float(gamma[c + 1]);
    f4.z = (src_scale * ch4.z - mean) * rvar * to_float(gamma[c + 2]);
    f4.w = (src_scale * ch4.w - mean) * rvar * to_float(gamma[c + 3]);
    if (beta) {
      f4.x += to_float(beta[c]);
      f4.y += to_float(beta[c + 1]);
      f4.z += to_float(beta[c + 2]);
      f4.w += to_float(beta[c + 3]);
    }
    *(char4*)(dst + c) = quantize_float4_char4(f4, dst_rscale);
  }
}

template <typename T>
void QOrderLayerNorm(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/, cublasLtOrder_t order,
                     const int8_t* src, const float src_scale, int8_t* dst, const float dst_scale,
                     const T* gamma, const T* beta, const float epsilon,
                     const unsigned batch, const unsigned rows, const unsigned cols) {
  if (cols & (order == CUBLASLT_ORDER_COL32 ? 0x1FLL : 0x3LL)) {
    throw std::runtime_error("cols can not divide by 4 in ROW order or 32 in COL32 order!");
  }

  dim3 threads(32, QORDER_LAYERNORM_ROWS_PER_BLOCK, 1);
  dim3 blocks((unsigned)(rows + QORDER_LAYERNORM_ROWS_PER_BLOCK - 1) / QORDER_LAYERNORM_ROWS_PER_BLOCK, (unsigned)batch, 1);
  if (order == CUBLASLT_ORDER_COL32) {
    QOrderLayerNormCol32Kernel<T><<<blocks, threads, 0, stream>>>(src, src_scale, dst, dst_scale, gamma, beta, epsilon, rows, cols);
  } else {  // order == CUBLASLT_ORDER_ROW
    QOrderLayerNormRowKernel<T><<<blocks, threads, 0, stream>>>(src, src_scale, dst, dst_scale, gamma, beta, epsilon, rows, cols);
  }
}

template void QOrderLayerNorm<float>(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/, cublasLtOrder_t order,
                                     const int8_t* src, const float src_scale, int8_t* dst, const float dst_scale,
                                     const float* gamma, const float* beta, const float epsilon,
                                     const unsigned batch, const unsigned rows, const unsigned cols);

template void QOrderLayerNorm<__half>(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/, cublasLtOrder_t order,
                                      const int8_t* src, const float src_scale, int8_t* dst, const float dst_scale,
                                      const __half* gamma, const __half* beta, const float epsilon,
                                      const unsigned batch, const unsigned rows, const unsigned cols);

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

__global__ void
BuildTableForSoftmaxPowerOfKernel(const float base, float* table) {
  int x = threadIdx.x - 255;
  table[x] = powf(base, x);
}

void BuildTableForSoftmaxPowerOf(cudaStream_t stream, const float base, float* table) {
  BuildTableForSoftmaxPowerOfKernel<<<1, 256, 0, stream>>>(base, table);
}

template <int TPB>
__global__ void
QOrderMaskedSoftmaxKernel(const int8_t* src, const float* lookup_table, const int32_t* mask_index,
                          int8_t* dst, const float scale_dst, const unsigned sequence_len) {
  using BlockReduceInt32 = cub::BlockReduce<int32_t, TPB>;
  using BlockReduceFP32 = cub::BlockReduce<float, TPB>;

  __shared__ typename BlockReduceInt32::TempStorage tmp_storage_int32;
  __shared__ typename BlockReduceFP32::TempStorage tmp_storage_fp32;
  __shared__ float sum_reverse_block;
  __shared__ int32_t max_in_block;

  int offset = (blockIdx.y * gridDim.x + blockIdx.x) * sequence_len; /* 4 bytes per thread */
  src += offset;
  dst += offset;
  mask_index += (blockIdx.y * sequence_len); // to to the batch
  for (offset = threadIdx.x * 4; offset < sequence_len; offset += TPB*4) {
    char4 ch4 = *(const char4*)(src + offset);
    int32_t max_of_4 = max(max((int)ch4.x, (int)ch4.y), max((int)ch4.z, (int)ch4.w));

    const int32_t max_all = BlockReduceInt32(tmp_storage_int32).Reduce(max_of_4, cub::Max());
    if (threadIdx.x == 0) {
      max_in_block = max_all;
    }
    const int4 mask_of_4 = *(const int4*)(mask_index + offset);
    __syncthreads();
    // TODO: bank conflick
    float4 epow_of_4 = { mask_of_4.x ? lookup_table[255 - max_in_block + (int)ch4.x] : 0.0f,
                         mask_of_4.y ? lookup_table[255 - max_in_block + (int)ch4.y] : 0.0f,
                         mask_of_4.z ? lookup_table[255 - max_in_block + (int)ch4.z] : 0.0f,
                         mask_of_4.w ? lookup_table[255 - max_in_block + (int)ch4.w] : 0.0f};
    float sum_of_4 = epow_of_4.x + epow_of_4.y + epow_of_4.z + epow_of_4.w;
    const float sum_all = BlockReduceFP32(tmp_storage_fp32).Reduce(sum_of_4, cub::Sum());
    if (threadIdx.x == 0) {
      sum_reverse_block = (float)(1.0 / ((double)sum_all * scale_dst));
    }
    __syncthreads();

    ch4.x = epow_of_4.x * sum_reverse_block;
    ch4.y = epow_of_4.y * sum_reverse_block;
    ch4.z = epow_of_4.z * sum_reverse_block;
    ch4.w = epow_of_4.w * sum_reverse_block;

    *(char4 *)(dst + offset) = ch4;
  }
}

void QOrderMaskedSoftmax(
    cudaStream_t stream, const cudaDeviceProp& device_prop,
    const int8_t* src, const float* lookup_table,
    const int32_t* mask_index,
    int8_t* dst, const float scale_dst,
    const unsigned batch, const unsigned num_heads, const unsigned sequence_len) {
  dim3 threads(256, 1, 1);
  dim3 blocks(sequence_len * num_heads, batch, 1);
  QOrderMaskedSoftmaxKernel<256><<<blocks, threads, 0, stream>>>(src, lookup_table, mask_index, dst, scale_dst, sequence_len);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
