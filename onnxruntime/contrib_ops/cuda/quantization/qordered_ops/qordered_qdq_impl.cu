// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_qdq_impl.h"

#include <cub/cub.cuh>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

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
    return char2{static_cast<signed char>(__float2int_rn(dqvalx)), static_cast<signed char>(__float2int_rn(dqvaly))};
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
    return char2{(signed char)s2xy.s2.x, (signed char)s2xy.s2.y};
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
    return QuantizeHalf4Char4(fpvals, __half2half2(inv_scales));
  }

  static __device__ inline __half4 Dequantize(const QuantizedVecT qvals, const __half scale) {
    return __half4{__half2{scale * __short2half_rn(qvals.x), scale * __short2half_rn(qvals.y)},
                   __half2{scale * __short2half_rn(qvals.z), scale * __short2half_rn(qvals.w)}};
  }
};

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
    *(char4*)(dst + dst_index) = QuantizeHalf4Char4(src_val4, inverse_scale2);
  }
}

// cols could be divide by 32
Status QOrderQuantizeRowToCol32(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                                const __half* src, int8_t* dst, float scale,
                                unsigned batch, unsigned rows, unsigned cols) {
  ORT_RETURN_IF(cols & 0x1f, "cols can not divide by 32!");

  __half2 inverse_scale2 = __float2half2_rn(1.0f / scale);
  dim3 threads(8, 32, 1);
  dim3 blocks(cols / 32, (rows + 31) / 32, batch);
  size_t stride = (size_t)rows * cols;
  QOrderQuantizeHalfRowToCol32Kernel<<<blocks, threads, 0, stream>>>(src, stride, dst, stride, inverse_scale2, rows, cols);
  return CUDA_CALL(cudaGetLastError());
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
        *(dst + dst_index) = QuantizeFloatS8(*(src + src_index), inverse_scale);
        c += kColsPerIncrement;
        src_index += kColsPerIncrement;
        dst_index += rows * kColsPerIncrement;
      }
    }
  }
}

// cols could be divide by 32
Status QOrderQuantizeRowToCol32(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                                const float* src, int8_t* dst, float scale,
                                unsigned batch, unsigned rows, unsigned cols) {
  ORT_RETURN_IF(cols & 0x1f, "cols can not divide by 32!");

  constexpr unsigned kElementsPerThread = 4;
  float inverse_scale = 1.0f / scale;
  dim3 threads(32, 32, 1);
  dim3 blocks((cols + (32 * kElementsPerThread - 1)) / (kElementsPerThread * 32), (rows + 31) / 32, batch);
  size_t stride = (size_t)rows * cols;
  QOrderQuantizeFloatRowToCol32Kernel<<<blocks, threads, 0, stream>>>(src, stride, dst, stride, inverse_scale, rows, cols);
  return CUDA_CALL(cudaGetLastError());
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
    *(__half4*)(dst + dst_index) = DeqantizeChar4Half4(src_ch4, scale2);
  }
}

// cols could be divide by 32
Status QOrderDequantizeCol32ToRow(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                                  const int8_t* src, __half* dst, float scale,
                                  unsigned batch, unsigned rows, unsigned cols) {
  ORT_RETURN_IF(cols & 0x1f, "cols can not divide by 32!");

  __half2 scale2 = __float2half2_rn(scale);
  dim3 threads(8, 32, 1);
  dim3 blocks(cols / 32, (rows + 31) / 32, batch);
  size_t stride = (size_t)rows * cols;
  QOrderDequantizeCol32ToHalfRowKernel<<<blocks, threads, 0, stream>>>(src, stride, dst, stride, scale2, rows, cols);
  return CUDA_CALL(cudaGetLastError());
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

Status QOrderDequantizeCol32ToRow(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                                  const int8_t* src, float* dst, float scale,
                                  unsigned batch, unsigned rows, unsigned cols) {
  ORT_RETURN_IF(cols & 0x1f, "cols can not divide by 32!");

  dim3 threads(32, 32, 1);
  dim3 blocks(cols / 32, (rows + 31) / 32, batch);
  size_t stride = (size_t)rows * cols;
  QOrderDequantizeCol32ToFloatRowKernel<<<blocks, threads, 0, stream>>>(src, stride, dst, stride, scale, rows, cols);
  return CUDA_CALL(cudaGetLastError());
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
      *(char4*)(dst + index) = QuantizeHalf4Char4Strict(src_vals, inverse_scale);
      index += inc_per_iter;
    }
  }
}

template <typename T>
Status QOrderQuantize(cudaStream_t stream, const cudaDeviceProp& /* device_prop */,
                      const T* src, int8_t* dst, float scale, size_t N) {
  ORT_RETURN_IF(N & 0x3LL, "N can not divide by 4!");

  typedef typename FloatVecSelector<T>::FloatVecT FloatVecT;
  typedef typename DequantizeVec<FloatVecT>::QuantizedVecT QuantizedVecT;

  static constexpr unsigned kElementsPerThread = 4;
  unsigned int threads = 256;
  unsigned int EPB = threads * sizeof(QuantizedVecT) * kElementsPerThread;
  T inverse_scale = (T)(1.0f / (scale));
  unsigned int blocks = (unsigned int)((N + (EPB - 1)) / EPB);
  QOrderQuantizeKernel<FloatVecT, kElementsPerThread><<<blocks, threads, 0, stream>>>(src, dst, N, inverse_scale);
  return CUDA_CALL(cudaGetLastError());
}

Status QOrderQuantize_Strict(cudaStream_t stream, const cudaDeviceProp& /* device_prop*/,
                             const __half* src, int8_t* dst, float scale, size_t N) {
  ORT_RETURN_IF(N & 0x3LL, "N can not divide by 4!");

  static constexpr unsigned kElementsPerThread = 4;
  unsigned int threads = 256;
  unsigned int EPB = threads * sizeof(char4) * kElementsPerThread;
  float inverse_scale = 1.0f / scale;
  unsigned int blocks = (unsigned int)((N + (EPB - 1)) / EPB);
  QOrderQuantizeHalfStrictKernel<kElementsPerThread><<<blocks, threads, 0, stream>>>(src, dst, N, inverse_scale);
  return CUDA_CALL(cudaGetLastError());
}

template Status QOrderQuantize<float>(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                      const float* src, int8_t* dst, float scale, size_t N);

template Status QOrderQuantize<__half>(cudaStream_t stream, const cudaDeviceProp& device_prop,
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
Status QOrderDequantize(cudaStream_t stream, const cudaDeviceProp& /* device_prop */,
                        const int8_t* src, T* dst, float scale, size_t N) {
  ORT_RETURN_IF(N & 0x3LL, "N can not divide by 4!");

  typedef typename FloatVecSelector<T>::FloatVecT FloatVecT;
  typedef typename DequantizeVec<FloatVecT>::QuantizedVecT QuantizedVecT;
  static constexpr unsigned kElementsPerThread = 2;

  unsigned int threads = 256;
  unsigned int EPB = threads * sizeof(QuantizedVecT) * kElementsPerThread;
  T scale_as_T = (T)(scale);
  unsigned int blocks = (unsigned int)((N + (EPB - 1)) / EPB);
  QOrderDequantizeKernel<FloatVecT, kElementsPerThread><<<blocks, threads, 0, stream>>>(src, dst, N, scale_as_T);
  return CUDA_CALL(cudaGetLastError());
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
      *(__half4*)(dst + index) = DeqantizeChar4Half4Strict(src_vals, scale);
      index += inc_per_iter;
    }
  }
}

Status QOrderDequantize_Strict(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/,
                               const int8_t* src, __half* dst, float scale, size_t N) {
  ORT_RETURN_IF(N & 0x3LL, "N can not divide by 4!");

  static constexpr unsigned kElementsPerThread = 2;
  unsigned int threads = 256;
  unsigned int EPB = threads * sizeof(char4) * kElementsPerThread;
  unsigned int blocks = (unsigned int)((N + (EPB - 1)) / EPB);
  QOrderDequantizeKernel_Strict<kElementsPerThread><<<blocks, threads, 0, stream>>>(src, dst, N, scale);
  return CUDA_CALL(cudaGetLastError());
}

template Status QOrderDequantize<float>(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                        const int8_t* src, float* dst, float scale, size_t N);

template Status QOrderDequantize<__half>(cudaStream_t stream, const cudaDeviceProp& device_prop,
                                         const int8_t* src, __half* dst, float scale, size_t N);

Status QOrderDequantizeToRow(cublasLtOrder_t input_order, cudaStream_t stream, const cudaDeviceProp& device_prop,
                             const int8_t* src, __half* dst, float scale, unsigned batch, unsigned rows, unsigned cols) {
  ORT_RETURN_IF((input_order != CUBLASLT_ORDER_ROW) && (input_order != CUBLASLT_ORDER_COL32), "Order currently not supported!");

  if (input_order == CUBLASLT_ORDER_ROW) {
    return QOrderDequantize_Strict(stream, device_prop, src, dst, scale, (size_t)batch * rows * cols);
  } else {  // if (input_order == CUBLASLT_ORDER_COL32) {
    return QOrderDequantizeCol32ToRow(stream, device_prop, src, dst, scale, batch, rows, cols);
  }
}

Status QOrderQuantizeRowTo(cublasLtOrder_t output_order, cudaStream_t stream, const cudaDeviceProp& device_prop,
                           const __half* src, int8_t* dst, float scale, unsigned batch, unsigned rows, unsigned cols) {
  ORT_RETURN_IF((output_order != CUBLASLT_ORDER_ROW) && (output_order != CUBLASLT_ORDER_COL32), "Order currently not supported!");

  if (output_order == CUBLASLT_ORDER_ROW) {
    return QOrderQuantize_Strict(stream, device_prop, src, dst, scale, (size_t)batch * rows * cols);
  } else {  // if (output_order == CUBLASLT_ORDER_COL32) {
    return QOrderQuantizeRowToCol32(stream, device_prop, src, dst, scale, batch, rows, cols);
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

Status ReorderS8RowToCol32(cudaStream_t stream, const cudaDeviceProp& /* device_prop */,
                           const int8_t* src, int8_t* dst,
                           unsigned batch, unsigned rows, unsigned cols) {
  dim3 threads(8, 32, 1);
  dim3 blocks((unsigned)(cols / 32), (unsigned)((rows + 31) / 32), batch);
  ReorderS8RowToCol32Kernel<<<blocks, threads, 0, stream>>>(src, dst, rows, cols);
  return CUDA_CALL(cudaGetLastError());
}

Status Reorder(cublasLtHandle_t cublasLt, cudaStream_t stream, const cudaDeviceProp& device_prop,
               int32_t batchCount, int64_t rows, int64_t cols, cudaDataType_t data_type,
               const void* input, cublasLtOrder_t order_input, void* output, cublasLtOrder_t order_output) {
  if (data_type == CUDA_R_8I && order_input == CUBLASLT_ORDER_ROW && order_output == CUBLASLT_ORDER_COL32) {
    return ReorderS8RowToCol32(stream, device_prop, (const int8_t*)input, (int8_t*)output,
                               (unsigned)batchCount, static_cast<unsigned>(rows), static_cast<unsigned>(cols));
  }

  cublasLtMatrixTransformDesc_t transform_desc = nullptr;
  auto clean_transform_desc = gsl::finally([&transform_desc]() {if (transform_desc) cublasLtMatrixTransformDescDestroy(transform_desc); });
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixTransformDescCreate(&transform_desc, CUDA_R_32I));

  cublasLtMatrixLayout_t InputLayout = nullptr;
  auto clean_InputLayout = gsl::finally([&InputLayout]() {if (InputLayout) cublasLtMatrixLayoutDestroy(InputLayout); });
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&InputLayout, data_type, rows, cols, CalcLeadingDimensionLt(rows, cols, order_input)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(InputLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_input, sizeof(order_input)));

  cublasLtMatrixLayout_t OutputLayout = nullptr;
  auto clean_OutputLayout = gsl::finally([&OutputLayout]() {if (OutputLayout) cublasLtMatrixLayoutDestroy(OutputLayout); });
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutCreate(&OutputLayout, data_type, rows, cols, CalcLeadingDimensionLt(rows, cols, order_output)));
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(OutputLayout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_output, sizeof(order_output)));

  if (batchCount > 1) {
    int64_t batch_stride_input = rows * cols;
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(InputLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(InputLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_input, sizeof(batch_stride_input)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(OutputLayout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batchCount, sizeof(batchCount)));
    CUBLAS_RETURN_IF_ERROR(cublasLtMatrixLayoutSetAttribute(OutputLayout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &batch_stride_input, sizeof(batch_stride_input)));
  }

  int32_t alpha = 1;
  int32_t beta = 0;
  CUBLAS_RETURN_IF_ERROR(cublasLtMatrixTransform(cublasLt, transform_desc, &alpha, input, InputLayout,
                                                 &beta, nullptr, nullptr, output, OutputLayout, stream));

  return Status::OK();
};

int64_t CalcLeadingDimensionLt(int64_t rows, int64_t cols, cublasLtOrder_t order) {
  switch (order) {
    case CUBLASLT_ORDER_ROW:
      return cols;
    case CUBLASLT_ORDER_COL:
      return rows;
    case CUBLASLT_ORDER_COL32:
      return 32 * rows;
    case CUBLASLT_ORDER_COL4_4R2_8C:
      return 32 * ((rows + 8 - 1) / 8) * 8;
    case CUBLASLT_ORDER_COL32_2R_4R4:
      return 32 * ((rows + 32 - 1) / 32) * 32;
    default:
      return 0;
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
