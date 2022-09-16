// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/qordered_ops/qordered_layer_norm_impl.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_common.cuh"

namespace onnxruntime {
namespace contrib {
namespace cuda {

static __device__ inline float ToFloat(const __half h) { return __half2float(h); }

static __device__ inline float ToFloat(const float f) { return f; }

constexpr static unsigned QORDER_LAYERNORM_ROWS_PER_BLOCK = 8;

template <typename T>
__global__ void QOrderedLayerNormRowKernel(const int8_t* __restrict__ src, const float src_scale,
                                           int8_t* __restrict__ dst, const float dst_scale,
                                           const T* __restrict__ gamma, const T* __restrict__ beta, const float epsilon,
                                           const unsigned rows, const unsigned cols) {
  int32_t sum = 0;
  int32_t square_sum = 0;

  unsigned r = blockIdx.x * QORDER_LAYERNORM_ROWS_PER_BLOCK + threadIdx.y;

  if (rows <= r) {
    return;
  }

  const size_t batch_row_index = static_cast<size_t>(blockIdx.y) * (rows * cols) + r * cols;
  src += batch_row_index;
  dst += batch_row_index;
  for (unsigned c = threadIdx.x << 2; c < cols; c += 128) {
    char4 ch4 = __ldg(reinterpret_cast<const char4*>(src + c));
    sum += (static_cast<short>(ch4.x) + static_cast<short>(ch4.y) +
            static_cast<short>(ch4.z) + static_cast<short>(ch4.w));
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610
    square_sum = __dp4a(ch4, ch4, square_sum);
#else
    square_sum = Dp4a_Defined(ch4, ch4);
#endif
  }

  sum = WarpReduceSum<int32_t>(sum);
  square_sum = WarpReduceSum<int32_t>(square_sum);

  const float mean = __double2float_rn(src_scale * (double)sum / cols);

  const float rvar = rsqrtf(src_scale * src_scale * __double2float_rn(static_cast<double>(square_sum) - (static_cast<double>(sum) * static_cast<double>(sum) / static_cast<double>(cols))) / cols + epsilon);

  const float dst_rscale = 1.0f / dst_scale;

  float4 f4;
  for (unsigned c = threadIdx.x << 2; c < cols; c += 128) {
    char4 ch4 = __ldg(reinterpret_cast<const char4*>(src + c));

    f4.x = (src_scale * ch4.x - mean) * rvar * ToFloat(gamma[c]);
    f4.y = (src_scale * ch4.y - mean) * rvar * ToFloat(gamma[c + 1]);
    f4.z = (src_scale * ch4.z - mean) * rvar * ToFloat(gamma[c + 2]);
    f4.w = (src_scale * ch4.w - mean) * rvar * ToFloat(gamma[c + 3]);

    if (beta) {
      f4.x += ToFloat(beta[c]);
      f4.y += ToFloat(beta[c + 1]);
      f4.z += ToFloat(beta[c + 2]);
      f4.w += ToFloat(beta[c + 3]);
    }

    *reinterpret_cast<char4*>(dst + c) = QuantizeFloat4Char4(f4, dst_rscale);
  }
}

template <typename T>
Status QOrderedLayerNorm(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/, cublasLtOrder_t order,
                       const int8_t* src, const float src_scale, int8_t* dst, const float dst_scale,
                       const T* gamma, const T* beta, const float epsilon,
                       const unsigned batch, const unsigned rows, const unsigned cols) {
  // The implementation only supports Row major tensor data ordering for now
  ORT_RETURN_IF(order != CUBLASLT_ORDER_ROW, "Order current not supported!");

  dim3 threads(32, QORDER_LAYERNORM_ROWS_PER_BLOCK, 1);

  dim3 blocks(static_cast<unsigned>(rows + QORDER_LAYERNORM_ROWS_PER_BLOCK - 1) / QORDER_LAYERNORM_ROWS_PER_BLOCK,
              static_cast<unsigned>(batch), 1);

  QOrderedLayerNormRowKernel<T><<<blocks, threads, 0, stream>>>(
      src, src_scale, dst, dst_scale, gamma, beta, epsilon, rows, cols);

  return CUDA_CALL(cudaGetLastError());  
}

template Status QOrderedLayerNorm<float>(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/, cublasLtOrder_t order,
                                       const int8_t* src, const float src_scale, int8_t* dst, const float dst_scale,
                                       const float* gamma, const float* beta, const float epsilon,
                                       const unsigned batch, const unsigned rows, const unsigned cols);

template Status QOrderedLayerNorm<__half>(cudaStream_t stream, const cudaDeviceProp& /*device_prop*/, cublasLtOrder_t order,
                                        const int8_t* src, const float src_scale, int8_t* dst, const float dst_scale,
                                        const __half* gamma, const __half* beta, const float epsilon,
                                        const unsigned batch, const unsigned rows, const unsigned cols);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
