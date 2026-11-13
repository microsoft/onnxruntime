// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_group_router_impl.h"

#include <limits>

#include "contrib_ops/cuda/quantization/gguf_quant_cuda.cuh"

namespace onnxruntime::contrib::cuda {
namespace {

template <typename CudaT, bool kAligned = false>
__global__ __launch_bounds__(128) void SigmoidScoresR8Kernel(
    const CudaT* __restrict__ input,
    const CudaT* __restrict__ weight,
    const float* __restrict__ bias,
    int width,
    int experts,
    float* __restrict__ scores,
    float* __restrict__ selection_scores) {
  const int expert = static_cast<int>(blockIdx.x);
  if (expert >= experts) return;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const CudaT* w = weight + static_cast<int64_t>(expert) * width;
  float sum[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
  for (int offset = (warp * 32 + lane) * 8; offset < width; offset += 1024) {
    if constexpr (sizeof(CudaT) == 2) {
      const uint4 wv = kAligned ? __ldg(reinterpret_cast<const uint4*>(w + offset)) : LoadPacked8(w + offset);
      const CudaT* wp = reinterpret_cast<const CudaT*>(&wv);
      float wf[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        wf[j] = gguf_cuda::CudaTypeToFloat<CudaT>(wp[j]);
      }
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const CudaT* xr = input + static_cast<int64_t>(r) * width + offset;
        const uint4 xv = kAligned ? __ldg(reinterpret_cast<const uint4*>(xr)) : LoadPacked8(xr);
        const CudaT* xp = reinterpret_cast<const CudaT*>(&xv);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          sum[r] += gguf_cuda::CudaTypeToFloat<CudaT>(xp[j]) * wf[j];
        }
      }
    } else {
      float wf[8];
#pragma unroll
      for (int j = 0; j < 8; ++j) wf[j] = __ldg(w + offset + j);
#pragma unroll
      for (int r = 0; r < 8; ++r) {
        const CudaT* xr = input + static_cast<int64_t>(r) * width + offset;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          sum[r] += __ldg(xr + j) * wf[j];
        }
      }
    }
  }
  __shared__ float s_acc[4][8];
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    for (int offset = 16; offset > 0; offset >>= 1) {
      sum[r] += __shfl_down_sync(0xffffffffu, sum[r], offset);
    }
    if (lane == 0) s_acc[warp][r] = sum[r];
  }
  __syncthreads();
  if (warp == 0 && lane < 8) {
    const float tot = (s_acc[0][lane] + s_acc[1][lane]) + (s_acc[2][lane] + s_acc[3][lane]);
    const float s = 1.0f / (1.0f + expf(-tot));
    const int64_t idx = static_cast<int64_t>(lane) * experts + expert;
    scores[idx] = s;
    selection_scores[idx] = s + bias[expert];
  }
}

template <typename CudaT>
__global__ void SigmoidScoresKernel(
    const CudaT* input, const CudaT* weight, const float* bias, int width, int experts,
    float* scores, float* selection_scores) {
  const int row = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int expert = blockIdx.y * 8 + warp;
  if (expert >= experts) return;
  const CudaT* x = input + static_cast<int64_t>(row) * width;
  const CudaT* w = weight + static_cast<int64_t>(expert) * width;
  float sum = 0.0f;
  for (int offset = lane * 8; offset < width; offset += 256) {
    if constexpr (sizeof(CudaT) == 2) {
      const uint4 xv = LoadPacked8(x + offset);
      const uint4 wv = LoadPacked8(w + offset);
      const CudaT* xp = reinterpret_cast<const CudaT*>(&xv);
      const CudaT* wp = reinterpret_cast<const CudaT*>(&wv);
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        sum += gguf_cuda::CudaTypeToFloat<CudaT>(xp[j]) * gguf_cuda::CudaTypeToFloat<CudaT>(wp[j]);
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        sum += x[offset + j] * w[offset + j];
      }
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xffffffffu, sum, offset);
  }
  if (lane == 0) {
    const float s = 1.0f / (1.0f + expf(-sum));
    const int64_t idx = static_cast<int64_t>(row) * experts + expert;
    scores[idx] = s;
    selection_scores[idx] = s + bias[expert];
  }
}

template <typename T>
__global__ void SigmoidLogitsKernel(
    const T* logits, const float* bias, int64_t count, int experts,
    float* scores, float* selection_scores) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    const float s = 1.0f / (1.0f + expf(-static_cast<float>(logits[i])));
    scores[i] = s;
    selection_scores[i] = s + bias[i % experts];
  }
}

}  // namespace

template <typename T>
Status LaunchFullMoEGroupRouter(
    cudaStream_t stream, int rows, int width, int experts,
    const T* input, const T* projection_weight, const float* bias,
    float* selection_scores, float* router_weights,
    cublasHandle_t cublas) {
  if (rows == 0) return Status::OK();
  const int64_t count = static_cast<int64_t>(rows) * experts;
  const int64_t blocks = (count + 255) / 256;
  ORT_RETURN_IF_NOT(blocks <= std::numeric_limits<int>::max(),
                    "MoEGroupRouter full-score routing exceeds the CUDA grid limit");
  if (projection_weight) {
    if (rows == 8) {
      if ((reinterpret_cast<uintptr_t>(input) & 15u) == 0 &&
          (reinterpret_cast<uintptr_t>(projection_weight) & 15u) == 0) {
        SigmoidScoresR8Kernel<T, true><<<experts, 128, 0, stream>>>(
            input, projection_weight, bias, width, experts, router_weights, selection_scores);
      } else {
        SigmoidScoresR8Kernel<T, false><<<experts, 128, 0, stream>>>(
            input, projection_weight, bias, width, experts, router_weights, selection_scores);
      }
    } else if (cublas != nullptr && rows > 16) {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      constexpr cudaDataType_t input_type =
          std::is_same_v<T, __nv_bfloat16> ? CUDA_R_16BF
          : std::is_same_v<T, half>        ? CUDA_R_16F
                                           : CUDA_R_32F;
      constexpr cublasComputeType_t compute_type =
          std::is_same_v<T, float> ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
      constexpr cublasGemmAlgo_t algo =
          std::is_same_v<T, float> ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
      CUBLAS_RETURN_IF_ERROR(cublasGemmEx(
          cublas, CUBLAS_OP_T, CUBLAS_OP_N,
          experts, rows, width,
          &alpha,
          projection_weight, input_type, width,
          input, input_type, width,
          &beta,
          router_weights, CUDA_R_32F, experts,
          compute_type, algo));
      SigmoidLogitsKernel<float><<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
          router_weights, bias, count, experts, router_weights, selection_scores);
    } else {
      const dim3 grid(rows, (experts + 7) / 8);
      SigmoidScoresKernel<T><<<grid, 256, 0, stream>>>(
          input, projection_weight, bias, width, experts, router_weights, selection_scores);
    }
  } else {
    SigmoidLogitsKernel<T><<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
        input, bias, count, experts, router_weights, selection_scores);
  }
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchFullMoEGroupRouter<float>(
    cudaStream_t, int, int, int, const float*, const float*, const float*, float*, float*, cublasHandle_t);
template Status LaunchFullMoEGroupRouter<half>(
    cudaStream_t, int, int, int, const half*, const half*, const float*, float*, float*, cublasHandle_t);
template Status LaunchFullMoEGroupRouter<__nv_bfloat16>(
    cudaStream_t, int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const float*, float*, float*, cublasHandle_t);

}  // namespace onnxruntime::contrib::cuda
