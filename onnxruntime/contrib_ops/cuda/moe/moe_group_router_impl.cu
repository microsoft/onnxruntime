// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_group_router_impl.h"

#include <limits>

#include "contrib_ops/cuda/quantization/gguf_quant_cuda.cuh"

namespace onnxruntime::contrib::cuda {
namespace {

template <typename CudaT>
__global__ void SigmoidScoresKernel(
    const CudaT* input, const CudaT* weight, int width, int experts, float* scores) {
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
  if (lane == 0) scores[static_cast<int64_t>(row) * experts + expert] = 1.0f / (1.0f + expf(-sum));
}

template <typename T>
__global__ void SigmoidLogitsKernel(const T* logits, int64_t count, float* scores) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) scores[i] = 1.0f / (1.0f + expf(-static_cast<float>(logits[i])));
}

__global__ void AddSelectionBiasKernel(
    const float* router_weights, const float* bias, int64_t count, int experts, float* selection_scores) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) selection_scores[i] = router_weights[i] + bias[i % experts];
}

}  // namespace

template <typename T>
Status LaunchFullMoEGroupRouter(
    cudaStream_t stream, int rows, int width, int experts,
    const T* input, const T* projection_weight, const float* bias,
    float* selection_scores, float* router_weights) {
  if (rows == 0) return Status::OK();
  const int64_t count = static_cast<int64_t>(rows) * experts;
  const int64_t blocks = (count + 255) / 256;
  ORT_RETURN_IF_NOT(blocks <= std::numeric_limits<int>::max(),
                    "MoEGroupRouter full-score routing exceeds the CUDA grid limit");
  if (projection_weight) {
    const dim3 grid(rows, (experts + 7) / 8);
    SigmoidScoresKernel<T><<<grid, 256, 0, stream>>>(input, projection_weight, width, experts, router_weights);
  } else {
    SigmoidLogitsKernel<T><<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
        input, count, router_weights);
  }
  ORT_RETURN_IF_ERROR(CUDA_CALL(cudaGetLastError()));
  AddSelectionBiasKernel<<<static_cast<unsigned int>(blocks), 256, 0, stream>>>(
      router_weights, bias, count, experts, selection_scores);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchFullMoEGroupRouter<float>(
    cudaStream_t, int, int, int, const float*, const float*, const float*, float*, float*);
template Status LaunchFullMoEGroupRouter<half>(
    cudaStream_t, int, int, int, const half*, const half*, const float*, float*, float*);
template Status LaunchFullMoEGroupRouter<__nv_bfloat16>(
    cudaStream_t, int, int, int, const __nv_bfloat16*, const __nv_bfloat16*, const float*, float*, float*);

}  // namespace onnxruntime::contrib::cuda
