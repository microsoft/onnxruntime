// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/moe/moe_kernels.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {
namespace {

__global__ void RemapMoeExpertIndicesKernel(const int* expert_indices, int* remapped_expert_indices,
                                            const int* expert_map, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    const int expert = expert_indices[index];
    remapped_expert_indices[index] = expert >= 0 ? expert_map[expert] : -1;
  }
}

__global__ void AddMoeFp16OutputKernel(half* output, const half* contribution, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __hadd(output[index], contribution[index]);
  }
}

__global__ void AddMoeBf16OutputKernel(__nv_bfloat16* output, const __nv_bfloat16* contribution, size_t count) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __float2bfloat16(__bfloat162float(output[index]) + __bfloat162float(contribution[index]));
  }
}

}  // namespace

void LaunchRemapMoeExpertIndices(const int* expert_indices, int* remapped_expert_indices,
                                 const int* expert_map, size_t count, cudaStream_t stream) {
  constexpr int threads = 256;
  RemapMoeExpertIndicesKernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0, stream>>>(
      expert_indices, remapped_expert_indices, expert_map, count);
  CUDA_CALL_THROW(cudaGetLastError());
}

void LaunchAddMoeFp16Output(half* output, const half* contribution,
                            size_t count, cudaStream_t stream) {
  constexpr int threads = 256;
  AddMoeFp16OutputKernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0, stream>>>(
      output, contribution, count);
  CUDA_CALL_THROW(cudaGetLastError());
}

void LaunchAddMoeBf16Output(__nv_bfloat16* output, const __nv_bfloat16* contribution,
                            size_t count, cudaStream_t stream) {
  constexpr int threads = 256;
  AddMoeBf16OutputKernel<<<static_cast<unsigned int>((count + threads - 1) / threads), threads, 0, stream>>>(
      output, contribution, count);
  CUDA_CALL_THROW(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda
