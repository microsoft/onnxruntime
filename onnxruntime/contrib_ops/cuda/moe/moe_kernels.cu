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

}  // namespace onnxruntime::contrib::cuda
