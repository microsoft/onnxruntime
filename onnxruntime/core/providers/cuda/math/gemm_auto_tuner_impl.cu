// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/math/gemm_auto_tuner.h"

#include <algorithm>

namespace onnxruntime {
namespace cuda {
namespace {

__device__ __forceinline__ uint64_t GlobalTimerNs() {
  uint64_t now;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
  return now;
}

__global__ void GpuDelayKernel(uint64_t nanoseconds) {
  const uint64_t start = GlobalTimerNs();
  while (GlobalTimerNs() - start < nanoseconds) {
  }
}

// Pure reads: lines evicted from L2 stay clean, so the next kernel pays no write-back.
__global__ void L2ReadKernel(const uint4* __restrict__ buffer, size_t count, uint32_t* __restrict__ sink) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < count;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const uint4 v = __ldcg(buffer + i);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  // Keeps the loads alive; callers pass a null sink, so nothing is written.
  if (acc == 0x9e3779b9u && sink != nullptr) {
    *sink = acc;
  }
}

}  // namespace

Status LaunchGpuDelay(cudaStream_t stream, uint64_t nanoseconds) {
  GpuDelayKernel<<<1, 1, 0, stream>>>(nanoseconds);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchL2Read(cudaStream_t stream, const void* buffer, size_t bytes, int num_sms) {
  if (buffer == nullptr) {
    return Status::OK();
  }
  // Reads whole 16-byte words only; a partial word at either end is left out.
  const uintptr_t begin = (reinterpret_cast<uintptr_t>(buffer) + 15) & ~uintptr_t{15};
  const uintptr_t end = reinterpret_cast<uintptr_t>(buffer) + bytes;
  const size_t count = end > begin ? (end - begin) / sizeof(uint4) : 0;
  if (count == 0) {
    return Status::OK();
  }
  constexpr int kThreads = 512;
  const size_t max_blocks = static_cast<size_t>(std::max(num_sms, 1)) * 4;
  const int blocks = static_cast<int>(std::min(max_blocks, (count + kThreads - 1) / kThreads));
  L2ReadKernel<<<blocks, kThreads, 0, stream>>>(reinterpret_cast<const uint4*>(begin), count, nullptr);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace cuda
}  // namespace onnxruntime
