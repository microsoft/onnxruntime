// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace onnxruntime::test {
namespace {

__global__ void HcIndependentProducer(const uint16_t* source, uint16_t* destination, size_t count,
                                     const uint16_t* weight_source, uint16_t* weight_destination,
                                     size_t weight_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
#endif
  // Finite independent work after the early trigger: completion never depends on HC.
  const auto start = clock64();
  while (clock64() - start < 200000ull) {}
  for (size_t i = threadIdx.x; i < count; i += blockDim.x) destination[i] = source[i];
  for (size_t i = threadIdx.x; i < weight_count; i += blockDim.x) weight_destination[i] = weight_source[i];
}

}  // namespace

cudaError_t LaunchHcIndependentProducer(const uint16_t* source, uint16_t* destination,
                                       size_t count, cudaStream_t stream,
                                       const uint16_t* weight_source, uint16_t* weight_destination,
                                       size_t weight_count) {
  HcIndependentProducer<<<1, 128, 0, stream>>>(source, destination, count, weight_source,
                                             weight_destination, weight_count);
  return cudaGetLastError();
}

}  // namespace onnxruntime::test
