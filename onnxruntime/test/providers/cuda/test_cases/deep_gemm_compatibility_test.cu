// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_DEEP_GEMM

#include <cstdio>
#include <cstring>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <deep_gemm/common/math.cuh>
#include <deep_gemm/ptx/ld_st.cuh>

namespace onnxruntime::test {
namespace {

__global__ void DeepGemmScalesKernel(float2* storage, int count) {
  const int index = threadIdx.x;
  if (index >= count) return;
  deep_gemm::math::get_e4m3_sf_and_sf_inv(storage[index], storage[count + index], storage[2 * count + index]);
}

__global__ void DeepGemmPackedStoreKernel(uint4* storage) {
  constexpr int count = 32;
  const int index = threadIdx.x;
  __shared__ __int128_t values[count];
  __int128_t value;
  static_assert(sizeof(value) == sizeof(uint4));
  memcpy(&value, &storage[index], sizeof(value));
  deep_gemm::ptx::st_shared(&values[index], value);
  __syncthreads();
  // Read another thread's store so the shared-memory round trip cannot be elided.
  memcpy(&storage[count + index], &values[(index + 1) % count], sizeof(uint4));
}

}  // namespace

cudaError_t LaunchDeepGemmScales(void* storage, int count) {
  DeepGemmScalesKernel<<<1, 32, 0, nullptr>>>(static_cast<float2*>(storage), count);
  return cudaGetLastError();
}

cudaError_t LaunchDeepGemmPackedStore(void* storage) {
  DeepGemmPackedStoreKernel<<<1, 32, 0, nullptr>>>(static_cast<uint4*>(storage));
  return cudaGetLastError();
}

}  // namespace onnxruntime::test

#endif  // USE_DEEP_GEMM
