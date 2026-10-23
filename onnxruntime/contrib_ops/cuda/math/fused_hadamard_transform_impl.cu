// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/fused_hadamard_transform_impl.cuh"

#include <algorithm>
#include <cmath>

namespace onnxruntime {
namespace contrib {
namespace cuda {

namespace {

__global__ void FusedHadamardTransformKernel(
    const half* input,
    const half* sign,
    half* output,
    int64_t sign_size,
    int block_size,
    float normalization) {
  extern __shared__ float values[];

  const int64_t block_offset = static_cast<int64_t>(blockIdx.x) * block_size;
  const int64_t sign_offset = block_offset % sign_size;
  for (int index = threadIdx.x; index < block_size; index += blockDim.x) {
    values[index] = __half2float(input[block_offset + index]) *
                    __half2float(sign[sign_offset + index]);
  }
  __syncthreads();

  for (int stride = 1; stride < block_size; stride <<= 1) {
    for (int pair = threadIdx.x; pair < block_size / 2; pair += blockDim.x) {
      const int group = pair / stride;
      const int offset = pair - group * stride;
      const int left = group * 2 * stride + offset;
      const int right = left + stride;
      const float a = values[left];
      const float b = values[right];
      values[left] = a + b;
      values[right] = a - b;
    }
    __syncthreads();
  }

  for (int index = threadIdx.x; index < block_size; index += blockDim.x) {
    output[block_offset + index] = __float2half_rn(values[index] * normalization);
  }
}

}  // namespace

cudaError_t LaunchFusedHadamardTransform(
    cudaStream_t stream,
    const half* input,
    const half* sign,
    half* output,
    int64_t block_count,
    int64_t sign_size,
    int block_size) {
  constexpr int kMaxThreadsPerBlock = 256;
  const int thread_count = std::min(block_size, kMaxThreadsPerBlock);
  const size_t shared_memory_bytes = static_cast<size_t>(block_size) * sizeof(float);
  const float normalization = 1.0f / std::sqrt(static_cast<float>(block_size));
  FusedHadamardTransformKernel<<<static_cast<uint32_t>(block_count), thread_count, shared_memory_bytes, stream>>>(
      input, sign, output, sign_size, block_size, normalization);
  return cudaGetLastError();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime