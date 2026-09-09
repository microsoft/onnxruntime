#pragma once

#include <cuda_runtime.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

__device__ __forceinline__ float PagedHadamard(float value, float* shared_values, int head_size) {
  const int channel = threadIdx.x;
  for (int stride = 1; stride < head_size; stride <<= 1) {
    shared_values[channel] = value;
    __syncthreads();
    const float partner = shared_values[channel ^ stride];
    value = (channel & stride) ? partner - value : value + partner;
    __syncthreads();
  }
  return value * rsqrtf(static_cast<float>(head_size));
}

template <typename T>
__global__ void PagedHadamardHeads(const T* input, T* output, int head_size, int input_stride, int num_heads) {
  extern __shared__ float shared_values[];
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int channel = threadIdx.x;
  const float value = static_cast<float>(input[static_cast<int64_t>(token) * input_stride + head * head_size + channel]);
  output[(static_cast<int64_t>(token) * num_heads + head) * head_size + channel] =
      static_cast<T>(PagedHadamard(value, shared_values, head_size));
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime