// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/hyper_connection_common.cuh"
#include "contrib_ops/cuda/bert/hyper_head.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void HyperHeadKernel(
    T* output, const T* hidden, const float* weight, const float* bias,
    const float* scale, int streams, int hidden_size, float epsilon) {
  const int row = blockIdx.x;
  const int input_size = streams * hidden_size;
  extern __shared__ float pre[];
  __shared__ float inverse_rms;
  if (threadIdx.x == 0) {
    float square_sum = 0.0f;
    const T* input_row = hidden + static_cast<int64_t>(row) * input_size;
    for (int input = 0; input < input_size; ++input) {
      const float value = HyperToFloat(input_row[input]);
      square_sum += value * value;
    }
    inverse_rms = rsqrtf(square_sum / static_cast<float>(input_size) + epsilon);
  }
  __syncthreads();
  for (int stream = threadIdx.x; stream < streams; stream += blockDim.x) {
    const float projected = ProjectHyperValue(hidden, weight, row, input_size, stream, inverse_rms) *
                                scale[0] +
                            bias[stream];
    pre[stream] = 1.0f / (1.0f + expf(-projected)) + epsilon;
  }
  __syncthreads();
  for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
    float value = 0.0f;
    for (int stream = 0; stream < streams; ++stream) {
      value += pre[stream] *
               HyperToFloat(hidden[(static_cast<int64_t>(row) * streams + stream) * hidden_size + d]);
    }
    output[static_cast<int64_t>(row) * hidden_size + d] = HyperFromFloat<T>(value);
  }
}

template <typename T>
Status LaunchHyperHeadKernel(
    cudaStream_t stream, T* output, const T* hidden, const float* weight,
    const float* bias, const float* scale, int rows, int streams,
    int hidden_size, float epsilon, int max_threads_per_block) {
  const int threads = HyperBlockSize(hidden_size, max_threads_per_block);
  HyperHeadKernel<T><<<rows, threads, static_cast<size_t>(streams) * sizeof(float), stream>>>(
      output, hidden, weight, bias, scale, streams, hidden_size, epsilon);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchHyperHeadKernel<float>(cudaStream_t, float*, const float*, const float*, const float*, const float*, int, int, int, float, int);
template Status LaunchHyperHeadKernel<half>(cudaStream_t, half*, const half*, const float*, const float*, const float*, int, int, int, float, int);
template Status LaunchHyperHeadKernel<BFloat16>(cudaStream_t, BFloat16*, const BFloat16*, const float*, const float*, const float*, int, int, int, float, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime