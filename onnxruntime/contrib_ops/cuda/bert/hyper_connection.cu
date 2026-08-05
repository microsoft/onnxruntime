// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/hyper_connection.h"
#include "contrib_ops/cuda/bert/hyper_connection_common.cuh"

#include <cfloat>
#include <cmath>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void HyperConnectionKernel(
    T* post, T* comb, T* collapsed, const T* hidden, const float* weight,
    const float* bias, const float* scale, int streams, int hidden_size,
    float epsilon, int sinkhorn_iterations) {
  const int row = blockIdx.x;
  const int input_size = streams * hidden_size;
  const int projection_size = (2 + streams) * streams;
  extern __shared__ float shared[];
  float* pre = shared;
  float* matrix = shared + streams;
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
    const float pre_value = ProjectHyperValue(hidden, weight, row, input_size, stream, inverse_rms) * scale[0] + bias[stream];
    const float post_value = ProjectHyperValue(hidden, weight, row, input_size, streams + stream, inverse_rms) * scale[1] + bias[streams + stream];
    pre[stream] = 1.0f / (1.0f + expf(-pre_value)) + epsilon;
    post[row * streams + stream] = HyperFromFloat<T>(2.0f / (1.0f + expf(-post_value)));
  }
  for (int index = threadIdx.x; index < streams * streams; index += blockDim.x) {
    matrix[index] = ProjectHyperValue(hidden, weight, row, input_size, 2 * streams + index, inverse_rms) * scale[2] +
            bias[2 * streams + index];
  }
  __syncthreads();

  for (int output_stream = threadIdx.x; output_stream < streams; output_stream += blockDim.x) {
    float maximum = -FLT_MAX;
    for (int input_stream = 0; input_stream < streams; ++input_stream) {
      maximum = fmaxf(maximum, matrix[output_stream * streams + input_stream]);
    }
    float denominator = 0.0f;
    for (int input_stream = 0; input_stream < streams; ++input_stream) {
      const int index = output_stream * streams + input_stream;
      matrix[index] = expf(matrix[index] - maximum);
      denominator += matrix[index];
    }
    for (int input_stream = 0; input_stream < streams; ++input_stream) {
      const int index = output_stream * streams + input_stream;
      matrix[index] = matrix[index] / denominator + epsilon;
    }
  }
  __syncthreads();

  for (int input_stream = threadIdx.x; input_stream < streams; input_stream += blockDim.x) {
    float sum = 0.0f;
    for (int output_stream = 0; output_stream < streams; ++output_stream) {
      sum += matrix[output_stream * streams + input_stream];
    }
    sum += epsilon;
    for (int output_stream = 0; output_stream < streams; ++output_stream) {
      matrix[output_stream * streams + input_stream] /= sum;
    }
  }
  __syncthreads();

  for (int iteration = 1; iteration < sinkhorn_iterations; ++iteration) {
    for (int output_stream = threadIdx.x; output_stream < streams; output_stream += blockDim.x) {
      float sum = 0.0f;
      for (int input_stream = 0; input_stream < streams; ++input_stream) {
        sum += matrix[output_stream * streams + input_stream];
      }
      sum += epsilon;
      for (int input_stream = 0; input_stream < streams; ++input_stream) {
        matrix[output_stream * streams + input_stream] /= sum;
      }
    }
    __syncthreads();
    for (int input_stream = threadIdx.x; input_stream < streams; input_stream += blockDim.x) {
      float sum = 0.0f;
      for (int output_stream = 0; output_stream < streams; ++output_stream) {
        sum += matrix[output_stream * streams + input_stream];
      }
      sum += epsilon;
      for (int output_stream = 0; output_stream < streams; ++output_stream) {
        matrix[output_stream * streams + input_stream] /= sum;
      }
    }
    __syncthreads();
  }

  for (int index = threadIdx.x; index < streams * streams; index += blockDim.x) {
    comb[static_cast<int64_t>(row) * streams * streams + index] = HyperFromFloat<T>(matrix[index]);
  }
  for (int d = threadIdx.x; d < hidden_size; d += blockDim.x) {
    float value = 0.0f;
    for (int stream = 0; stream < streams; ++stream) {
      value += pre[stream] * HyperToFloat(hidden[(static_cast<int64_t>(row) * streams + stream) * hidden_size + d]);
    }
    collapsed[static_cast<int64_t>(row) * hidden_size + d] = HyperFromFloat<T>(value);
  }
}

template <typename T>
Status LaunchHyperConnectionKernel(
    cudaStream_t stream, T* post, T* comb, T* collapsed, const T* hidden,
    const float* weight, const float* bias, const float* scale, int rows,
    int streams, int hidden_size, float epsilon, int sinkhorn_iterations,
    int max_threads_per_block) {
  const int threads = HyperBlockSize(hidden_size, max_threads_per_block);
  const size_t shared_bytes = static_cast<size_t>(streams + streams * streams) * sizeof(float);
  HyperConnectionKernel<T><<<rows, threads, shared_bytes, stream>>>(
      post, comb, collapsed, hidden, weight, bias, scale, streams, hidden_size,
      epsilon, sinkhorn_iterations);
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchHyperConnectionKernel<float>(cudaStream_t, float*, float*, float*, const float*, const float*, const float*, const float*, int, int, int, float, int, int);
template Status LaunchHyperConnectionKernel<half>(cudaStream_t, half*, half*, half*, const half*, const float*, const float*, const float*, int, int, int, float, int, int);
template Status LaunchHyperConnectionKernel<BFloat16>(cudaStream_t, BFloat16*, BFloat16*, BFloat16*, const BFloat16*, const float*, const float*, const float*, int, int, int, float, int, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime