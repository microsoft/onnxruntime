// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "normalize_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, int blockSize>
__launch_bounds__(blockSize)
    __global__ void normalizeKernel(const T* input, T* output, int dim1, float* gamma, float* beta) {
  __shared__ float average_shared[blockSize];
  __shared__ float std_shared[blockSize];

  float average = 0.;
  int stride = dim1;
  const T* input_start = input + stride * blockIdx.x;
  T* output_start = output + stride * blockIdx.x;
  for (int i = threadIdx.x; i < dim1; i += blockSize) {
    average += (float)input_start[i];  // load from memory from native precision, convert to float for ops
  }
  average_shared[threadIdx.x] = average;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < blockSize; ++i) {
      average += average_shared[i];
    }
    average_shared[0] = (T)(average / dim1);  //typecast to native precision
  }
  __syncthreads();

  // std deviation
  average = average_shared[0];
  float stdDev = 0.;
  for (int i = threadIdx.x; i < dim1; i += blockSize) {
    float val = (float)(input_start[i]) - average;
    stdDev += val * val;
  }
  std_shared[threadIdx.x] = stdDev;
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int i = 1; i < blockSize; ++i) {
      stdDev += std_shared[i];
    }
    std_shared[0] = (T)(sqrtf(stdDev / dim1 + 1E-12));  //typecast to native precision
  }
  __syncthreads();
  stdDev = std_shared[0];

  for (int i = threadIdx.x; i < dim1; i += blockSize) {
    float x = input_start[i];
    output_start[i] = (T)(((x - average) / stdDev) * gamma[i] + beta[i]);
  }
}

void launchNormalizeKernel(const float* input,
                           float* output,
                           float* gamma_ptr,  // gamma
                           float* beta_ptr,   // beta
                           int nBatch,
                           int sequence_len,
                           int encode_len  //,
                           /*int isFP16*/) {
  // size_t elementSize = isFP16 ? 2 : 4;
  const int blockSize = 32;
  const int gridSize = sequence_len * nBatch;

  //if( isFP16 )
  //   normalizeKernel<__half, blockSize> << <gridSize, blockSize, 0, stream >> > ( static_cast< const __half* >( input ),
  //      static_cast< __half* >( output ), meanArray, stdArray, d[ 1 ], params[ 0 ], params[ 1 ] );
  //else
  normalizeKernel<float, blockSize><<<gridSize, blockSize, 0>>>(input, output, encode_len, gamma_ptr, beta_ptr);
}

}  // namespace cuda
}  //namespace contrib
}  // namespace onnxruntime
