// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>

#include "contrib_ops/cuda/math/ordered_reductions.h"

namespace onnxruntime::contrib::cuda {
namespace {

// These trees, including the row-count-dependent partition, are part of the v1
// contract. Explicit round-to-nearest operations prevent compiler reassociation.
__global__ void OrderedRowReduceKernel(const float* input, float* output, size_t rows,
                                       int width, int vector, int threads, bool normalize) {
  __shared__ float partial[512];
  for (size_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const int tid = threadIdx.x;
    const int groups = width / (threads * vector);
    float accumulators[4];
    for (int v = 0; v < vector; ++v) {
      accumulators[v] = input[row * width + tid * vector + v];
    }
    for (int group = 1; group < groups; ++group) {
      for (int v = 0; v < vector; ++v) {
        accumulators[v] = __fadd_rn(accumulators[v],
                                    input[row * width + (group * threads + tid) * vector + v]);
      }
    }
    float sum = accumulators[0];
    for (int v = 1; v < vector; ++v) {
      sum = __fadd_rn(sum, accumulators[v]);
    }
    partial[tid] = sum;
    __syncthreads();
    for (int count = threads; count > 32; count /= 2) {
      if (tid < count / 2) {
        partial[tid] = __fadd_rn(partial[tid], partial[tid + count / 2]);
      }
      __syncthreads();
    }
    if (tid < 32) {
      sum = partial[tid];
      for (int stride = 1; stride < 32; stride *= 2) {
        sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, stride));
      }
      if (tid == 0) {
        output[row] = normalize ? __fmul_rn(sum, 1.0f / width) : sum;
      }
    }
    __syncthreads();
  }
}

__global__ void OrderedMatrixReduce128Kernel(const float* input, float* output, size_t matrices) {
  __shared__ float partial[4][128];
  for (size_t matrix = blockIdx.x; matrix < matrices; matrix += gridDim.x) {
    const int column = threadIdx.x * 4;
    const int lane = threadIdx.y;
    float accumulators[4][4] = {};
    for (int group = 0; group < 8; ++group) {
      for (int accumulator = 0; accumulator < 4; ++accumulator) {
        for (int v = 0; v < 4; ++v) {
          accumulators[accumulator][v] = __fadd_rn(
              accumulators[accumulator][v],
              input[matrix * 128 * 128 + (group * 16 + accumulator * 4 + lane) * 128 + column + v]);
        }
      }
    }
    for (int v = 0; v < 4; ++v) {
      float sum = accumulators[0][v];
      for (int accumulator = 1; accumulator < 4; ++accumulator) {
        sum = __fadd_rn(sum, accumulators[accumulator][v]);
      }
      partial[lane][column + v] = sum;
    }
    __syncthreads();
    if (lane == 0) {
      for (int v = 0; v < 4; ++v) {
        output[matrix * 128 + column + v] =
            __fadd_rn(__fadd_rn(partial[0][column + v], partial[2][column + v]),
                      __fadd_rn(partial[1][column + v], partial[3][column + v]));
      }
    }
    __syncthreads();
  }
}

}  // namespace

Status LaunchOrderedRowReduce(cudaStream_t stream, const float* input, float* output,
                              size_t rows, int width, bool normalize) {
  const int vector = width > 128 ? 4 : 1;
  int threads = std::min(width / vector, 512);
  for (int cutoff : {2, 4, 8, 16}) {
    if (rows >= static_cast<size_t>(cutoff)) {
      threads = std::min(width / vector, 512 / cutoff);
    }
  }
  const auto blocks = static_cast<unsigned int>(std::min(rows, size_t{65535}));
  OrderedRowReduceKernel<<<blocks, threads, 0, stream>>>(input, output, rows, width, vector, threads, normalize);
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchOrderedMatrixReduce128(cudaStream_t stream, const float* input, float* output, size_t matrices) {
  const auto blocks = static_cast<unsigned int>(std::min(matrices, size_t{65535}));
  OrderedMatrixReduce128Kernel<<<blocks, dim3(32, 4), 0, stream>>>(input, output, matrices);
  return CUDA_CALL(cudaGetLastError());
}

}  // namespace onnxruntime::contrib::cuda
