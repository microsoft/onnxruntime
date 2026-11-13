// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cub/cub.cuh>

#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/accumulation_type.h"
#include "core/providers/cuda/tensor/mean_variance_normalization_impl.h"

namespace onnxruntime::cuda {

__device__ __forceinline__ int64_t MvnOffset(int64_t row, int64_t column,
                                             int64_t rows, int64_t columns, int64_t inner) {
  return inner == 0 ? row * columns + column
                    : ((column / inner) * rows + row) * inner + column % inner;
}

template <typename T, int Threads>
__global__ void MeanVarianceNormalizationKernel(const T* input, T* output,
                                                int64_t rows, int64_t columns, int64_t inner, float epsilon) {
  using AccT = AccumulationType_t<T>;
  using Reduce = cub::BlockReduce<AccT, Threads>;
  __shared__ typename Reduce::TempStorage storage;
  __shared__ AccT mean, denominator;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    AccT sum = 0;
    for (int64_t col = threadIdx.x; col < columns; col += Threads) {
      sum += static_cast<AccT>(input[MvnOffset(row, col, rows, columns, inner)]);
    }
    const AccT total = Reduce(storage).Sum(sum);
    if (threadIdx.x == 0) mean = total / static_cast<AccT>(columns);
    __syncthreads();
    AccT squares = 0;
    for (int64_t col = threadIdx.x; col < columns; col += Threads) {
      const AccT centered = static_cast<AccT>(input[MvnOffset(row, col, rows, columns, inner)]) - mean;
      squares += centered * centered;
    }
    const AccT total_squares = Reduce(storage).Sum(squares);
    if (threadIdx.x == 0) {
      denominator = _Sqrt(total_squares / static_cast<AccT>(columns)) + static_cast<AccT>(epsilon);
    }
    __syncthreads();
    for (int64_t col = threadIdx.x; col < columns; col += Threads) {
      const int64_t offset = MvnOffset(row, col, rows, columns, inner);
      output[offset] = static_cast<T>((static_cast<AccT>(input[offset]) - mean) / denominator);
    }
    __syncthreads();
  }
}

template <typename T, bool Centered>
__global__ void MvnPartialSums(const T* input, AccumulationType_t<T>* partials,
                               const AccumulationType_t<T>* means,
                               int64_t rows, int64_t columns, int64_t inner, int64_t chunks) {
  using AccT = AccumulationType_t<T>;
  using Reduce = cub::BlockReduce<AccT, 256>;
  __shared__ typename Reduce::TempStorage storage;
  for (int64_t unit = blockIdx.x; unit < rows * chunks; unit += gridDim.x) {
    const int64_t row = unit / chunks, chunk = unit % chunks;
    const int64_t remainder = columns % chunks;
    const int64_t begin = (columns / chunks) * chunk + (chunk < remainder ? chunk : remainder);
    const int64_t end = begin + columns / chunks + (chunk < columns % chunks);
    AccT sum = 0;
    for (int64_t col = begin + threadIdx.x; col < end; col += 256) {
      AccT value = static_cast<AccT>(input[MvnOffset(row, col, rows, columns, inner)]);
      if constexpr (Centered) {
        value -= means[row];
        value *= value;
      }
      sum += value;
    }
    const AccT total = Reduce(storage).Sum(sum);
    if (threadIdx.x == 0) partials[unit] = total;
    __syncthreads();
  }
}

template <typename AccT, bool Variance>
__global__ void MvnFinalizeSums(const AccT* partials, AccT* result,
                                int64_t rows, int64_t columns, int64_t chunks, float epsilon) {
  using Reduce = cub::BlockReduce<AccT, 256>;
  __shared__ typename Reduce::TempStorage storage;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    const AccT sum = threadIdx.x < chunks ? partials[row * chunks + threadIdx.x] : AccT{0};
    const AccT total = Reduce(storage).Sum(sum);
    if (threadIdx.x == 0) {
      const AccT average = total / static_cast<AccT>(columns);
      if constexpr (Variance) {
        result[row] = _Sqrt(average) + static_cast<AccT>(epsilon);
      } else {
        result[row] = average;
      }
    }
    __syncthreads();
  }
}

template <typename T>
__global__ void MvnNormalize(const T* input, T* output,
                             const AccumulationType_t<T>* means, const AccumulationType_t<T>* denominators,
                             int64_t rows, int64_t columns, int64_t inner) {
  using AccT = AccumulationType_t<T>;
  for (int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < rows * columns; index += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t row = index / columns, column = index % columns;
    const int64_t offset = MvnOffset(row, column, rows, columns, inner);
    output[offset] = static_cast<T>((static_cast<AccT>(input[offset]) - means[row]) / denominators[row]);
  }
}

template <typename T>
void MeanVarianceNormalizationImpl(cudaStream_t stream, const T* input, T* output,
                                   int64_t rows, int64_t columns, int64_t inner, float epsilon, void* workspace) {
  const int blocks = static_cast<int>(std::min<int64_t>(rows, 65535));
  const int64_t chunks = MeanVarianceNormalizationChunks(rows, columns);
  if (chunks > 1) {
    using AccT = AccumulationType_t<T>;
    auto* partials = static_cast<AccT*>(workspace);
    auto* means = partials + rows * chunks;
    auto* denominators = means + rows;
    const int partial_blocks = static_cast<int>(std::min<int64_t>(rows * chunks, 65535));
    const int element_blocks = static_cast<int>(std::min<int64_t>((rows * columns - 1) / 256 + 1, 65535));
    MvnPartialSums<T, false><<<partial_blocks, 256, 0, stream>>>(input, partials, means, rows, columns, inner, chunks);
    MvnFinalizeSums<AccT, false><<<blocks, 256, 0, stream>>>(partials, means, rows, columns, chunks, epsilon);
    MvnPartialSums<T, true><<<partial_blocks, 256, 0, stream>>>(input, partials, means, rows, columns, inner, chunks);
    MvnFinalizeSums<AccT, true><<<blocks, 256, 0, stream>>>(partials, denominators, rows, columns, chunks, epsilon);
    MvnNormalize<<<element_blocks, 256, 0, stream>>>(input, output, means, denominators, rows, columns, inner);
    return;
  }
  if (columns <= 32) {
    MeanVarianceNormalizationKernel<T, 32><<<blocks, 32, 0, stream>>>(input, output, rows, columns, inner, epsilon);
  } else {
    MeanVarianceNormalizationKernel<T, 256><<<blocks, 256, 0, stream>>>(input, output, rows, columns, inner, epsilon);
  }
}

template void MeanVarianceNormalizationImpl<float>(cudaStream_t, const float*, float*, int64_t, int64_t, int64_t, float, void*);
template void MeanVarianceNormalizationImpl<double>(cudaStream_t, const double*, double*, int64_t, int64_t, int64_t, float, void*);
template void MeanVarianceNormalizationImpl<half>(cudaStream_t, const half*, half*, int64_t, int64_t, int64_t, float, void*);
template void MeanVarianceNormalizationImpl<BFloat16>(cudaStream_t, const BFloat16*, BFloat16*, int64_t, int64_t, int64_t, float, void*);

}  // namespace onnxruntime::cuda
