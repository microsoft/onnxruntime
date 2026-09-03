// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <array>
#include <cooperative_groups.h>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <limits>
#include <type_traits>

#include "core/providers/cuda/cu_inc/topk_warp_sort.cuh"

namespace onnxruntime {
namespace cuda {
namespace hybrid_topk {

namespace cg = cooperative_groups;

constexpr int kMaxK = 256;
constexpr int kMaxPartitions = 256;
constexpr int kStage1Threads = 256;
constexpr std::array<int, 4> kPartitionSizes = {1792, 2304, 2816, 3328};

constexpr int DivUp(int value, int divisor) {
  return value / divisor + (value % divisor != 0);
}

constexpr int Max(int a, int b, int c) {
  return std::max(a, std::max(b, c));
}

struct ReductionFactors {
  int factor1;
  int factor2;
  int factor3;
  int steps;
};

constexpr ReductionFactors GetReductionFactors(int partitions) {
  if (partitions <= 1) return {1, 1, 1, 0};
  if (partitions <= 4) return {4, 1, 1, 1};
  if (partitions <= 8) return {8, 1, 1, 1};
  if (partitions <= 16) return {16, 1, 1, 1};
  if (partitions <= 32) return {8, 4, 1, 2};
  if (partitions <= 64) return {8, 8, 1, 2};
  if (partitions <= 128) return {8, 8, 2, 3};
  return {8, 8, 4, 3};
}

inline int EstimateBestPartitionSize(int dimension) {
  constexpr std::array<int, 8> targets = {2, 4, 8, 16, 32, 64, 128, 256};
  int best_partition_size = kPartitionSizes[0];
  double best_waste = std::numeric_limits<double>::infinity();
  for (int partition_size : kPartitionSizes) {
    const int partitions = DivUp(dimension, partition_size);
    if (partitions > kMaxPartitions) {
      continue;
    }
    const auto target = std::lower_bound(targets.begin(), targets.end(), partitions);
    if (target != targets.end()) {
      const double waste = static_cast<double>(*target - partitions) / *target;
      if (waste < best_waste) {
        best_waste = waste;
        best_partition_size = partition_size;
      }
    }
  }
  return best_partition_size;
}

inline int PaddedK(int k) {
  if (k <= 4) return 4;
  if (k <= 8) return 8;
  if (k <= 16) return 16;
  if (k <= 32) return 32;
  if (k <= 64) return 64;
  if (k <= 128) return 128;
  return 256;
}

// H200 sweeps show that the K crossover rises with the row count. SmallK or RadixTopK is faster
// outside these regions, especially for K = 1.
inline bool IsPreferred(int64_t rows, int64_t dimension, int64_t k) {
  return dimension >= 8192 &&
         ((rows <= 2 && k >= 8) || (rows <= 4 && k >= 32) || (rows > 4 && k >= 64));
}

template <int K>
constexpr int ReductionBlockSize() {
  return K <= 16 ? 128 : 256;
}

template <typename T, int PartitionSize, int K>
__global__ void FindPartitionTopK(const T* __restrict__ input,
                                  float* __restrict__ scores,
                                  int* __restrict__ indices,
                                  int dimension,
                                  int num_partitions) {
  constexpr int items_per_thread = PartitionSize / kStage1Threads;
  using Sort = cub::BlockRadixSort<uint64_t, kStage1Threads, items_per_thread>;
  __shared__ typename Sort::TempStorage storage;

  const int row = blockIdx.y;
  const int partition = blockIdx.x;
  const int partition_start = partition * PartitionSize;
  const T* row_input = input + static_cast<size_t>(row) * dimension;
  uint64_t keys[items_per_thread];
#pragma unroll
  for (int item = 0; item < items_per_thread; ++item) {
    const int index = partition_start + threadIdx.x + item * kStage1Threads;
    keys[item] = index < dimension
                     ? topk::PackStableSortKey(static_cast<float>(row_input[index]), index)
                     : topk::PackStableSortKey(topk::kNegativeInfinity, INT_MAX);
  }
  Sort(storage).SortDescendingBlockedToStriped(keys);
  if (threadIdx.x < K) {
    const size_t offset =
        (static_cast<size_t>(row) * num_partitions + partition) * K + threadIdx.x;
    scores[offset] = topk::UnpackStableSortScore(keys[0]);
    indices[offset] = topk::UnpackStableSortIndex(keys[0]);
  }
}

template <int BlockSize, int MaxSortSize>
union ReductionStorage {
  typename cub::WarpMergeSort<uint64_t, DivUp(MaxSortSize, topk::kWarpSize),
                              topk::kWarpSize, cub::NullType>::TempStorage warp_merge;
  typename cub::BlockMergeSort<uint64_t, BlockSize, DivUp(MaxSortSize, BlockSize),
                               cub::NullType>::TempStorage block_merge;
  struct {
    __align__(128) float scores[MaxSortSize];
    __align__(128) int indices[MaxSortSize];
  } values;
};

template <int BlockSize, int SortSize, int K, typename Storage>
__device__ void ReducePartitions(const float* input_scores,
                                 const int* input_indices,
                                 float* output_scores,
                                 int* output_indices,
                                 int input_partitions,
                                 int output_partition,
                                 Storage& storage) {
  const int first_partition = output_partition * (SortSize / K);
  const int valid_partitions = min(SortSize / K, input_partitions - first_partition);
  const int valid_items = valid_partitions * K;

  if constexpr (SortSize <= topk::kWarpBitonicMaxSize) {
    for (int item = threadIdx.x; item < SortSize; item += BlockSize) {
      if (item < valid_items) {
        const size_t offset = static_cast<size_t>(first_partition) * K + item;
        storage.values.scores[item] = input_scores[offset];
        storage.values.indices[item] = input_indices[offset];
      }
    }
    __syncthreads();
    if (threadIdx.x < topk::kWarpSize) {
      float score = threadIdx.x < valid_items ? storage.values.scores[threadIdx.x]
                                              : topk::kNegativeInfinity;
      int index = threadIdx.x < valid_items ? storage.values.indices[threadIdx.x] : INT_MAX;
      topk::WarpBitonicSortDescending(score, index);
      if (threadIdx.x < K) {
        storage.values.scores[threadIdx.x] = score;
        storage.values.indices[threadIdx.x] = index;
      }
    }
  } else if constexpr (SortSize <= topk::kWarpMergeMaxSize) {
    for (int item = threadIdx.x; item < SortSize; item += BlockSize) {
      if (item < valid_items) {
        const size_t offset = static_cast<size_t>(first_partition) * K + item;
        storage.values.scores[item] = input_scores[offset];
        storage.values.indices[item] = input_indices[offset];
      }
    }
    __syncthreads();
    using Sorter = topk::WarpMergeSorter<SortSize>;
    Sorter::Sort(storage.values.scores, storage.values.indices,
                 reinterpret_cast<typename Sorter::TempStorage&>(storage.warp_merge), valid_items);
  } else {
    constexpr int items_per_thread = DivUp(SortSize, BlockSize);
    using Sort = cub::BlockMergeSort<uint64_t, BlockSize, items_per_thread, cub::NullType>;
    uint64_t keys[items_per_thread];
#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
      const int item_index = threadIdx.x * items_per_thread + item;
      if (item_index < valid_items) {
        const size_t offset = static_cast<size_t>(first_partition) * K + item_index;
        keys[item] = topk::PackStableSortKey(input_scores[offset], input_indices[offset]);
      } else {
        keys[item] = topk::PackStableSortKey(topk::kNegativeInfinity, INT_MAX);
      }
    }
    Sort(reinterpret_cast<typename Sort::TempStorage&>(storage.block_merge))
        .Sort(keys, topk::Greater<uint64_t>());
    float thread_scores[items_per_thread];
    int thread_indices[items_per_thread];
#pragma unroll
    for (int item = 0; item < items_per_thread; ++item) {
      thread_scores[item] = topk::UnpackStableSortScore(keys[item]);
      thread_indices[item] = topk::UnpackStableSortIndex(keys[item]);
    }
    cub::StoreDirectBlocked(threadIdx.x, output_scores + static_cast<size_t>(output_partition) * K,
                            thread_scores, K);
    cub::StoreDirectBlocked(threadIdx.x, output_indices + static_cast<size_t>(output_partition) * K,
                            thread_indices, K);
    return;
  }

  __syncthreads();
  if (threadIdx.x < K) {
    const size_t offset = static_cast<size_t>(output_partition) * K + threadIdx.x;
    output_scores[offset] = storage.values.scores[threadIdx.x];
    output_indices[offset] = storage.values.indices[threadIdx.x];
  }
}

template <int K, int BlockSize, int Factor1, int Factor2, int Factor3>
__global__ void CooperativeReduce(int* indices1,
                                  float* scores1,
                                  int* indices2,
                                  float* scores2,
                                  int num_partitions) {
  cg::grid_group grid = cg::this_grid();
  constexpr int max_sort_size = Max(K * Factor1, K * Factor2, K * Factor3);
  __shared__ ReductionStorage<BlockSize, max_sort_size> storage;
  const int partition = blockIdx.x;
  const int row = blockIdx.y;

  int partitions1 = num_partitions;
  if constexpr (Factor1 > 1) {
    partitions1 = DivUp(num_partitions, Factor1);
    if (partition < partitions1) {
      const size_t input_row = static_cast<size_t>(row) * num_partitions * K;
      const size_t output_row = static_cast<size_t>(row) * partitions1 * K;
      ReducePartitions<BlockSize, K * Factor1, K>(
          scores1 + input_row, indices1 + input_row, scores2 + output_row, indices2 + output_row,
          num_partitions, partition, storage);
    }
    grid.sync();
  }

  int partitions2 = partitions1;
  if constexpr (Factor2 > 1) {
    partitions2 = DivUp(partitions1, Factor2);
    if (partition < partitions2) {
      const size_t input_row = static_cast<size_t>(row) * partitions1 * K;
      const size_t output_row = static_cast<size_t>(row) * partitions2 * K;
      ReducePartitions<BlockSize, K * Factor2, K>(
          scores2 + input_row, indices2 + input_row, scores1 + output_row, indices1 + output_row,
          partitions1, partition, storage);
    }
    grid.sync();
  }

  if constexpr (Factor3 > 1) {
    const int partitions3 = DivUp(partitions2, Factor3);
    if (partition < partitions3) {
      const size_t input_row = static_cast<size_t>(row) * partitions2 * K;
      const size_t output_row = static_cast<size_t>(row) * partitions3 * K;
      ReducePartitions<BlockSize, K * Factor3, K>(
          scores1 + input_row, indices1 + input_row, scores2 + output_row, indices2 + output_row,
          partitions2, partition, storage);
    }
  }
}

template <typename T>
__global__ void WriteOutput(const float* scores,
                            const int* indices,
                            T* output_scores,
                            int64_t* output_indices,
                            int rows,
                            int k,
                            int stride) {
  const int row = blockIdx.x;
  if (row < rows && threadIdx.x < k) {
    const size_t input_offset = static_cast<size_t>(row) * stride + threadIdx.x;
    const size_t output_offset = static_cast<size_t>(row) * k + threadIdx.x;
    output_scores[output_offset] = static_cast<T>(scores[input_offset]);
    output_indices[output_offset] = indices[input_offset];
  }
}

template <int K, int BlockSize>
void* ReductionKernel(const ReductionFactors& factors) {
  if (factors.factor1 == 8 && factors.factor2 == 8 && factors.factor3 == 4)
    return reinterpret_cast<void*>(CooperativeReduce<K, BlockSize, 8, 8, 4>);
  if (factors.factor1 == 8 && factors.factor2 == 8 && factors.factor3 == 2)
    return reinterpret_cast<void*>(CooperativeReduce<K, BlockSize, 8, 8, 2>);
  if (factors.factor1 == 8 && factors.factor2 == 8)
    return reinterpret_cast<void*>(CooperativeReduce<K, BlockSize, 8, 8, 1>);
  if (factors.factor1 == 8 && factors.factor2 == 4)
    return reinterpret_cast<void*>(CooperativeReduce<K, BlockSize, 8, 4, 1>);
  if (factors.factor1 == 8)
    return reinterpret_cast<void*>(CooperativeReduce<K, BlockSize, 8, 1, 1>);
  if (factors.factor1 == 16)
    return reinterpret_cast<void*>(CooperativeReduce<K, BlockSize, 16, 1, 1>);
  if (factors.factor1 == 4)
    return reinterpret_cast<void*>(CooperativeReduce<K, BlockSize, 4, 1, 1>);
  return reinterpret_cast<void*>(CooperativeReduce<K, BlockSize, 1, 1, 1>);
}

template <int K>
bool CheckCooperativeSupport(const CudaKernel* kernel, int rows, int num_partitions) {
  const ReductionFactors factors = GetReductionFactors(num_partitions);
  if (factors.steps == 0) {
    return true;
  }
  constexpr int block_size = ReductionBlockSize<K>();
  void* reduction_kernel = ReductionKernel<K, block_size>(factors);
  int blocks_per_sm = 0;
  const cudaError_t result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks_per_sm, reduction_kernel, block_size, 0);
  if (result != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  const int grid_x = DivUp(num_partitions, factors.factor1);
  return static_cast<int64_t>(rows) * grid_x <=
         static_cast<int64_t>(blocks_per_sm) * kernel->GetDeviceProp().multiProcessorCount;
}

inline bool IsSupported(const CudaKernel* kernel, int64_t rows, int64_t dimension, int64_t k) {
  if (rows <= 0 || rows > kernel->GetDeviceProp().maxGridSize[1] ||
      dimension <= 0 || dimension > std::numeric_limits<int>::max() || k <= 0 || k > kMaxK) {
    return false;
  }
  const int partition_size = EstimateBestPartitionSize(static_cast<int>(dimension));
  const int num_partitions = DivUp(static_cast<int>(dimension), partition_size);
  if (num_partitions > kMaxPartitions) {
    return false;
  }
  switch (PaddedK(static_cast<int>(k))) {
    case 4:
      return CheckCooperativeSupport<4>(kernel, static_cast<int>(rows), num_partitions);
    case 8:
      return CheckCooperativeSupport<8>(kernel, static_cast<int>(rows), num_partitions);
    case 16:
      return CheckCooperativeSupport<16>(kernel, static_cast<int>(rows), num_partitions);
    case 32:
      return CheckCooperativeSupport<32>(kernel, static_cast<int>(rows), num_partitions);
    case 64:
      return CheckCooperativeSupport<64>(kernel, static_cast<int>(rows), num_partitions);
    case 128:
      return CheckCooperativeSupport<128>(kernel, static_cast<int>(rows), num_partitions);
    default:
      return CheckCooperativeSupport<256>(kernel, static_cast<int>(rows), num_partitions);
  }
}

template <typename T, int PartitionSize, int K>
void LaunchStage1(const T* input, float* scores, int* indices,
                  int dimension, int rows, int num_partitions, cudaStream_t stream) {
  FindPartitionTopK<T, PartitionSize, K><<<dim3(num_partitions, rows), kStage1Threads, 0, stream>>>(
      input, scores, indices, dimension, num_partitions);
}

template <typename T, int K>
Status Launch(const CudaKernel* kernel,
              cudaStream_t stream,
              void* alloc_stream,
              const T* input,
              T* output_scores,
              int64_t* output_indices,
              int rows,
              int dimension,
              int k,
              int partition_size) {
  const int num_partitions = DivUp(dimension, partition_size);
  const size_t elements = SafeInt<size_t>(rows) * num_partitions * K;
  auto scores1 = kernel->GetScratchBuffer<float>(elements, alloc_stream);
  auto indices1 = kernel->GetScratchBuffer<int>(elements, alloc_stream);
  auto scores2 = kernel->GetScratchBuffer<float>(elements, alloc_stream);
  auto indices2 = kernel->GetScratchBuffer<int>(elements, alloc_stream);

  if (partition_size == kPartitionSizes[0])
    LaunchStage1<T, kPartitionSizes[0], K>(input, scores1.get(), indices1.get(), dimension, rows, num_partitions, stream);
  else if (partition_size == kPartitionSizes[1])
    LaunchStage1<T, kPartitionSizes[1], K>(input, scores1.get(), indices1.get(), dimension, rows, num_partitions, stream);
  else if (partition_size == kPartitionSizes[2])
    LaunchStage1<T, kPartitionSizes[2], K>(input, scores1.get(), indices1.get(), dimension, rows, num_partitions, stream);
  else
    LaunchStage1<T, kPartitionSizes[3], K>(input, scores1.get(), indices1.get(), dimension, rows, num_partitions, stream);
  CUDA_RETURN_IF_ERROR(cudaGetLastError());

  const ReductionFactors factors = GetReductionFactors(num_partitions);
  if (factors.steps > 0) {
    constexpr int block_size = ReductionBlockSize<K>();
    void* reduction_kernel = ReductionKernel<K, block_size>(factors);
    float* scores1_ptr = scores1.get();
    int* indices1_ptr = indices1.get();
    float* scores2_ptr = scores2.get();
    int* indices2_ptr = indices2.get();
    int num_partitions_arg = num_partitions;
    void* args[] = {&indices1_ptr, &scores1_ptr, &indices2_ptr, &scores2_ptr, &num_partitions_arg};
    CUDA_RETURN_IF_ERROR(cudaLaunchCooperativeKernel(
        reduction_kernel, dim3(DivUp(num_partitions, factors.factor1), rows),
        dim3(block_size), args, 0, stream));
  }

  const float* final_scores = (factors.steps == 1 || factors.steps == 3) ? scores2.get() : scores1.get();
  const int* final_indices = (factors.steps == 1 || factors.steps == 3) ? indices2.get() : indices1.get();
  WriteOutput<T><<<rows, K, 0, stream>>>(
      final_scores, final_indices, output_scores, output_indices, rows, k, K);
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
inline Status Run(const CudaKernel* kernel,
                  cudaStream_t stream,
                  void* alloc_stream,
                  const T* input,
                  T* output_scores,
                  int64_t* output_indices,
                  int rows,
                  int dimension,
                  int k) {
  const int partition_size = EstimateBestPartitionSize(dimension);
  switch (PaddedK(k)) {
    case 4:
      return Launch<T, 4>(kernel, stream, alloc_stream, input, output_scores, output_indices,
                          rows, dimension, k, partition_size);
    case 8:
      return Launch<T, 8>(kernel, stream, alloc_stream, input, output_scores, output_indices,
                          rows, dimension, k, partition_size);
    case 16:
      return Launch<T, 16>(kernel, stream, alloc_stream, input, output_scores, output_indices,
                           rows, dimension, k, partition_size);
    case 32:
      return Launch<T, 32>(kernel, stream, alloc_stream, input, output_scores, output_indices,
                           rows, dimension, k, partition_size);
    case 64:
      return Launch<T, 64>(kernel, stream, alloc_stream, input, output_scores, output_indices,
                           rows, dimension, k, partition_size);
    case 128:
      return Launch<T, 128>(kernel, stream, alloc_stream, input, output_scores, output_indices,
                            rows, dimension, k, partition_size);
    default:
      return Launch<T, 256>(kernel, stream, alloc_stream, input, output_scores, output_indices,
                            rows, dimension, k, partition_size);
  }
}

}  // namespace hybrid_topk
}  // namespace cuda
}  // namespace onnxruntime