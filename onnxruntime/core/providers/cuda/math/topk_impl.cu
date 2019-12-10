// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "cub/cub.cuh"
#include <limits>

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void FillInput(const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t offset, int64_t dimension) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, dimension);
  auto left = offset / (axis == size - 1 ? 1 : elem_nums[axis + 1]) * elem_nums[axis];
  auto right = axis == size - 1 ? 0 : offset % elem_nums[axis + 1];
  auto input_offset = left + id * (axis == size - 1 ? 1 : elem_nums[axis + 1]) + right;
  output_v[id] = input_x[input_offset];
  output_i[id] = id;
}

template <typename T>
__global__ void FillOutput(const T* input_v, const int64_t* input_i, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t offset, int64_t dimension) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, K);
  auto left = offset / (axis == size - 1 ? 1 : elem_nums[axis + 1]) * elem_nums[axis] * K / dimension;
  auto right = axis == size - 1 ? 0 : offset % elem_nums[axis + 1];
  auto output_offset = left + id * (axis == size - 1 ? 1 : elem_nums[axis + 1]) + right;
  output_v[output_offset] = input_v[id];
  output_i[output_offset] = input_i[id];
}

__global__ void ExcludeOutput(int64_t* output_i, int64_t K, int64_t dimension) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, dimension);
  if (id >= K) {
    output_i[id] = dimension;
  }
}

template <typename T>
Status TopKImpl(const CudaKernel* kernel, const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t N, int64_t dimension) {
  auto input_key_buffer = kernel->GetScratchBuffer<T>(dimension);
  auto output_key_buffer = kernel->GetScratchBuffer<T>(dimension);
  auto input_value_buffer = kernel->GetScratchBuffer<int64_t>(dimension);
  auto output_value_buffer = kernel->GetScratchBuffer<int64_t>(dimension);
  auto input_key = input_key_buffer.get();
  auto output_key = output_key_buffer.get();
  auto input_value = input_value_buffer.get();
  auto output_value = output_value_buffer.get();
  size_t temp_bytes = 0;
  CUDA_RETURN_IF_ERROR(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, input_key, output_key, input_value, output_value, dimension));
  auto temp_storage_buffer = kernel->GetScratchBuffer<char>(temp_bytes);
  auto temp_storage = temp_storage_buffer.get();
  auto blocksPerGridD = (int)(ceil(static_cast<float>(dimension) / GridDim::maxThreadsPerBlock));
  auto blocksPerGridK = (int)(ceil(static_cast<float>(K) / GridDim::maxThreadsPerBlock));
  for (int64_t i = 0; i < N; i++) {
    FillInput<T><<<blocksPerGridD, GridDim::maxThreadsPerBlock, 0>>>(input_x, input_key, input_value, elem_nums, size, axis, K, i, dimension);
    CUDA_RETURN_IF_ERROR(1 == largest ? cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_bytes, input_key, output_key, input_value, output_value, dimension) : cub::DeviceRadixSort::SortPairs(temp_storage, temp_bytes, input_key, output_key, input_value, output_value, dimension));
    if (1 == sorted) {
      FillOutput<T><<<blocksPerGridK, GridDim::maxThreadsPerBlock, 0>>>(output_key, output_value, output_v, output_i, elem_nums, size, axis, K, i, dimension);
    } else {  //reorder by ascending index
      ExcludeOutput<<<blocksPerGridD, GridDim::maxThreadsPerBlock, 0>>>(output_value, K, dimension);
      CUDA_RETURN_IF_ERROR(cub::DeviceRadixSort::SortPairs(temp_storage, temp_bytes, output_value, input_value, output_key, input_key, dimension));
      FillOutput<T><<<blocksPerGridK, GridDim::maxThreadsPerBlock, 0>>>(input_key, input_value, output_v, output_i, elem_nums, size, axis, K, i, dimension);
    }
  }
  return Status::OK();
}

#define TOPKIMPLE(T) template Status TopKImpl<T>(const CudaKernel* kernel, \
                                                 const T* input_x,         \
                                                 T* output_v,              \
                                                 int64_t* output_i,        \
                                                 const int64_t* elem_nums, \
                                                 size_t size,              \
                                                 int64_t axis,             \
                                                 int64_t K,                \
                                                 int64_t largest,          \
                                                 int64_t sorted,           \
                                                 int64_t N,                \
                                                 int64_t dimension)

TOPKIMPLE(uint8_t);
TOPKIMPLE(uint16_t);
TOPKIMPLE(uint32_t);
TOPKIMPLE(uint64_t);
TOPKIMPLE(int8_t);
TOPKIMPLE(int16_t);
TOPKIMPLE(int32_t);
TOPKIMPLE(int64_t);
TOPKIMPLE(float);
TOPKIMPLE(double);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// buffer has N elements
// each block deals with blockDim.x * K elements start from blockDim.x * blockIdx.x
// each thread deals K elements start from blockDim.x * blockIdx.x + threadIdx.x
// top K elements will be saved to buffer at blockDim.x * blockIdx.x
template <typename T>
__global__ void BitonicSortMerge(T* buffer, int64_t N, int64_t K) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  __shared__ T* shared_mem;  // blockDim.x * K * sizeof(T)
  auto A = shared_mem + threadIdx.x;
  memcpy(A, buffer + id, K * sizeof(T));
  //sort
  for (int64_t len = 1; len < K; len <<= 1) {
    auto len_dir = len << 1;
    for (auto inc = len; inc > 0; inc >>= 1) {
      auto inc_dir = inc << 1;
      for (int64_t i = 0; i <= K - inc_dir; i += inc_dir) {
        auto ascending = i / len_dir % 2 == 0;
        for (auto j = i; j < i + inc; ++j) {
          auto k = j + inc;
          if (ascending && A[j] > A[k] || !ascending && A[j] < A[k]) {
            auto a = A[j];
            A[j] = A[k];
            A[k] = a;
          }
        }
      }
    }
  }
  __syncthreads();
  //merge
  int64_t step = 0;
  auto ascending = threadIdx.x % 2 == 0;
  while ((1 << step) <= blockDim.x) {
    auto left_K_index = (1 << step) * (1 + (threadIdx.x << 1));
    auto right_K_index = left_K_index + (1 << step);
    if (right_K_index < blockDim.x) {
      auto left_K = shared_mem + left_K_index;
      auto right_K = shared_mem + right_K_index;
      //merge bigger into right_K
      for (int64_t i = 0; i < K; ++i) {
        if (left_K[i] > right_K[i]) {
          right_K[i] = left_K[i];
        }
      }
      //sort right_K
      for (auto inc = K >> 1; inc > 0; inc >>= 1) {
        auto inc_dir = inc << 1;
        for (int64_t i = 0; i <= K - inc_dir; i += inc_dir) {
          for (auto j = i; j < i + inc; ++j) {
            auto k = j + inc;
            if (ascending && right_K[j] > right_K[k] || !ascending && right_K[j] < right_K[k]) {
              auto tmp = right_K[j];
              right_K[j] = right_K[k];
              right_K[k] = tmp;
            }
          }
        }
      }
    }  //if right_K_...
    __syncthreads();
    ++step;
  }
  __syncthreads();
  //save
  if (threadIdx.x == 0) {
    auto final_K_index = 1 << step;
    memcpy(buffer + blockDim.x * blockIdx.x, shared_mem + final_K_index, K * sizeof(T));
  }
}

template <typename TK, typename TV>
struct KV {
  TK key;
  TV val;
  __device__ bool operator<(const KV& kv) const {
    return key < kv.key;
  }
  __device__ bool operator>(const KV& kv) const {
    return key > kv.key;
  }
};

using KV_uint8_int64 = KV<uint8_t, int64_t>;
using KV_uint16_int64 = KV<uint16_t, int64_t>;
using KV_uint32_int64 = KV<uint32_t, int64_t>;
using KV_uint64_int64 = KV<uint64_t, int64_t>;
using KV_int8_int64 = KV<int8_t, int64_t>;
using KV_int16_int64 = KV<int16_t, int64_t>;
using KV_int32_int64 = KV<int32_t, int64_t>;
using KV_int64_int64 = KV<int64_t, int64_t>;
using KV_float_int64 = KV<float, int64_t>;
using KV_double_int64 = KV<double, int64_t>;

template void __global__ BitonicSortMerge<KV_uint8_int64>(KV_uint8_int64* buffer, int64_t N, int64_t K);
template void __global__ BitonicSortMerge<KV_uint16_int64>(KV_uint16_int64* buffer, int64_t N, int64_t K);
template void __global__ BitonicSortMerge<KV_uint32_int64>(KV_uint32_int64* buffer, int64_t N, int64_t K);
template void __global__ BitonicSortMerge<KV_uint64_int64>(KV_uint64_int64* buffer, int64_t N, int64_t K);
template void __global__ BitonicSortMerge<KV_int8_int64>(KV_int8_int64* buffer, int64_t N, int64_t K);
template void __global__ BitonicSortMerge<KV_int16_int64>(KV_int16_int64* buffer, int64_t N, int64_t K);
template void __global__ BitonicSortMerge<KV_int32_int64>(KV_int32_int64* buffer, int64_t N, int64_t K);
template void __global__ BitonicSortMerge<KV_int64_int64>(KV_int64_int64* buffer, int64_t N, int64_t K);
template void __global__ BitonicSortMerge<KV_float_int64>(KV_float_int64* buffer, int64_t N, int64_t K);
template void __global__ BitonicSortMerge<KV_double_int64>(KV_double_int64* buffer, int64_t N, int64_t K);

template <typename T>
__global__ void FillInput2(const T* input, KV<T, int64_t> output, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t offset, int64_t dimension) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, dimension);
  auto left = offset / (axis == size - 1 ? 1 : elem_nums[axis + 1]) * elem_nums[axis];
  auto right = axis == size - 1 ? 0 : offset % elem_nums[axis + 1];
  auto input_offset = left + id * (axis == size - 1 ? 1 : elem_nums[axis + 1]) + right;
  output.key = input_x[input_offset];
  output.val = id;
}

template<typename T>
Status TopKImpl2(const CudaKernel* kernel, const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t N, int64_t dimension) {
  using KVT = KV<T, int64_t>;
  auto buffer = kernel->GetScratchBuffer<KVT>(dimension);
  auto blocksPerGridD = (int)(ceil(static_cast<float>(dimension) / GridDim::maxThreadsPerBlock));
  auto blocksPerGridK = (int)(ceil(static_cast<float>(K) / GridDim::maxThreadsPerBlock));
  for (int64_t i = 0; i < N; i++) {
    FillInput2<T><<<blocksPerGridD, GridDim::maxThreadsPerBlock, 0>>>(input_x, input_key, input_value, elem_nums, size, axis, K, i, dimension);
  }
}

#define TOPKIMPLE2(T) template Status TopKImpl2<T>(const CudaKernel* kernel, \
                                                   const T* input_x,         \
                                                   T* output_v,              \
                                                   int64_t* output_i,        \
                                                   const int64_t* elem_nums, \
                                                   size_t size,              \
                                                   int64_t axis,             \
                                                   int64_t K,                \
                                                   int64_t largest,          \
                                                   int64_t sorted,           \
                                                   int64_t N,                \
                                                   int64_t dimension)

TOPKIMPLE2(uint8_t);
TOPKIMPLE2(uint16_t);
TOPKIMPLE2(uint32_t);
TOPKIMPLE2(uint64_t);
TOPKIMPLE2(int8_t);
TOPKIMPLE2(int16_t);
TOPKIMPLE2(int32_t);
TOPKIMPLE2(int64_t);
TOPKIMPLE2(float);
TOPKIMPLE2(double);

}  // namespace cuda
}  // namespace onnxruntime