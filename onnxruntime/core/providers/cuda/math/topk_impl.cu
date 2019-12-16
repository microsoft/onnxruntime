// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "cub/cub.cuh"
#include <limits>

namespace onnxruntime {
namespace cuda {

/*
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
*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct KV {
  T key;
  int64_t val;
};

#define AT(idx) (left_dim + idx * mid_dim + right_dim)
#define TRIVIAL (1 == largest ? type_min : type_max)
/*
#define BIGGER(n, m) (n.key > m.key ? n : (n.key < m.key ? m : (n.val > m.val ? m : n)))
#define SMALLER(n, m) (n.key < m.key ? n : (n.key > m.key ? m : (n.val < m.val ? m : n)))
#define IS_SMALLER(n, m) (n.key < m.key || !(n.key > m.key) && n.val > m.val)
*/
#define BIGGER(n, m) (n.key > m.key ? n : (n.key < m.key ? m : (n.val > m.val ? (1 == largest ? m : n) : (1 == largest ? n : m))))
#define SMALLER(n, m) (n.key < m.key ? n : (n.key > m.key ? m : (n.val < m.val ? (1 == largest ? m : n) : (1 == largest ? n : m))))
#define IS_SMALLER(n, m) (n.key < m.key || !(n.key > m.key) && (1 == largest ? n.val > m.val : n.val < m.val))
#define MIN(n, m) (n <= m ? n : m)

template <typename T>
__global__ void BitonicSortMerge(const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t aligned_K, int64_t largest, int64_t sorted, int64_t dimension, int64_t aligned_dimension, T type_min, T type_max) {
  auto tid = threadIdx.x;
  auto thread_start_at = tid << 1;
  if (thread_start_at >= aligned_dimension) {
    return;
  }
  auto bid = blockIdx.x;
  auto mid_dim = axis == size - 1 ? 1 : elem_nums[axis + 1];
  auto left_dim = bid / mid_dim * elem_nums[axis];
  auto right_dim = axis == size - 1 ? 0 : bid % elem_nums[axis + 1];
  auto elem_per_block = blockDim.x << 1;  //each thread deals with two elems
  extern __shared__ char shared[];
  auto shared_mem = (KV<T>*)shared;
  auto topK_result = shared_mem + elem_per_block;  //keep current topK from loop
  int64_t offset = 0;
  while (offset < aligned_dimension) {
    //sort with bitonic len is 1 based on input
    auto shared_index = thread_start_at;
    auto shared_index_next = shared_index + 1;
    auto input_index = offset + shared_index;
    auto input_index_next = offset + shared_index_next;
    auto x0 = input_index < dimension ? input_x[AT(input_index)] : TRIVIAL;
    auto x1 = input_index_next < dimension ? input_x[AT(input_index_next)] : TRIVIAL;
    auto reverse = (2 & shared_index) == 0;
    auto swap = reverse ^ (x0 < x1);
    if (swap) {
      shared_mem[shared_index].key = x1;
      shared_mem[shared_index].val = input_index_next;
      shared_mem[shared_index + 1].key = x0;
      shared_mem[shared_index + 1].val = input_index;
    } else {
      shared_mem[shared_index].key = x0;
      shared_mem[shared_index].val = input_index;
      shared_mem[shared_index + 1].key = x1;
      shared_mem[shared_index + 1].val = input_index_next;
    }
    //sort with bitonic len is above 1 based on shared
    for (int64_t len = 2; len < aligned_K; len <<= 1) {
      auto dir = len << 1;
      for (auto inc = len; inc > 0; inc >>= 1) {
        auto low = tid & (inc - 1);
        shared_index = (tid << 1) - low;
        auto shared_index_next = shared_index + inc;
        reverse = (dir & shared_index) == 0;
        swap = reverse ^ IS_SMALLER(shared_mem[shared_index], shared_mem[shared_index_next]);
        if (swap) {
          auto tmp_kv = shared_mem[shared_index];
          shared_mem[shared_index] = shared_mem[shared_index_next];
          shared_mem[shared_index_next] = tmp_kv;
        }
      }  //for
    }    //for
    __syncthreads();
    /*
    auto num_of_K = elem_per_block / aligned_K;
    if (aligned_dimension < elem_per_block) {
      num_of_K = aligned_dimension / aligned_K + ((aligned_dimension % aligned_K) == 0 ? 0 : 1);
    }
    num_of_K >>= 1;
    */
    auto num_of_K = MIN((elem_per_block / aligned_K), (aligned_dimension / aligned_K));
    int64_t merge_step = 0;
    while ((1 << merge_step) < num_of_K) {
      //merge
      auto low = tid & (aligned_K - 1);
      auto base_left_elem_index = (tid << 1) - low;
      auto left_elem_index = (base_left_elem_index / aligned_K << merge_step) * aligned_K + (base_left_elem_index & (aligned_K - 1));
      auto right_elem_index = left_elem_index + (aligned_K << merge_step);
      if (right_elem_index < aligned_dimension) {
        shared_mem[left_elem_index] = 1 == largest ? BIGGER(shared_mem[left_elem_index], shared_mem[right_elem_index]) : SMALLER(shared_mem[left_elem_index], shared_mem[right_elem_index]);
      }
      __syncthreads();
      ++merge_step;
      //rebuild
      int64_t len = aligned_K >> 1;
      auto dir = len << 1;
      for (auto inc = len; inc > 0; inc >>= 1) {
        auto low = tid & (inc - 1);
        auto base_shared_index = (tid << 1) - low;
        shared_index = (base_shared_index / aligned_K << merge_step) * aligned_K +(base_shared_index & (aligned_K - 1));
        auto shared_index_next = shared_index + inc;
        // if (shared_index_next < (aligned_dimension << merge_step)) {
        if (shared_index_next < aligned_dimension) {
          reverse = (dir & base_shared_index) == 0;
          swap = reverse ^ IS_SMALLER(shared_mem[shared_index], shared_mem[shared_index_next]);
          if (swap) {
            auto tmp_kv = shared_mem[shared_index];
            shared_mem[shared_index] = shared_mem[shared_index_next];
            shared_mem[shared_index_next] = tmp_kv;
          }
        }
      }  //for
      __syncthreads();
    }  //while
    //save local top K
    __syncthreads();
    if (0 == offset && tid == 0) {
      for (int64_t i = 0; i < aligned_K; ++i) {
        topK_result[i] = shared_mem[i];
      }
    } else if (0 < offset && tid < aligned_K) {
      auto saved_index = tid;
      auto share_index = aligned_K - saved_index - 1;
      topK_result[saved_index] = 1 == largest ? BIGGER(topK_result[saved_index], shared_mem[share_index]) : SMALLER(topK_result[saved_index], shared_mem[share_index]);
      __syncthreads();
      //rebuild topk result ascending
      int64_t len = aligned_K >> 1;
      for (auto inc = len; inc > 0; inc >>= 1) {
        auto low = tid & (inc - 1);
        shared_index = (tid << 1) - low;
        auto shared_index_next = shared_index + inc;
        if (shared_index_next < aligned_K) {
          if (topK_result[shared_index].key > topK_result[shared_index_next].key) {
            auto tmp_kv = topK_result[shared_index];
            topK_result[shared_index] = topK_result[shared_index_next];
            topK_result[shared_index_next] = tmp_kv;
          }
        }
      }  //for
    }    //else if
    offset += elem_per_block;
  }  //while
  __syncthreads();
  //save global top K
  if (1 == sorted) {
    /*
    if (1 == largest && tid >= aligned_K - K && tid < aligned_K) {
      auto target_index = tid - aligned_K + K;
      auto reversed_index = aligned_K - target_index - 1;
      auto output_offset = left_dim * K / dimension + target_index * mid_dim + right_dim;
      output_v[output_offset] = topK_result[reversed_index].key;
      output_i[output_offset] = topK_result[reversed_index].val;
      */
    if (1 == largest && tid == 0) {
      for (int64_t i = 0; i < K; ++i) {
        auto output_offset = left_dim * K / dimension + i * mid_dim + right_dim;
        output_v[output_offset] = topK_result[aligned_K - i - 1].key;
        output_i[output_offset] = topK_result[aligned_K - i - 1].val;
      }
    } else if (0 == largest && tid == 0) {
      for (int64_t i = 0; i < K; ++i) {
        auto output_offset = left_dim * K / dimension + i * mid_dim + right_dim;
        output_v[output_offset] = topK_result[i].key;
        output_i[output_offset] = topK_result[i].val;
      }
    }
  } else {
    if (tid == 0) {
      if (1 == largest) {
        for (int64_t i = 0; i < aligned_K - K; ++i) {
          topK_result[i].val = dimension;
        }
      } else {
        for (int64_t i = K; i < aligned_K; ++i) {
          topK_result[i].val = dimension;
        }
      }
    }
    __syncthreads();
    //sort by index ascending
    int64_t len = aligned_K >> 1;
    for (auto inc = len; inc > 0; inc >>= 1) {
      auto low = tid & (inc - 1);
      auto shared_index = (tid << 1) - low;
      auto shared_index_next = shared_index + inc;
      if (shared_index_next < aligned_K) {
        if (topK_result[shared_index].val > topK_result[shared_index_next].val) {
          auto tmp_kv = topK_result[shared_index];
          topK_result[shared_index] = topK_result[shared_index_next];
          topK_result[shared_index_next] = tmp_kv;
        }
      }
    }  //for
    __syncthreads();
    if (tid == 0) {
      for (int64_t i = 0; i < K; ++i) {
        auto output_offset = left_dim * K / dimension + i * mid_dim + right_dim;
        output_v[output_offset] = topK_result[i].key;
        output_i[output_offset] = topK_result[i].val;
      }
    }
  }  //else
}  //BitonicSortMerge

template <typename T>
Status TopKImpl(const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t N, int64_t dimension) {
  auto aligned_K = static_cast<int64_t>(pow(2, ceil(log2(K))));
  auto aligned_dimension = static_cast<int64_t>(pow(2, ceil(log2(dimension))));
  BitonicSortMerge<T><<<N, GridDim::maxThreadsPerBlock, sizeof(KV<T>) * (aligned_K + (GridDim::maxThreadsPerBlock << 1))>>>(input_x, output_v, output_i, elem_nums, size, axis, K, aligned_K, largest, sorted, dimension, aligned_dimension, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  return Status::OK();
}

#define TOPKIMPLE(T) template Status TopKImpl<T>(const T* input_x,         \
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

/*
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

template <typename TK, typename TV, bool reversed_compare = false>
struct KV {
  TK key;
  TV val;
  __device__ bool operator<(const KV& kv) const {
    return reversed_compare ? key > kv.key : key < kv.key;
  }
  __device__ bool operator>(const KV& kv) const {
    return reversed_compare ? key < kv.key : key > kv.key;
  }
};


using KV_uint8_int64 = KV<uint8_t, int64_t>;
bool KV_uint8_int64::reversed_compare;
using KV_uint16_int64 = KV<uint16_t, int64_t>;
bool KV_uint16_int64::reversed_compare;
using KV_uint32_int64 = KV<uint32_t, int64_t>;
bool KV_uint32_int64::reversed_compare;
using KV_uint64_int64 = KV<uint64_t, int64_t>;
bool KV_uint64_int64::reversed_compare;
using KV_int8_int64 = KV<int8_t, int64_t>;
bool KV_int8_int64::reversed_compare;
using KV_int16_int64 = KV<int16_t, int64_t>;
bool KV_int16_int64::reversed_compare;
using KV_int32_int64 = KV<int32_t, int64_t>;
bool KV_int32_int64::reversed_compare;
using KV_int64_int64 = KV<int64_t, int64_t>;
bool KV_int64_int64::reversed_compare;
using KV_float_int64 = KV<float, int64_t>;
bool KV_float_int64::reversed_compare;
using KV_double_int64 = KV<double, int64_t>;
bool KV_double_int64::reversed_compare;

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
__global__ void FillInput2(const T* input, KV<T, int64_t>* output, const int64_t* elem_nums, size_t size, int64_t axis, int64_t offset, int64_t dimension, int64_t aligned_dimension, T default_key, int64_t default_val) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, aligned_dimension);
  if (id < dimension) {
    auto left = offset / (axis == size - 1 ? 1 : elem_nums[axis + 1]) * elem_nums[axis];
    auto right = axis == size - 1 ? 0 : offset % elem_nums[axis + 1];
    auto input_offset = left + id * (axis == size - 1 ? 1 : elem_nums[axis + 1]) + right;
    output.key = input[input_offset];
    output.val = id;
  } else {
    output.key = default_key;
    output.val = default_val;
  }
}

template <typename T>
Status TopKImpl2(const CudaKernel* kernel, const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t N, int64_t dimension) {
  using KVT = KV<T, int64_t>;
  auto aligned_dimension = static_cast<int64_t>(pow(2, ceil(log2(dimension))));
  auto aligned_K = static_cast<int64_t>(pow(2, ceil(log2(K))));
  auto buffer_object = kernel->GetScratchBuffer<KVT>(aligned_dimension);
  auto blocksPerGridD = (int)(ceil(static_cast<float>(aligned_dimension) / GridDim::maxThreadsPerBlock));
  auto blocksPerGridK = (int)(ceil(static_cast<float>(aligned_K) / GridDim::maxThreadsPerBlock));
  for (int64_t i = 0; i < N; i++) {
    // FillInput2<T><<<blocksPerGridD, GridDim::maxThreadsPerBlock, 0>>>(input_x, static_cast<KVT*>(buffer_object.get()), elem_nums, size, axis, i, dimension, aligned_dimension, 1 == largest ? std::numeric_limits<T>::min() : std::numeric_limits<T>::max(), std::numeric_limits<int64_t>::max);
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
*/

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cuda
}  // namespace onnxruntime