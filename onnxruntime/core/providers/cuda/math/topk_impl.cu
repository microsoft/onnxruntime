// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "cub/cub.cuh"
#include <limits>

namespace onnxruntime {
namespace cuda {

template <typename T>
struct KV {
  T key;
  int64_t val;
};

#define AT(idx) (left_dim + idx * mid_dim + right_dim)
#define TRIVIAL (1 == largest ? type_min : type_max)
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

}  // namespace cuda
}  // namespace onnxruntime