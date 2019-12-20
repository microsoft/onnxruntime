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

#define FROM(idx) (left_dim + (idx)*mid_dim + right_dim)
#define TO(idx) (left_dim * K / dimension + (idx)*mid_dim + right_dim)
#define TRIVIAL (1 == largest ? type_min : type_max)
#define BIGGER(n, m) (n.key > m.key ? n : (n.key < m.key ? m : (n.val > m.val ? (1 == largest ? m : n) : (1 == largest ? n : m))))
#define SMALLER(n, m) (n.key < m.key ? n : (n.key > m.key ? m : (n.val < m.val ? (1 == largest ? m : n) : (1 == largest ? n : m))))
#define IS_SMALLER(n, m) (n.key < m.key || !(n.key > m.key) && (1 == largest ? n.val > m.val : n.val < m.val))
#define MAX(n, m) ((n) >= (m) ? (n) : (m))
#define MIN(n, m) ((n) <= (m) ? (n) : (m))

template <typename T>
__global__ void BitonicTopK(const T* X, T* V, int64_t* I, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t aligned_K, int64_t largest, int64_t sorted, int64_t dimension, int64_t aligned_dimension, T type_min, T type_max) {
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  extern __shared__ char shared_mem[];
  auto S = (KV<T>*)(shared_mem);
  auto mid_dim = axis == size - 1 ? 1 : elem_nums[axis + 1];
  auto left_dim = bid / mid_dim * elem_nums[axis];
  auto right_dim = axis == size - 1 ? 0 : bid % elem_nums[axis + 1];
  //copy x to shared memory
  for (auto i = 0; i < 2; ++i) {
    auto j = (tid << 1) + i;
    if (j < aligned_dimension) {
      S[j].key = j < dimension ? X[FROM(j)] : TRIVIAL;
      S[j].val = j;
    }
  }
  __syncthreads();
  //sort each K
  for (int64_t len = 1; len < aligned_K; len <<= 1) {
    auto dir = len << 1;
    for (auto inc = len; inc > 0; inc >>= 1) {
      auto low = tid & (inc - 1);
      auto i = (tid << 1) - low;
      auto j = i + inc;
      if (j < aligned_dimension) {
        auto reverse = (dir & i) == 0;
        auto swap = reverse ^ IS_SMALLER(S[i], S[j]);
        if (swap) {
          auto tmp = S[i];
          S[i] = S[j];
          S[j] = tmp;
        }
      }
    }
  }
  __syncthreads();
  //merge and rebuild K
  for (int64_t len = aligned_K; len < aligned_dimension; len <<= 1) {
    auto dir = len << 1;
    auto i = (tid << 1) - (tid & (len - 1));
    auto j = i + len;
    if (i % dir < aligned_K && j < aligned_dimension) {
      S[i] = 1 == largest ? BIGGER(S[i], S[j]) : SMALLER(S[i], S[j]);
    }
    __syncthreads();
    for (auto inc = aligned_K >> 1; inc > 0; inc >>= 1) {
      auto ii = (tid << 1) - (tid & (inc - 1));
      auto jj = ii + inc;
      if (ii % dir < aligned_K && jj < aligned_dimension) {
        auto reverse = (dir & ii) == 0;
        auto swap = reverse ^ IS_SMALLER(S[ii], S[jj]);
        if (swap) {
          auto tmp = S[ii];
          S[ii] = S[jj];
          S[jj] = tmp;
        }
      }
    }
    __syncthreads();
  }
  //save top K
  if (1 == sorted) {
    if (1 == largest) {
      auto start = aligned_K - K;
      if (tid >= start && tid < aligned_K) {
        auto to = TO(aligned_K - 1 - tid);
        V[to] = S[tid].key;
        I[to] = S[tid].val;
      }
    } else {
      if (tid < K) {
        auto to = TO(tid);
        V[to] = S[tid].key;
        I[to] = S[tid].val;
      }
    }
  } else {
    if (1 == largest) {
      auto start = aligned_K - K;
      if (tid < start) {
        S[tid].val = aligned_dimension;
      }
    } else {
      if (tid >= K && tid < aligned_K) {
        S[tid].val = aligned_dimension;
      }
    }
    __syncthreads();
    //sort by index ascending
    for (int64_t len = 1; len < aligned_K; len <<= 1) {
      for (int64_t inc = len; inc > 0; inc >>= 1) {
        auto low = tid & (inc - 1);
        auto i = (tid << 1) - low;
        auto j = i + inc;
        if (j < aligned_K && S[i].val > S[j].val) {
          auto tmp = S[i];
          S[i] = S[j];
          S[j] = tmp;
        }
      }
    }
    __syncthreads();
    if (tid < K) {
      auto to = TO(tid);
      V[to] = S[tid].key;
      I[to] = S[tid].val;
    }
  }
}

template <typename T>
__global__ void RadixTopK(const T* X, T* V, int64_t* I, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t dimension, int64_t XPT, int64_t bits, bool integer) {
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  extern __shared__ char shared_mem[];
  auto S32 = (int32_t*)(shared_mem);
  auto S64 = (int64_t*)(shared_mem);
  auto mid_dim = axis == size - 1 ? 1 : elem_nums[axis + 1];
  auto left_dim = bid / mid_dim * elem_nums[axis];
  auto right_dim = axis == size - 1 ? 0 : bid % elem_nums[axis + 1];
  T Kth = (T)0;
  int32_t global_positive = 0, global_negative = 0;
  int32_t& thread_positive = S32[tid << 1];
  int32_t& thread_negative = S32[(tid << 1) + 1];
  thread_positive = thread_negative = 0;
  auto KK = dimension - K + 1;
  auto offset = tid * XPT;
  for (int64_t i = 0; i < XPT; ++i) {
    auto& x = X[FROM(offset + i)];
    if (x > 0) {
      ++thread_positive;
    } else if (x < 0) {
      ++thread_negative;
    }
  }
  __syncthreads();
}

template <typename T>
int64_t CountBits() {
  auto bits = sizeof(T) << 3;
  switch (typeid(T).name()[0] - 'a') {
    case 0: //int8
    case 3: //double
    case 5: //float
    case 8: //int32_t
    case 11://int64_t
    case 18://int16_t
      return bits - 1;
    default:
      return bits;
  }
}

template <typename T>
bool Integer() {
  switch (typeid(T).name()[0] - 'a') {
    case 3:
    case 5:
      return false;
    default:
      return true;
  }
}

template <typename T>
Status TopKImpl(const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t N, int64_t dimension) {
  auto aligned_K = static_cast<int64_t>(pow(2, ceil(log2(K))));
  auto aligned_dimension = static_cast<int64_t>(pow(2, ceil(log2(dimension))));
  if (aligned_dimension <= GridDim::maxThreadsPerBlock << 1) {
    BitonicTopK<T><<<N, GridDim::maxThreadsPerBlock, aligned_dimension * sizeof(KV<T>)>>>(input_x, output_v, output_i, elem_nums, size, axis, K, aligned_K, largest, sorted, dimension, aligned_dimension, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    return Status::OK();
  } else {
    auto XPT = static_cast<int64_t>(ceil(static_cast<double>(dimension) / GridDim::maxThreadsPerBlock));
    RadixTopK<T><<<N, GridDim::maxThreadsPerBlock, GridDim::maxThreadsPerBlock * sizeof(int64_t)>>>(input_x, output_v, output_i, elem_nums, size, axis, K, largest, sorted, dimension, XPT, CountBits<T>(), Integer<T>());
    return Status::OK();
  }
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
}  // namespace cuda