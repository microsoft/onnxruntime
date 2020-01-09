// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "device_atomic_functions.h"
#include "cub/cub.cuh"
#include "cub/util_type.cuh"
#include "cub/util_allocator.cuh"
#include "cub/device/device_radix_sort.cuh"
#include <limits>

namespace onnxruntime {
namespace cuda {

using namespace cub;

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
      auto dir = len << 1;
      for (int64_t inc = len; inc > 0; inc >>= 1) {
        auto low = tid & (inc - 1);
        auto i = (tid << 1) - low;
        auto j = i + inc;
        if (j < aligned_K) {
          auto reverse = (dir & i) == 0;
          auto swap = reverse ^ (S[i].val < S[j].val);
          if (swap) {
            auto tmp = S[i];
            S[i] = S[j];
            S[j] = tmp;
          }
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
__device__ __inline__ bool Equal(const T& t0, const T& t1) {
  auto t2 = t0 > t1 ? t0 - t1 : t1 - t0;
  return (double)t2 < 1.0e-5;
}

template<typename T>
__device__ bool SamePrefix(const T* t0, const T* t1, int64_t skip) {
  return (((*t0)^(*t1))>>skip) == 0;
}

__device__ bool SamePrefix(const float* f0, const float* f1, int64_t skip) {
  return SamePrefix((const int32_t*)f0, (const int32_t*)f1, skip);
}

__device__ bool SamePrefix(const double* d0, const double* d1, int64_t skip) {
  return SamePrefix((const int64_t*)d0, (const int64_t*)d1, skip);
}

template<typename T>
__device__ int32_t Radix(const T* t, int64_t skip, int64_t mod) {
  return ((*t)>>skip)&mod;
}

__device__ int32_t Radix(const float* f, int64_t skip, int64_t mod) {
  return Radix((const int32_t*)f, skip, mod);
}

__device__ int32_t Radix(const double* d, int64_t skip, int64_t mod) {
  return Radix((const double*)d, skip, mod);
}

template<typename T>
__device__ void SetByte(T* t, int64_t byte) {
  (*t) |= byte;
}

__device__ void SetByte(float* f, int64_t byte) {
  SetByte((int32_t*)f, byte);
}

__device__ void SetByte(double* d, int64_t byte) {
  SetByte((int64_t*)d, byte);
}

template<typename T, int64_t THREADS, int64_t KPT>
__global__ void RadixTopK(const T* X, T* V, int64_t* I, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t dimension, int64_t XPT, T type_min, T type_max) {
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  extern __shared__ char shared_mem[];
  auto H = (uint32_t*)shared_mem;
  auto C = H + 256;
  auto replica_K = H + 257;
  auto all_superior = H + 258;
  auto mid_dim = axis == size - 1 ? 1 : elem_nums[axis + 1];
  auto left_dim = bid / mid_dim * elem_nums[axis];
  auto right_dim = axis == size - 1 ? 0 : bid % elem_nums[axis + 1];
  T Kth = (T)0;
  int64_t mod = 255;
  auto offset = (int64_t)tid * XPT;
  if (0 == tid) {
    *C = 256;
    *replica_K = K;
    *all_superior = 0;
  }
  typedef BlockScan<uint32_t, THREADS> BlockScan;
  __shared__ typename BlockScan::TempStorage temp_storage;
  __syncthreads();
  #pragma unroll 
  for (int64_t byte = sizeof(T)-1; byte > -1; --byte) {
    if (tid < 256) H[tid] = 0;
    __syncthreads();
    auto skip = 8 * byte, prev_skip = 8 * (byte + 1);
    for (int64_t xpt_i = 0; xpt_i < XPT; ++xpt_i) {
      auto x_i = offset + xpt_i;
      if (x_i < dimension) {
        T x = X[FROM(x_i)];
        if (byte == sizeof(T) - 1 || SamePrefix(&x, &Kth, prev_skip)) {
          atomicAdd(&H[Radix(&x, skip, mod)], 1);
        }
      }
    }
    __syncthreads();
    auto h = tid < 256 ? H[1==largest?255-tid:tid] : 0;
    auto sum_h = h;
    BlockScan(temp_storage).ExclusiveSum(sum_h, sum_h);
    if (sum_h + h >= *replica_K) {
      atomicMin(C, tid);
    }
    __syncthreads();
    if (*C == tid) {
      atomicAdd(all_superior, sum_h);
      atomicSub(replica_K, sum_h);
    }
    __syncthreads();
    if (1 == largest) {
      SetByte(&Kth, (255-(*C))<<skip);
    } else {
      SetByte(&Kth, (*C)<<skip);
    }
    __syncthreads();
    if (0 == tid) {
      *C = 256;
    }
    __syncthreads();
  }
  uint32_t superior = 0, equal = 0;
  for (int64_t xpt_i = 0; xpt_i < XPT; ++xpt_i) {
    auto x_i = offset + xpt_i;
    if (x_i < dimension) {
      auto x = X[FROM(x_i)];
      if (1 == largest && x > Kth || 0 == largest && x < Kth) {
        ++superior;
      } else if (Equal(x, Kth)) {
        ++equal;
      }
    }
  }
  BlockScan(temp_storage).ExclusiveSum(superior, superior);
  BlockScan(temp_storage).ExclusiveSum(equal, equal);
  auto equal_quota = K - *all_superior - equal;
  auto output_i = superior + MIN(K - *all_superior, equal);
  for (int64_t xpt_i = 0; xpt_i < XPT; ++xpt_i) {
    auto x_i = offset + xpt_i;
    if (x_i < dimension) {
      auto x = X[FROM(x_i)];
      if (1 == largest && x > Kth || 0 == largest && x < Kth) {
        auto to_i = TO(output_i);
        V[to_i] = x;
        I[to_i] = x_i;
        ++output_i;
      } else if (Equal(x, Kth) && equal_quota > 0) {
        auto to_i = TO(output_i);
        V[to_i] = x;
        I[to_i] = x_i;
        ++output_i;
        --equal_quota;
      }
    }
  }
  __syncthreads();
  if (1 == sorted) {
    T keys[KPT];
    int64_t vals[KPT];
    auto kpt_offset = tid * KPT;
    #pragma unroll 
    for (int64_t kpt_i = 0; kpt_i < KPT; ++kpt_i) {
      auto k_i = kpt_offset + kpt_i;
      if (k_i < K) {
        auto to_i = TO(k_i);
        keys[kpt_i] = V[to_i];
        vals[kpt_i] = I[to_i];
      } else {
        if (1 == largest) {
          keys[kpt_i] = type_min;
        } else {
          keys[kpt_i] = type_max;
        }
        vals[kpt_i] = dimension;
      }
    }
    __syncthreads();
    typedef BlockRadixSort<T, THREADS, KPT, int64_t> BlockRadixSort;
    __shared__ typename BlockRadixSort::TempStorage temp_storage;
    if (1 == largest) {
      BlockRadixSort(temp_storage).SortDescending(keys, vals);
    } else {
      BlockRadixSort(temp_storage).Sort(keys, vals);
    }
    __syncthreads();
    #pragma unroll
    for (int64_t kpt_i = 0; kpt_i < KPT; ++kpt_i) {
      auto k_i = kpt_offset + kpt_i;
      if (k_i < K) {
        auto to_i = TO(k_i);
        V[to_i] = keys[kpt_i];
        I[to_i] = vals[kpt_i];
      }
    }
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
    #define BT GridDim::maxThreadsPerBlock
    if (BT*2 >= K) {
      RadixTopK<T,BT,2><<<N,BT,259*sizeof(uint32_t)>>>(input_x, output_v, output_i, elem_nums, size, axis, K, largest, sorted, dimension, XPT, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    } else if (BT*4>=K) {
      RadixTopK<T,BT,4><<<N,BT,259*sizeof(uint32_t)>>>(input_x, output_v, output_i, elem_nums, size, axis, K, largest, sorted, dimension, XPT, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    } else if (BT*8>=K) {
      RadixTopK<T,BT,8><<<N,BT,259*sizeof(uint32_t)>>>(input_x, output_v, output_i, elem_nums, size, axis, K, largest, sorted, dimension, XPT, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    } else if (BT*16>=K) {
      RadixTopK<T,BT,16><<<N,BT,259*sizeof(uint32_t)>>>(input_x, output_v, output_i, elem_nums, size, axis, K, largest, sorted, dimension, XPT, std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
    }
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
}  // namespace onnxruntime
