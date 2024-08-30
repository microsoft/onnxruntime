// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "topk_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "device_atomic_functions.h"
#include "cub/cub.cuh"
#include "cub/util_type.cuh"
#include "cub/util_allocator.cuh"
#include "cub/device/device_radix_sort.cuh"
#include <limits>
// TODO:fix the warnings
#ifdef _MSC_VER
#pragma warning(disable : 4244)
#endif
namespace onnxruntime {
namespace cuda {

using namespace cub;

template <typename T>
struct KV {
  T key;
  int64_t val;
};

#define BT GridDim::maxThreadsPerBlock
#define ALIGN(N) static_cast<int64_t>(pow(2, ceil(log2(static_cast<double>(N)))))
#define FROM(idx) (left_dim + (idx)*mid_dim + right_dim)
#define TO(idx) (left_dim * K / dimension + (idx)*mid_dim + right_dim)
#define TRIVIAL (1 == largest ? type_min : type_max)
#define BIGGER(n, m) (n.key > m.key ? n : (n.key < m.key ? m : (n.val > m.val ? (1 == largest ? m : n) : (1 == largest ? n : m))))
#define SMALLER(n, m) (n.key < m.key ? n : (n.key > m.key ? m : (n.val < m.val ? (1 == largest ? m : n) : (1 == largest ? n : m))))
#define IS_SMALLER(n, m) (n.key < m.key || !(n.key > m.key) && (1 == largest ? n.val > m.val : n.val < m.val))
#define LESS(n, m) ((n) <= (m) ? (n) : (m))

template <typename T>
__global__ void BitonicTopK(const T* X, T* V, int64_t* I, const TArray<int64_t> elem_nums, size_t size, int32_t axis, int64_t K, int64_t aligned_K, int64_t largest, int64_t sorted, int64_t dimension, int64_t aligned_dimension, T type_min, T type_max) {
  int64_t tid = threadIdx.x;
  int64_t bid = blockIdx.x;
  int64_t bdim = blockDim.x;
  extern __shared__ char shared_mem[];
  auto S = (KV<T>*)(shared_mem);
  auto mid_dim = axis == size - 1 ? 1 : elem_nums[axis + 1];
  auto left_dim = bid / mid_dim * elem_nums[axis];
  auto right_dim = axis == size - 1 ? 0 : bid % elem_nums[axis + 1];
  for (auto i = tid; i < aligned_dimension; i += bdim) {
    S[i].key = i < dimension ? X[FROM(i)] : TRIVIAL;
    S[i].val = i;
  }
  __syncthreads();
  // sort each K
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
      __syncthreads();
    }
    __syncthreads();
  }
  // merge and rebuild K
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
      __syncthreads();
    }
    __syncthreads();
  }
  // save top K
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
    // sort by index ascending
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
        __syncthreads();
      }
      __syncthreads();
    }
    if (tid < K) {
      auto to = TO(tid);
      V[to] = S[tid].key;
      I[to] = S[tid].val;
    }
  }
}

template <typename T>
__device__ __forceinline__ bool Equal(const T& t0, const T& t1) {
  return t0 == t1;
}

__device__ __forceinline__ bool Equal(const float& t0, const float& t1) {
  return !(t0 > t1 || t1 > t0);
}

__device__ __forceinline__ bool Equal(const double& t0, const double& t1) {
  return !(t0 > t1 || t1 > t0);
}

template <typename T>
__device__ __forceinline__ bool SamePrefix(const T* t0, const T* t1, int64_t skip) {
  return ((*t0) ^ (*t1)) >> skip == 0;
}

__device__ __forceinline__ bool SamePrefix(const half* f0, const half* f1, int64_t skip) {
  return SamePrefix((const int16_t*)f0, (const int16_t*)f1, skip);
}

__device__ __forceinline__ bool SamePrefix(const float* f0, const float* f1, int64_t skip) {
  return SamePrefix((const int32_t*)f0, (const int32_t*)f1, skip);
}

__device__ __forceinline__ bool SamePrefix(const double* d0, const double* d1, int64_t skip) {
  return SamePrefix((const int64_t*)d0, (const int64_t*)d1, skip);
}

template <typename T>
__device__ __forceinline__ int32_t Radix(const T* t, int64_t skip) {
  return ((*t) >> skip) & 255;
}

__device__ __forceinline__ int32_t Radix(const half* f, int64_t skip) {
  return Radix((const int16_t*)f, skip);
}

__device__ __forceinline__ int32_t Radix(const float* f, int64_t skip) {
  return Radix((const int32_t*)f, skip);
}

__device__ __forceinline__ int32_t Radix(const double* d, int64_t skip) {
  return Radix((const int64_t*)d, skip);
}

template <typename T>
__device__ void SetByte(T* t, int64_t byte) {
  (*t) |= byte;
}

__device__ __forceinline__ void SetByte(half* f, int64_t byte) {
  SetByte((int16_t*)f, byte);
}

__device__ __forceinline__ void SetByte(float* f, int64_t byte) {
  SetByte((int32_t*)f, byte);
}

__device__ __forceinline__ void SetByte(double* d, int64_t byte) {
  SetByte((int64_t*)d, byte);
}

template <typename T, int64_t THREADS, int64_t KPT>
__global__ void RadixTopK(const T* X, T* V, int64_t* I, const TArray<int64_t> elem_nums, size_t size, int32_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t dimension, int64_t XPT, T type_min, T type_max) {
  auto tid = threadIdx.x;
  auto bid = blockIdx.x;
  extern __shared__ char shared_mem[];
  auto H = (uint32_t*)shared_mem;
  auto mid_dim = axis == size - 1 ? 1 : elem_nums[axis + 1];
  auto left_dim = bid / mid_dim * elem_nums[axis];
  auto right_dim = axis == size - 1 ? 0 : bid % elem_nums[axis + 1];
  T Kth = (T)0, sign = (T)1;
  typedef BlockScan<uint32_t, THREADS> BlockScan;
  typedef BlockReduce<uint32_t, THREADS> BlockReduce;
  typedef BlockRadixSort<T, THREADS, KPT, int64_t> BlockRadixSort;
  __shared__ union {
    typename BlockScan::TempStorage scan;
    typename BlockReduce::TempStorage reduce;
    typename BlockRadixSort::TempStorage sort;
  } temp_storage;
  uint32_t positive = 0, negative = 0;
  for (int64_t x_i = tid; x_i < dimension; x_i += blockDim.x) {
    T x = X[FROM(x_i)];
    if (x > (T)0) {
      ++positive;
    } else if (x < (T)0) {
      ++negative;
    }
  }
  __syncthreads();
  positive = BlockReduce(temp_storage.reduce).Sum(positive);
  __syncthreads();
  negative = BlockReduce(temp_storage.reduce).Sum(negative);
  if (0 == tid) {
    H[0] = positive;
    H[1] = negative;
  }
  __syncthreads();
  positive = H[0];
  negative = H[1];
  if ((1 == largest && (K <= positive || dimension - K + 1 <= negative)) ||
      (0 == largest && (K <= negative || dimension - K + 1 <= positive))) {
    auto KK = K;
    if (1 == largest) {
      if (KK > positive) {
        KK = dimension - KK + 1;
        sign = (T)-1;
      }
    } else {
      if (KK > negative) {
        KK = dimension - KK + 1;
      } else {
        sign = (T)-1;
      }
    }
    __syncthreads();
#pragma unroll
    for (int64_t byte = sizeof(T) - 1; byte > -1; --byte) {
      if (tid < 256) H[tid] = 0;
      __syncthreads();
      auto skip = 8 * byte, prev_skip = 8 * (byte + 1);
      for (int64_t x_i = tid; x_i < dimension; x_i += blockDim.x) {
        T x = sign * X[FROM(x_i)];
        if (x > (T)0 && (byte == sizeof(T) - 1 || SamePrefix(&x, &Kth, prev_skip))) {
          atomicAdd(&H[Radix(&x, skip)], 1);
        }
      }
      __syncthreads();
      for (int64_t radix = 255; radix > 0; --radix) {
        if (H[radix] < KK) {
          KK -= H[radix];
        } else {
          SetByte(&Kth, radix << skip);
          break;
        }
      }
      __syncthreads();
    }
    Kth *= sign;
  }
  uint32_t superior = 0, equal = 0;
  for (int64_t x_i = tid; x_i < dimension; x_i += blockDim.x) {
    auto x = X[FROM(x_i)];
    if ((1 == largest && x > Kth) || (0 == largest && x < Kth)) {
      ++superior;
    } else if (Equal(x, Kth)) {
      ++equal;
    }
  }
  __syncthreads();
  auto all_superior = superior;
  all_superior = BlockReduce(temp_storage.reduce).Sum(all_superior);
  if (0 == tid) {
    H[0] = all_superior;
  }
  __syncthreads();
  all_superior = H[0];
  BlockScan(temp_storage.scan).ExclusiveSum(superior, superior);
  __syncthreads();
  BlockScan(temp_storage.scan).ExclusiveSum(equal, equal);
  __syncthreads();
  auto equal_quota = K - all_superior - equal;
  auto output_i = superior + LESS(K - all_superior, equal);
  for (int64_t x_i = tid; x_i < dimension; x_i += blockDim.x) {
    auto x = X[FROM(x_i)];
    if ((1 == largest && x > Kth) || (0 == largest && x < Kth)) {
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
  __syncthreads();
  if (1 == sorted) {
    T keys[KPT];
    int64_t vals[KPT];
    for (int64_t k_i = tid, k_c = 0; k_c < KPT; k_i += blockDim.x, ++k_c) {
      if (k_i < K) {
        auto to_i = TO(k_i);
        keys[k_c] = V[to_i];
        vals[k_c] = I[to_i];
      } else {
        if (1 == largest) {
          keys[k_c] = type_min;
        } else {
          keys[k_c] = type_max;
        }
      }
    }
    __syncthreads();
    if (1 == largest) {
      BlockRadixSort(temp_storage.sort).SortDescending(keys, vals);
    } else {
      BlockRadixSort(temp_storage.sort).Sort(keys, vals);
    }
    __syncthreads();
#pragma unroll
    for (int64_t k_c = 0; k_c < KPT; ++k_c) {
      auto k_i = tid * KPT + k_c;
      if (k_i < K) {
        auto to_i = TO(k_i);
        V[to_i] = keys[k_c];
        I[to_i] = vals[k_c];
      }
    }
  }
}

template <typename T>
__global__ void FillInput(const T* input_x, T* output_v, int64_t* output_i, const TArray<int64_t> elem_nums, size_t size, int32_t axis, int64_t K, int64_t offset, int64_t dimension) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, dimension);
  auto left = offset / (axis == size - 1 ? 1 : elem_nums[axis + 1]) * elem_nums[axis];
  auto right = axis == size - 1 ? 0 : offset % elem_nums[axis + 1];
  auto input_offset = left + id * (axis == size - 1 ? 1 : elem_nums[axis + 1]) + right;
  output_v[id] = input_x[input_offset];
  output_i[id] = id;
}

template <typename T>
__global__ void FillOutput(const T* input_v, const int64_t* input_i, T* output_v, int64_t* output_i, const TArray<int64_t> elem_nums, size_t size, int32_t axis, int64_t K, int64_t offset, int64_t dimension) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, K);
  auto left = offset / (axis == size - 1 ? 1 : elem_nums[axis + 1]) * elem_nums[axis] * K / dimension;
  auto right = axis == size - 1 ? 0 : offset % elem_nums[axis + 1];
  auto output_offset = left + id * (axis == size - 1 ? 1 : elem_nums[axis + 1]) + right;
  output_v[output_offset] = input_v[id];
  output_i[output_offset] = input_i[id];
}

// template is used to avoid linking issue, since __global__ function cannot be inline-ed
template <typename T>
__global__ void ExcludeOutput(T* output_i, T K, T dimension) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, dimension);
  if (id >= K) {
    output_i[id] = dimension;
  }
}

template <typename T>
Status TopKImpl(const CudaKernel* kernel, bool use_deterministic_compute,
                Stream* ort_stream, const T* input_x, T* output_v, int64_t* output_i,
                const TArray<int64_t>& elem_nums, size_t size, int32_t axis, int64_t K, int64_t largest,
                int64_t sorted, int64_t N, int64_t dimension) {
  typedef typename ToCudaType<T>::MappedType CudaT;
  const CudaT* input_x_ptr = reinterpret_cast<const CudaT*>(input_x);
  CudaT* output_v_ptr = reinterpret_cast<CudaT*>(output_v);
  cudaStream_t stream = ort_stream ? static_cast<cudaStream_t>(ort_stream->GetHandle()) : nullptr;

  auto aligned_K = ALIGN(K);
  auto aligned_dimension = ALIGN(dimension);
  if (aligned_dimension <= GridDim::maxThreadsPerBlock) {
    BitonicTopK<CudaT><<<N, GridDim::maxThreadsPerBlock, aligned_dimension * sizeof(KV<CudaT>), stream>>>(
        input_x_ptr, output_v_ptr, output_i, elem_nums, size, axis, K, aligned_K, largest, sorted, dimension,
        aligned_dimension, NumericLimits<T>::Min(), NumericLimits<T>::Max());
  } else if (K <= BT * 16 || 0 == sorted) {
    if (use_deterministic_compute) {
      static std::once_flag log_warning;
      std::call_once(log_warning, []() {
        LOGS_DEFAULT(WARNING) << "Non-deterministic TopKImpl kernel is called, its outputs may still be nondeterministic.";
      });
    }

    auto XPT = static_cast<int64_t>(ceil(static_cast<double>(dimension) / GridDim::maxThreadsPerBlock));
    if (BT * 2 >= K || 0 == sorted) {
      RadixTopK<CudaT, BT, 2><<<N, BT, 256 * sizeof(uint32_t), stream>>>(
          input_x_ptr, output_v_ptr, output_i, elem_nums, size, axis, K, largest, sorted, dimension, XPT,
          NumericLimits<T>::Min(), NumericLimits<T>::Max());
    } else if (BT * 4 >= K) {
      RadixTopK<CudaT, BT, 4><<<N, BT, 256 * sizeof(uint32_t), stream>>>(
          input_x_ptr, output_v_ptr, output_i, elem_nums, size, axis, K, largest, sorted, dimension, XPT,
          NumericLimits<T>::Min(), NumericLimits<T>::Max());
    } else if (BT * 8 >= K) {
      RadixTopK<CudaT, BT, 8><<<N, BT, 256 * sizeof(uint32_t), stream>>>(
          input_x_ptr, output_v_ptr, output_i, elem_nums, size, axis, K, largest, sorted, dimension, XPT,
          NumericLimits<T>::Min(), NumericLimits<T>::Max());
    } else {
      RadixTopK<CudaT, BT, 16><<<N, BT, 256 * sizeof(uint32_t), stream>>>(
          input_x_ptr, output_v_ptr, output_i, elem_nums, size, axis, K, largest, sorted, dimension, XPT,
          NumericLimits<T>::Min(), NumericLimits<T>::Max());
    }
  } else {
    auto input_key_buffer = kernel->GetScratchBuffer<CudaT>(dimension, ort_stream);
    auto output_key_buffer = kernel->GetScratchBuffer<CudaT>(dimension, ort_stream);
    auto input_value_buffer = kernel->GetScratchBuffer<int64_t>(dimension, ort_stream);
    auto output_value_buffer = kernel->GetScratchBuffer<int64_t>(dimension, ort_stream);
    auto* input_key = input_key_buffer.get();
    auto* output_key = output_key_buffer.get();
    auto* input_value = input_value_buffer.get();
    auto* output_value = output_value_buffer.get();
    size_t temp_bytes = 0;
    CUDA_RETURN_IF_ERROR(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, input_key, output_key, input_value, output_value, dimension, 0, sizeof(T) * 8, stream));
    auto temp_storage_buffer = kernel->GetScratchBuffer<char>(temp_bytes, ort_stream);
    auto* temp_storage = temp_storage_buffer.get();
    auto blocks_per_grid_D = (int)(ceil(static_cast<float>(dimension) / BT));
    auto blocks_per_grid_K = (int)(ceil(static_cast<float>(K) / BT));
    for (int64_t i = 0; i < N; i++) {
      FillInput<CudaT><<<blocks_per_grid_D, BT, 0, stream>>>(input_x_ptr, input_key, input_value, elem_nums, size, axis, K, i, dimension);
      CUDA_RETURN_IF_ERROR(1 == largest ? cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_bytes, input_key, output_key, input_value, output_value, dimension, 0, sizeof(T) * 8, stream)
                                        : cub::DeviceRadixSort::SortPairs(temp_storage, temp_bytes, input_key, output_key, input_value, output_value, dimension, 0, sizeof(T) * 8, stream));
      if (1 == sorted) {
        FillOutput<CudaT><<<blocks_per_grid_K, BT, 0, stream>>>(output_key, output_value, output_v_ptr, output_i, elem_nums, size, axis, K, i, dimension);
      } else {  // reorder by ascending index
        ExcludeOutput<int64_t><<<blocks_per_grid_D, BT, 0, stream>>>(output_value, K, dimension);
        CUDA_RETURN_IF_ERROR(cub::DeviceRadixSort::SortPairs(temp_storage, temp_bytes, output_value, input_value, output_key, input_key, dimension, 0, sizeof(T) * 8, stream));
        FillOutput<CudaT><<<blocks_per_grid_K, BT, 0, stream>>>(input_key, input_value, output_v_ptr, output_i, elem_nums, size, axis, K, i, dimension);
      }
    }
  }
  return Status::OK();
}

#define TOPKIMPLE(T) template Status TopKImpl<T>(const CudaKernel* kernel,         \
                                                 bool use_deterministic_compute,   \
                                                 Stream* ort_stream,               \
                                                 const T* input_x,                 \
                                                 T* output_v,                      \
                                                 int64_t* output_i,                \
                                                 const TArray<int64_t>& elem_nums, \
                                                 size_t size,                      \
                                                 int32_t axis,                     \
                                                 int64_t K,                        \
                                                 int64_t largest,                  \
                                                 int64_t sorted,                   \
                                                 int64_t N,                        \
                                                 int64_t dimension)

// This file is causing excessive long compilation time in ROCm EP. Split all those compilations into multiple
// translation units to speed it up.
TOPKIMPLE(TOPK_IMPL_TYPE);

}  // namespace cuda
}  // namespace onnxruntime
