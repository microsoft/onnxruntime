/*
 The implementation of this file is based on bert plugins in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/

Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kWarpSize = 32;
constexpr int KBlockSizeWrapImpl = 128;

template <typename ComputeType>
__device__ inline ComputeType rsqrt(const ComputeType& x) {
  return ComputeType(1) / sqrt(x);
}

template <>
__device__ inline float rsqrt(const float& x) {
  return rsqrtf(x);
}

template <>
__device__ inline half rsqrt(const half& x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return hrsqrt(x);
#else
  return half(rsqrtf(float(x)));
#endif
}

template <>
__device__ inline double rsqrt(const double& x) {
  return rsqrt(x);
}

template<typename T>
__inline__ __device__ T Div(T a, T b);

template<>
__inline__ __device__ float Div<float>(float a, float b) {
#ifdef OF_LAYER_NORM_USE_FAST_MATH
  return __fdividef(a, b);
#else
  return a / b;
#endif
}

template<>
__inline__ __device__ double Div<double>(double a, double b) {
  return a / b;
}

int64_t GetNumBlocksWrapImpl(const int64_t rows, int64_t const thread_group_width) {
  const int64_t thread_groups_per_block = KBlockSizeWrapImpl / thread_group_width;
  const int64_t num_blocks =
      (rows + thread_groups_per_block - 1) / thread_groups_per_block;
  return num_blocks;
}

template<class Func>
inline cudaError_t GetNumBlocks(Func func, int64_t block_size, size_t dynamic_smem_size,
                                int64_t max_blocks, int64_t waves, int* num_blocks) {
  int dev;
  {
    cudaError_t err = cudaGetDevice(&dev);
    if (err != cudaSuccess) { return err; }
  }
  int sm_count;
  {
    cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
    if (err != cudaSuccess) { return err; }
  }
  int max_active_blocks;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_active_blocks, func,
                                                                    block_size, dynamic_smem_size);
  }
  *num_blocks =
      std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * max_active_blocks * waves));
  return cudaSuccess;
}

template<typename T>
struct DefaultComputeType {
  using type = T;
};

template<>
struct DefaultComputeType<half> {
  using type = float;
};

#if CUDA_VERSION >= 11000
template<>
struct DefaultComputeType<nv_bfloat16> {
  using type = float;
};
#endif  // CUDA_VERSION >= 11000

template<typename T, int N>
struct GetPackType {
  using type = typename std::aligned_storage<N * sizeof(T), N * sizeof(T)>::type;
};

template<typename T, int N>
using PackType = typename GetPackType<T, N>::type;

template<typename T, int N>
union Pack {
  static_assert(sizeof(PackType<T, N>) == sizeof(T) * N, "");
  __device__ Pack() {
    // do nothing
  }
  PackType<T, N> storage;
  T elem[N];
};
template<typename SRC, typename DST>
struct DirectLoad {
  DirectLoad(const SRC* src, int64_t row_size) : src(src), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = static_cast<DST>(pack.elem[i]); }
  }
  const SRC* src;
  int64_t row_size;
};

template<typename SRC, typename DST>
struct DirectLoadSkip {
  DirectLoadSkip(const SRC* src, const SRC* skip, const SRC* bias, int64_t row_size) : src(src), skip(skip), bias(bias), row_size(row_size) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<SRC, N>*>(src) + offset);
    Pack<SRC, N> pack_skip;
    pack_skip.storage = *(reinterpret_cast<const PackType<SRC, N>*>(skip) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = (bias == nullptr) ? static_cast<DST>(pack.elem[i]) + static_cast<DST>(pack_skip.elem[i]) : static_cast<DST>(pack.elem[i]) + static_cast<DST>(pack_skip.elem[i]) + static_cast<DST>(bias[col]); }
  }
  const SRC* src;
  const SRC* skip;
  const SRC* bias;
  int64_t row_size;
};

template<typename DST>
struct DirectLoadSkip<half, DST> {
  DirectLoadSkip(const half* src, const half* skip, const half* bias, int64_t row_size) : src(src), skip(skip), bias(bias), row_size(row_size) {}
  template<int N>
  __device__ typename std::enable_if<N % 2 != 0, void>::type load(DST* dst, int64_t row, int64_t col) const {
    Pack<half, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<half, N>*>(src) + offset);
    Pack<half, N> pack_skip;
    pack_skip.storage = *(reinterpret_cast<const PackType<half, N>*>(skip) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) { dst[i] = (bias == nullptr) ? static_cast<DST>(pack.elem[i]) + static_cast<DST>(pack_skip.elem[i]) : static_cast<DST>(pack.elem[i]) + static_cast<DST>(pack_skip.elem[i]) + static_cast<DST>(bias[col]); }
  } 

  template<int N>
  __device__ typename std::enable_if<N % 2 == 0, void>::type load(DST* dst, int64_t row, int64_t col) const {
    Pack<half, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const PackType<half, N>*>(src) + offset);
    Pack<half, N> pack_skip;
    pack_skip.storage = *(reinterpret_cast<const PackType<half, N>*>(skip) + offset);
    #pragma unroll
    for (int i = 0; i < N / 2; ++i) {
      const half2 pack2 = __halves2half2(pack.elem[i], pack.elem[i + 1]);
      const half2 pack_size2 = __halves2half2(pack_skip.elem[i], pack_skip.elem[i + 1]);
      half2 res = AddHalf2(pack2, pack_size2);
      if (bias == nullptr) {
        dst[i] = static_cast<DST>(res.x);
        dst[i + 1] = static_cast<DST>(res.y);
      } else {
        dst[i] = static_cast<DST>(res.x + bias[col]); 
        dst[i + 1] = static_cast<DST>(res.y + bias[col]);
      }
    }
  }
  const half* src;
  const half* skip;
  const half* bias;
  int64_t row_size;
};

template<typename SRC, typename DST, bool do_scale, bool do_center, bool simplified = false>
struct AffineStore {
  AffineStore(DST* y, int64_t row_size, const DST* gamma, const DST* beta)
      : y(y), row_size(row_size), gamma(gamma), beta(beta) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    Pack<DST, N> y_pack;
    Pack<DST, N> gamma_pack;
    Pack<DST, N> beta_pack;
    const int64_t offset = (row * row_size + col) / N;
    const int64_t gamma_offset = col / N;
    if (do_scale) {
      gamma_pack.storage =
          *(reinterpret_cast<const PackType<DST, N>*>(gamma) + gamma_offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) { gamma_pack.elem[i] = 1; }
    }
    if (do_center && ! simplified) {
      beta_pack.storage =
          *(reinterpret_cast<const PackType<DST, N>*>(beta) + gamma_offset);
    } else {
#pragma unroll
      for (int i = 0; i < N; ++i) { beta_pack.elem[i] = 0; }
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      DST normalized_i = static_cast<DST>(src[i]);
      if (do_scale || do_center) {
        y_pack.elem[i] = normalized_i * gamma_pack.elem[i] + beta_pack.elem[i];
      } else {
        y_pack.elem[i] = normalized_i;
      }
    }
    *(reinterpret_cast<PackType<DST, N>*>(y) + offset) = y_pack.storage;
  }
  DST* y;
  int64_t row_size;
  const DST* gamma;
  const DST* beta;
};

template<typename ComputeType, bool simplified>
inline __device__ void WelfordCombine(ComputeType val, ComputeType* mean, ComputeType* m2, ComputeType* count) {
  // Use Welford Online algorithem to compute mean and variance
  // For more details you can refer to:
  // https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
  *count += 1;
  ComputeType delta1 = val - *mean;
  *mean += Div(delta1, *count);
  ComputeType delta2 = val - *mean;
  if (simplified) {
    *m2 += val * val;
  } else {
    *m2 += delta1 * delta2;
  }
}

template<typename ComputeType, bool simplified>
inline __device__ void WelfordCombine(ComputeType b_mean, ComputeType b_m2, ComputeType b_count, ComputeType* mean, ComputeType* m2, ComputeType* count) {
  if (b_count == 0) { return; }
  ComputeType new_count = *count + b_count;
  ComputeType nb_over_n = Div(b_count, new_count);
  ComputeType delta = b_mean - *mean;
  *mean += delta * nb_over_n;
  if (simplified) {
    *m2 += b_m2;
  } else {
    *m2 += b_m2 + delta * delta * (*count) * nb_over_n;
  }
  *count = new_count;
}

template<typename ComputeType, bool simplified, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpReduce(ComputeType thread_mean, ComputeType thread_m2, ComputeType thread_count, ComputeType* mean,
                                             ComputeType* m2, ComputeType* count) {
  *mean = thread_mean;
  *m2 = thread_m2;
  *count = thread_count;
  for (int mask = thread_group_width / 2; mask > 0; mask /= 2) {
    ComputeType b_mean = __shfl_down_sync(0xffffffff, *mean, mask, thread_group_width);
    ComputeType b_m2 = __shfl_down_sync(0xffffffff, *m2, mask, thread_group_width);
    ComputeType b_count = __shfl_down_sync(0xffffffff, *count, mask, thread_group_width);
    WelfordCombine<ComputeType, simplified>(b_mean, b_m2, b_count, mean, m2, count);
  }
}

template<typename ComputeType, bool simplified, int thread_group_width = kWarpSize>
__inline__ __device__ void WelfordWarpAllReduce(ComputeType thread_mean, ComputeType thread_m2, ComputeType thread_count, ComputeType* mean,
                                                ComputeType* m2, ComputeType* count) {
  WelfordWarpReduce<ComputeType, simplified, thread_group_width>(thread_mean, thread_m2, thread_count, mean, m2, count);
  *mean = __shfl_sync(0xffffffff, *mean, 0, thread_group_width);
  *m2 = __shfl_sync(0xffffffff, *m2, 0, thread_group_width);
  *count = __shfl_sync(0xffffffff, *count, 0, thread_group_width);
}

template<typename ComputeType, bool simplified>
__inline__ __device__ void WelfordBlockAllReduce(ComputeType thread_mean, ComputeType thread_m2, ComputeType thread_count,
                                                 ComputeType* result_mean, ComputeType* result_m2, ComputeType* result_count) {
  __shared__ ComputeType mean_shared[kWarpSize];
  __shared__ ComputeType m2_shared[kWarpSize];
  __shared__ ComputeType count_shared[kWarpSize];
  __shared__ ComputeType mean_result_broadcast;
  __shared__ ComputeType m2_result_broadcast;
  __shared__ ComputeType count_result_broadcast;
  const int lid = threadIdx.x % kWarpSize;
  const int wid = threadIdx.x / kWarpSize;
  ComputeType warp_mean = 0;
  ComputeType warp_m2 = 0;
  ComputeType warp_count = 0;
  WelfordWarpReduce<ComputeType, simplified>(thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2, &warp_count);
  __syncthreads();
  if (lid == 0) {
    mean_shared[wid] = warp_mean;
    m2_shared[wid] = warp_m2;
    count_shared[wid] = warp_count;
  }
  __syncthreads();
  if (wid == 0) {
    if (threadIdx.x < blockDim.x / kWarpSize) {
      warp_mean = mean_shared[lid];
      warp_m2 = m2_shared[lid];
      warp_count = count_shared[lid];
    } else {
      warp_mean = static_cast<ComputeType>(0);
      warp_m2 = static_cast<ComputeType>(0);
      warp_count = static_cast<ComputeType>(0);
    }
    __syncwarp();
    ComputeType block_mean = 0;
    ComputeType block_m2 = 0;
    ComputeType block_count = 0;
    WelfordWarpReduce<ComputeType, simplified>(warp_mean, warp_m2, warp_count, &block_mean, &block_m2, &block_count);
    if (lid == 0) {
      mean_result_broadcast = block_mean;
      m2_result_broadcast = block_m2;
      count_result_broadcast = block_count;
    }
  }
  __syncthreads();
  *result_mean = mean_result_broadcast;
  *result_m2 = m2_result_broadcast;
  *result_count = count_result_broadcast;
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, bool simplified, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
__global__ void LayerNormWarpImpl(LOAD load, STORE store, const int64_t rows, const int64_t cols,
                                  const double epsilon, ComputeType* mean,
                                  ComputeType* inv_variance) {
  static_assert(cols_per_thread % pack_size == 0, "");
  static_assert(thread_group_width <= kWarpSize, "");
  static_assert(kWarpSize % thread_group_width == 0, "");
  constexpr int num_packs = cols_per_thread / pack_size;
  assert(cols <= cols_per_thread * thread_group_width);
  ComputeType buf[rows_per_access][cols_per_thread];
  const int64_t global_thread_group_id = blockIdx.x * blockDim.y + threadIdx.y; //分到的第几个row
  const int64_t num_global_thread_group = gridDim.x * blockDim.y; //总共有多少个row
  const int64_t lane_id = threadIdx.x; //这个线程从当前row的第几个col开始处理，间隔是blockDimX(thread_group_width)
  const int64_t step = num_global_thread_group * rows_per_access;
  for (int64_t row = global_thread_group_id * rows_per_access; row < rows; row += step) {
    ComputeType thread_mean[rows_per_access];
    ComputeType thread_m2[rows_per_access];
    ComputeType thread_count[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      thread_mean[row_id] = 0;
      thread_m2[row_id] = 0;
      thread_count[row_id] = 0;
      ComputeType* row_buf = buf[row_id];
#pragma unroll
      for (int pack_id = 0; pack_id < num_packs; ++pack_id) {
        const int col = (pack_id * thread_group_width + lane_id) * pack_size;
        const int pack_offset = pack_id * pack_size;
        if (!padding || col < cols) {
          load.template load<pack_size>(row_buf + pack_offset, row + row_id, col);
#pragma unroll
          for (int i = 0; i < pack_size; ++i) {
            WelfordCombine<ComputeType, simplified>(row_buf[pack_offset + i], thread_mean + row_id, thread_m2 + row_id,
                           thread_count + row_id);
          }
        } else {
#pragma unroll
          for (int i = 0; i < pack_size; ++i) { row_buf[pack_offset + i] = 0; }
        }
      }
    }
    ComputeType warp_mean[rows_per_access];
    ComputeType warp_m2[rows_per_access];
    ComputeType warp_count[rows_per_access];
#pragma unroll
    for (int row_id = 0; row_id < rows_per_access; ++row_id) {
      int global_row_id = row + row_id;
      ComputeType* row_buf = buf[row_id];
      //一个wrap一行，当thread_groups_width不足32，一个wrap负责wrapsize/thread_groups_width行
      WelfordWarpAllReduce<ComputeType, simplified, thread_group_width>(
          thread_mean[row_id], thread_m2[row_id], thread_count[row_id], warp_mean + row_id,
          warp_m2 + row_id, warp_count + row_id);
      ComputeType row_mean = warp_mean[row_id];
      ComputeType row_variance =
          max(Div(warp_m2[row_id], warp_count[row_id]), static_cast<ComputeType>(0.0));
      ComputeType row_inv_var = rsqrt(row_variance + static_cast<ComputeType>(epsilon));
      if (lane_id == 0) {
        if (mean != nullptr) { mean[global_row_id] = row_mean; }
        if (inv_variance != nullptr) { inv_variance[global_row_id] = row_inv_var; }
      }
#pragma unroll
      for (int i = 0; i < cols_per_thread; ++i) {
        if (simplified) {
          row_buf[i] = row_buf[i] * row_inv_var;
        } else {
          row_buf[i] = (row_buf[i] - row_mean) * row_inv_var;
        }
      }
#pragma unroll
      for (int i = 0; i < num_packs; ++i) {
        const int col = (i * thread_group_width + lane_id) * pack_size;
        if (!padding || col < cols) {
          store.template store<pack_size>(row_buf + i * pack_size, global_row_id, col);
        }
      }
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, bool simplified, int cols_per_thread,
         int thread_group_width, int rows_per_access, bool padding>
inline cudaError_t LaunchLayerNormWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                           const int64_t rows, const int64_t cols,
                                           const double epsilon, ComputeType* mean,
                                           ComputeType* inv_variance) {
  constexpr int block_size = 128;
  constexpr int waves = 32;
  static_assert(block_size % thread_group_width == 0, "");
  constexpr int thread_groups_per_block = block_size / thread_group_width;
  dim3 block_dim(thread_group_width, thread_groups_per_block);
  const int64_t num_blocks =
      (rows / rows_per_access + thread_groups_per_block - 1) / thread_groups_per_block;
  int grid_dim_x;
  {
    cudaError_t err =
        GetNumBlocks(LayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, simplified, cols_per_thread,
                                       thread_group_width, rows_per_access, padding>,
                     block_size, 0, num_blocks, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, simplified, cols_per_thread, thread_group_width,
                    rows_per_access, padding>
      <<<grid_dim_x, block_dim, 0, stream>>>(load, store, rows, cols, epsilon, mean, inv_variance);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, bool simplified, int cols_per_thread,
         int thread_group_width, int rows_per_access>
inline cudaError_t DispatchLayerNormWarpImplPadding(cudaStream_t stream, LOAD load, STORE store,
                                                    const int64_t rows, const int64_t cols,
                                                    const double epsilon, ComputeType* mean,
                                                    ComputeType* inv_variance) {
  if (cols == cols_per_thread * thread_group_width) {
    return LaunchLayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, simplified, cols_per_thread,
                                   thread_group_width, rows_per_access, false>(
        stream, load, store, rows, cols, epsilon, mean, inv_variance);
  } else {
    return LaunchLayerNormWarpImpl<LOAD, STORE, ComputeType, pack_size, simplified, cols_per_thread,
                                   thread_group_width, rows_per_access, true>(
        stream, load, store, rows, cols, epsilon, mean, inv_variance);
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, bool simplified>
typename std::enable_if<pack_size == 1, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                   \
  else if (cols <= (thread_group_width)*pack_size) {                                          \
    int64_t block_num = GetNumBlocksWrapImpl(rows, thread_group_width);                       \
    if (rows % 2 == 0 && block_num > 1024) {                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, simplified, pack_size, \
                                              thread_group_width, 2>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    } else {                                                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, simplified, pack_size, \
                                              thread_group_width, 1>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    }                                                                                         \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                     \
  else if (cols <= (col)*kWarpSize) {                                                            \
    return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, simplified, col, kWarpSize, \
                                            1>(stream, load, store, rows, cols, epsilon, mean,   \
                                               inv_variance);                                    \
  }
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(3)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(5)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(7)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(9)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(11)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(13)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(15)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(17)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(19)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(21)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(23)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(25)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(27)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(29)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(31)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, bool simplified>
typename std::enable_if<pack_size == 2, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                   \
  else if (cols <= (thread_group_width)*pack_size) {                                          \
    int64_t block_num = GetNumBlocksWrapImpl(rows, thread_group_width);                       \
    if (rows % 2 == 0 && block_num > 1024) {                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, simplified, pack_size, \
                                              thread_group_width, 2>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    } else {                                                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, simplified, pack_size, \
                                              thread_group_width, 1>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    }                                                                                         \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                     \
  else if (cols <= (col)*kWarpSize) {                                                            \
    return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, simplified, col, kWarpSize, \
                                            1>(stream, load, store, rows, cols, epsilon, mean,   \
                                               inv_variance);                                    \
  }
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(6)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(10)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(14)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(18)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(22)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(26)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(30)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}
template<typename LOAD, typename STORE, typename ComputeType, int pack_size, bool simplified>
typename std::enable_if<pack_size == 4, cudaError_t>::type DispatchLayerNormWarpImplCols(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance) {
  if (cols <= 0) { return cudaErrorInvalidValue; }
#define DEFINE_ONE_ELIF(thread_group_width)                                                   \
  else if (cols <= (thread_group_width)*pack_size) {                                          \
    int64_t block_num = GetNumBlocksWrapImpl(rows, thread_group_width);                       \
    if (rows % 2 == 0 && block_num > 1024) {                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, simplified, pack_size, \
                                              thread_group_width, 2>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    } else {                                                                                  \
      return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, simplified, pack_size, \
                                              thread_group_width, 1>(                         \
          stream, load, store, rows, cols, epsilon, mean, inv_variance);                      \
    }                                                                                         \
  }
  DEFINE_ONE_ELIF(1)
  DEFINE_ONE_ELIF(2)
  DEFINE_ONE_ELIF(4)
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
#define DEFINE_ONE_ELIF(col)                                                                     \
  else if (cols <= (col)*kWarpSize) {                                                            \
    return DispatchLayerNormWarpImplPadding<LOAD, STORE, ComputeType, pack_size, simplified, col, kWarpSize, \
                                            1>(stream, load, store, rows, cols, epsilon, mean,   \
                                               inv_variance);                                    \
  }
  DEFINE_ONE_ELIF(8)
  DEFINE_ONE_ELIF(12)
  DEFINE_ONE_ELIF(16)
  DEFINE_ONE_ELIF(20)
  DEFINE_ONE_ELIF(24)
  DEFINE_ONE_ELIF(28)
  DEFINE_ONE_ELIF(32)
#undef DEFINE_ONE_ELIF
  else {
    return cudaErrorInvalidValue;
  }
}

template<typename LOAD, typename STORE, typename ComputeType, bool simplified>
struct DispatchLayerNormWarpImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, ComputeType* mean,
                         ComputeType* inv_variance) {
    if (cols % 4 == 0) {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 4, simplified>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else if (cols % 2 == 0) {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 2, simplified>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else {
      return DispatchLayerNormWarpImplCols<LOAD, STORE, ComputeType, 1, simplified>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, bool simplified>
inline cudaError_t DispatchLayerNormWarpImpl(cudaStream_t stream, LOAD load, STORE store,
                                             const int64_t rows, const int64_t cols,
                                             const double epsilon, ComputeType* mean,
                                             ComputeType* inv_variance) {
  return DispatchLayerNormWarpImplPackSize<LOAD, STORE, ComputeType, simplified>()(
      stream, load, store, rows, cols, epsilon, mean, inv_variance);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, bool simplified, int block_size>
__global__ void LayerNormBlockSMemImpl(LOAD load, STORE store, const int64_t rows,
                                       const int64_t cols, const double epsilon, ComputeType* mean,
                                       ComputeType* inv_variance) {
  extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
  auto* buf = reinterpret_cast<ComputeType*>(shared_buf);
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = static_cast<int>(cols) / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_mean = 0;
    ComputeType thread_m2 = 0;
    ComputeType thread_count = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        buf[i * num_packs + pack_id] = pack[i];
        WelfordCombine<ComputeType, simplified>(pack[i], &thread_mean, &thread_m2, &thread_count);
      }
    }
    ComputeType row_mean = 0;
    ComputeType row_m2 = 0;
    ComputeType row_count = 0;
    WelfordBlockAllReduce<ComputeType, simplified>(thread_mean, thread_m2, thread_count, &row_mean, &row_m2,
                                       &row_count);
    ComputeType row_variance = max(Div(row_m2, row_count), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) {
      if (mean != nullptr) { mean[row] = row_mean; }
      if (inv_variance != nullptr) { inv_variance[row] = row_inv_var; }
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (simplified) {
          pack[i] = buf[i * num_packs + pack_id] * row_inv_var;
        } else {
          pack[i] = (buf[i * num_packs + pack_id] - row_mean) * row_inv_var;
        }
      }
      store.template store<pack_size>(pack, row, pack_id * pack_size);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, bool simplified, int block_size>
inline cudaError_t LaunchLayerNormBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                                int smem, const int64_t rows, const int64_t cols,
                                                const double epsilon, ComputeType* mean,
                                                ComputeType* inv_variance) {
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err =
        GetNumBlocks(LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, simplified, block_size>,
                     block_size, smem, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, simplified, block_size>
      <<<grid_dim_x, block_size, smem, stream>>>(load, store, rows, cols, epsilon, mean,
                                                 inv_variance);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, bool simplified>
inline cudaError_t TryDispatchLayerNormBlockSMemImplBlockSize(
    cudaStream_t stream, LOAD load, STORE store, const int64_t rows, const int64_t cols,
    const double epsilon, ComputeType* mean, ComputeType* inv_variance, bool* success) {
  constexpr int block_size_conf_1 = 128;
  constexpr int block_size_conf_2 = 256;
  constexpr int block_size_conf_3 = 512;
  constexpr int block_size_conf_4 = 1024;
  const size_t smem = cols * sizeof(ComputeType);
  int max_active_blocks_conf_1;

  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_1,
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, simplified, block_size_conf_1>,
        block_size_conf_1, smem);
    if (err != cudaSuccess) { return err; }
  }
  if (max_active_blocks_conf_1 <= 0) {
    *success = false;
    return cudaSuccess;
  }
  int max_active_blocks_conf_4;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_4,
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, simplified, block_size_conf_4>,
        block_size_conf_4, smem);
    if (err != cudaSuccess) { return err; }
  }

  if (max_active_blocks_conf_4 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, simplified, block_size_conf_4>(
        stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
  }
  int max_active_blocks_conf_3;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_3,
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, simplified, block_size_conf_3>,
        block_size_conf_3, smem);
    if (err != cudaSuccess) { return err; }
  }

  if (max_active_blocks_conf_3 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, simplified, block_size_conf_3>(
        stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
  }
  int max_active_blocks_conf_2;
  {
    cudaError_t err = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks_conf_2,
        LayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, simplified, block_size_conf_2>,
        block_size_conf_2, smem);
    if (err != cudaSuccess) { return err; }
  }

  if (max_active_blocks_conf_2 == max_active_blocks_conf_1) {
    *success = true;
    return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, simplified, block_size_conf_2>(
        stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
  }
  *success = true;
  return LaunchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, pack_size, simplified, block_size_conf_1>(
      stream, load, store, smem, rows, cols, epsilon, mean, inv_variance);
}

template<typename LOAD, typename STORE, typename ComputeType, bool simplified>
struct TryDispatchLayerNormBlockSMemImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, ComputeType* mean,
                         ComputeType* inv_variance, bool* success) {
    if (cols % 4 == 0) {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 4, simplified>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
    } else if (cols % 2 == 0) {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 2, simplified>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
    } else {
      return TryDispatchLayerNormBlockSMemImplBlockSize<LOAD, STORE, ComputeType, 1, simplified>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, bool simplified>
inline cudaError_t TryDispatchLayerNormBlockSMemImpl(cudaStream_t stream, LOAD load, STORE store,
                                                     const int64_t rows, const int64_t cols,
                                                     const double epsilon, ComputeType* mean,
                                                     ComputeType* inv_variance, bool* success) {
  return TryDispatchLayerNormBlockSMemImplPackSize<LOAD, STORE, ComputeType, simplified>()(
      stream, load, store, rows, cols, epsilon, mean, inv_variance, success);
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, bool simplified, int block_size>
__global__ void LayerNormBlockUncachedImpl(LOAD load, STORE store, const int64_t rows,
                                           const int64_t cols, const double epsilon,
                                           ComputeType* mean, ComputeType* inv_variance) {
  const int tid = threadIdx.x;
  assert(cols % pack_size == 0);
  const int num_packs = static_cast<int>(cols) / pack_size;
  for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
    ComputeType thread_mean = 0;
    ComputeType thread_m2 = 0;
    ComputeType thread_count = 0;
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      load.template load<pack_size>(pack, row, pack_id * pack_size);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        WelfordCombine<ComputeType, simplified>(pack[i], &thread_mean, &thread_m2, &thread_count);
      }
    }
    ComputeType row_mean = 0;
    ComputeType row_m2 = 0;
    ComputeType row_count = 0;
    WelfordBlockAllReduce<ComputeType, simplified>(thread_mean, thread_m2, thread_count, &row_mean, &row_m2,
                                       &row_count);
    ComputeType row_variance = max(Div(row_m2, row_count), static_cast<ComputeType>(0.0));
    ComputeType row_inv_var = rsqrt(row_variance + static_cast<ComputeType>(epsilon));
    if (threadIdx.x == 0) {
      if (mean != nullptr) { mean[row] = row_mean; }
      if (inv_variance != nullptr) { inv_variance[row] = row_inv_var; }
    }
    for (int pack_id = tid; pack_id < num_packs; pack_id += block_size) {
      ComputeType pack[pack_size];
      const int pack_offset = pack_id * pack_size;
      load.template load<pack_size>(pack, row, pack_offset);
#pragma unroll
      for (int i = 0; i < pack_size; ++i) {
        if (simplified) {
          pack[i] = pack[i] * row_inv_var; 
        } else {
          pack[i] = (pack[i] - row_mean) * row_inv_var; 
        }
      }
      store.template store<pack_size>(pack, row, pack_offset);
    }
  }
}

template<typename LOAD, typename STORE, typename ComputeType, int pack_size, bool simplified>
inline cudaError_t LaunchLayerNormBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                    const int64_t rows, const int64_t cols,
                                                    const double epsilon, ComputeType* mean,
                                                    ComputeType* inv_variance) {
  constexpr int block_size = 1024;
  constexpr int waves = 32;
  int grid_dim_x;
  {
    cudaError_t err =
        GetNumBlocks(LayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, simplified, block_size>,
                     block_size, 0, rows, waves, &grid_dim_x);
    if (err != cudaSuccess) { return err; }
  }
  LayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, pack_size, simplified, block_size>
      <<<grid_dim_x, block_size, 0, stream>>>(load, store, rows, cols, epsilon, mean, inv_variance);
  return cudaPeekAtLastError();
}

template<typename LOAD, typename STORE, typename ComputeType, bool simplified>
struct DispatchLayerNormBlockUncachedImplPackSize {
  cudaError_t operator()(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                         const int64_t cols, const double epsilon, ComputeType* mean,
                         ComputeType* inv_variance) {
    if (cols % 4 == 0) {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 4, simplified>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else if (cols % 2 == 0) {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 2, simplified>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    } else {
      return LaunchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, 1, simplified>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
  }
};

template<typename LOAD, typename STORE, typename ComputeType, bool simplified>
inline cudaError_t DispatchLayerNormBlockUncachedImpl(cudaStream_t stream, LOAD load, STORE store,
                                                      const int64_t rows, const int64_t cols,
                                                      const double epsilon, ComputeType* mean,
                                                      ComputeType* inv_variance) {
  return DispatchLayerNormBlockUncachedImplPackSize<LOAD, STORE, ComputeType, simplified>()(
      stream, load, store, rows, cols, epsilon, mean, inv_variance);
}

template<typename LOAD, typename STORE, typename ComputeType, bool simplified = false>
inline typename std::enable_if<!std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLayerNorm(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                  const int64_t cols, const double epsilon, ComputeType* mean,
                  ComputeType* inv_variance) {
  if (cols <= 1024) {
    return DispatchLayerNormWarpImpl<LOAD, STORE, ComputeType, simplified>(stream, load, store, rows, cols,
                                                               epsilon, mean, inv_variance);
  } else {
    bool dispatch_smem_impl_success;
    {
      cudaError_t err = TryDispatchLayerNormBlockSMemImpl<LOAD, STORE, ComputeType, simplified>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance,
          &dispatch_smem_impl_success);
      if (err != cudaSuccess) { return err; }
    }
    if (!dispatch_smem_impl_success) {
      return DispatchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, simplified>(
          stream, load, store, rows, cols, epsilon, mean, inv_variance);
    }
    return cudaSuccess;
  }
}

template<typename LOAD, typename STORE, typename ComputeType, bool simplified = false>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchLayerNorm(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                  const int64_t cols, const double epsilon, ComputeType* mean,
                  ComputeType* inv_variance) {
  return DispatchLayerNormBlockUncachedImpl<LOAD, STORE, ComputeType, simplified>(
      stream, load, store, rows, cols, epsilon, mean, inv_variance);
}


}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
