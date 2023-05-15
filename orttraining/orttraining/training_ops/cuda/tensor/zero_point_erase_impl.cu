// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/zero_point_erase_impl.h"
#include "core/providers/cuda/cu_inc/bitmask.cuh"
#include <cub/cub.cuh>

namespace onnxruntime {
namespace cuda {

template <typename T>
struct NotEqualCubOp {
  float zero_point_value_;
  __host__ __device__ __forceinline__ NotEqualCubOp(float zero_point_value)
      : zero_point_value_(zero_point_value) {}

  __host__ __device__ __forceinline__ bool operator()(const T& val) const {
    return (val != static_cast<T>(zero_point_value_));
  }
};

template <typename T>
void GetTempStorageBytesImpl(cudaStream_t stream,
                             size_t& temp_storage_bytes,
                             float zero_point_value,
                             int total_element_count) {
  NotEqualCubOp<T> select_op(zero_point_value);
  cub::DeviceSelect::If(
      static_cast<void*>(nullptr),  // input, when NULL, the required allocation size is written to temp_storage_bytes and no work is done.
      temp_storage_bytes,           // input or output
      static_cast<T*>(nullptr),     // input
      static_cast<T*>(nullptr),     // output
      static_cast<int*>(nullptr),   // output
      total_element_count,          // input
      select_op,
      stream);
}

template <typename T>
void CopyOnConditionImpl(cudaStream_t stream,
                         void* d_temp_storage,
                         size_t& temp_storage_bytes,
                         const T* input_data,
                         T* output_buffer,
                         int& d_num_selected_out,
                         float zero_point_value,
                         int total_element_count) {
  NotEqualCubOp<T> select_op(zero_point_value);
  cub::DeviceSelect::If(
      d_temp_storage,       // input, when NULL, the required allocation size is written to temp_storage_bytes and no work is done.
      temp_storage_bytes,   // input or output
      input_data,           // input
      output_buffer,        // output
      &d_num_selected_out,  // output
      total_element_count,  // input
      select_op,
      stream);
}

constexpr int kBlockSize = 256;
constexpr int kNumUnroll = 4;

template <typename T>
__global__ void SetMaskOutputKernel(const CUDA_LONG N,
                                    const CUDA_LONG mask_element_count,
                                    const int step_size,
                                    const int steps_per_thread,
                                    const fast_divmod fdm_bits_per_element,
                                    const float zero_point_value,
                                    const T* X_data,
                                    void* mask_data) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;

  for (int i = 0; i < steps_per_thread; ++i) {
    CUDA_LONG id = idx * kNumUnroll + i * step_size;
    BitmaskElementType thread_bitmask = 0;

#pragma unroll
    for (int i = 0; i < kNumUnroll; ++i) {
      CUDA_LONG li = id + i;
      if (li < N) {
        thread_bitmask |= (X_data[li] != static_cast<T>(zero_point_value));
      }
    }

    SetBitmask<kNumUnroll>(id, mask_element_count, fdm_bits_per_element, thread_bitmask,
                           reinterpret_cast<BitmaskElementType*>(mask_data));

    __syncthreads();
  }
}

template <typename T>
__global__ void SetMaskOutputVectorizedKernel(const CUDA_LONG N,
                                              const CUDA_LONG mask_element_count,
                                              const int step_size,
                                              const int steps_per_thread,
                                              const fast_divmod fdm_bits_per_element,
                                              const float zero_point_value,
                                              const T* X_data,
                                              void* mask_data) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;

  using LoadT = aligned_vector<T, kNumUnroll>;
  using MaskLoadT = aligned_vector<bool, kNumUnroll>;

  for (int i = 0; i < steps_per_thread; ++i) {
    CUDA_LONG id = idx * kNumUnroll + i * step_size;

    BitmaskElementType thread_bitmask = 0;
    if (id < N) {
      // vectorized load into storage
      T src[kNumUnroll];
      LoadT* value = reinterpret_cast<LoadT*>(&src);
      *value = *reinterpret_cast<const LoadT*>(&X_data[id]);

// actual computation
#pragma unroll
      for (int ii = 0; ii < kNumUnroll; ++ii) {
        thread_bitmask |= (src[ii] != static_cast<T>(zero_point_value));
      }
    }

    SetBitmask<kNumUnroll>(id, mask_element_count, fdm_bits_per_element, thread_bitmask,
                           reinterpret_cast<BitmaskElementType*>(mask_data));

    __syncthreads();
  }
}

template <typename T>
void SetMaskOutputImpl(const cudaDeviceProp& prop,
                       cudaStream_t stream,
                       const int64_t N,
                       const int64_t mask_element_count,
                       const float zero_point_value,
                       const T* X_data,
                       void* mask_data) {
  const int blocks_per_sm = prop.maxThreadsPerMultiProcessor / kBlockSize;
  const int grid_size =
      std::min(prop.multiProcessorCount * blocks_per_sm, static_cast<int>(CeilDiv(N, kBlockSize * kNumUnroll)));

  const int step_size = kBlockSize * grid_size * kNumUnroll;
  const int steps_per_thread = static_cast<int>(CeilDiv(N, step_size));

  fast_divmod fdm_bits_per_element(kNumBitsPerBitmaskElement);
  // Using vectorized data load/store approach when N % 4 == 0 since this is
  // typical case for input shape size
  if (N % kNumUnroll != 0) {
    SetMaskOutputKernel<T><<<grid_size, kBlockSize, 0, stream>>>(
        static_cast<CUDA_LONG>(N), static_cast<CUDA_LONG>(mask_element_count), step_size, steps_per_thread,
        fdm_bits_per_element, zero_point_value, X_data, mask_data);
  } else {
    SetMaskOutputVectorizedKernel<T><<<grid_size, kBlockSize, 0, stream>>>(
        static_cast<CUDA_LONG>(N), static_cast<CUDA_LONG>(mask_element_count), step_size, steps_per_thread,
        fdm_bits_per_element, zero_point_value, X_data, mask_data);
  }
}

#define SPECIALIZED_ZERO_POINT_CLIP_IMPL(T)                                                                           \
  template void SetMaskOutputImpl<T>(const cudaDeviceProp& prop, cudaStream_t stream, const int64_t N,                \
                                     const int64_t mask_element_count, const float zero_point_value, const T* X_data, \
                                     void* mask_data);

SPECIALIZED_ZERO_POINT_CLIP_IMPL(float)
SPECIALIZED_ZERO_POINT_CLIP_IMPL(double)
SPECIALIZED_ZERO_POINT_CLIP_IMPL(half)
SPECIALIZED_ZERO_POINT_CLIP_IMPL(BFloat16)

#undef SPECIALIZED_ZERO_POINT_CLIP_IMPL

}  // namespace cuda
}  // namespace onnxruntime
