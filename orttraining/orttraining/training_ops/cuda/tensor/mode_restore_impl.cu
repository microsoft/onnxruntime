// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/mode_restore_impl.h"
#include "core/providers/cuda/cu_inc/bitmask.cuh"
#include <cub/cub.cuh>
#include "contrib_ops/cuda/bert/utils.cuh"
namespace onnxruntime {
namespace cuda {

void GetZeroPointRestoreTempStorageBytesImpl(cudaStream_t stream,
                                             size_t& temp_storage_bytes,
                                             int total_element_count) {
  cub::DeviceScan::InclusiveSum(
      static_cast<void*>(nullptr),  // input, when NULL, the required allocation size is written to temp_storage_bytes and no work is done.
      temp_storage_bytes,           // input or output
      static_cast<int*>(nullptr),   // input
      static_cast<int*>(nullptr),   // output
      total_element_count,          // input
      stream);
}

void CalculateInputOffsetForEachOutputImpl(cudaStream_t stream,
                                           void* d_temp_storage,
                                           size_t& temp_storage_bytes,
                                           int* restored_output_mask,
                                           int* output_idx_to_input_idx_map_buffer,
                                           int total_element_count) {
  cub::DeviceScan::InclusiveSum(
      d_temp_storage,                      // input, when NULL, the required allocation size is written to temp_storage_bytes and no work is done.
      temp_storage_bytes,                  // input or output
      restored_output_mask,                // input
      output_idx_to_input_idx_map_buffer,  // output
      total_element_count,                 // input
      stream);
}

constexpr int kBlockSize = 256;
constexpr int kNumUnroll = 4;

__global__ void FillOutputFromMaskKernel(const CUDA_LONG N,
                                         const fast_divmod fdm_bits_per_element,
                                         const BitmaskElementType* mask_data,
                                         int* restored_output_mask) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG id = idx * kNumUnroll;

  // int masks[kNumUnroll];
  if (id < N) {
    int bitmask_idx, bitmask_shift;
    fdm_bits_per_element.divmod(id, bitmask_idx, bitmask_shift);
    BitmaskElementType shifted_mask = mask_data[bitmask_idx] >> bitmask_shift;
#pragma unroll
    for (int i = 0; i < kNumUnroll; i++) {
      CUDA_LONG li = id + i;
      if (li < N) {
        restored_output_mask[li] = ((shifted_mask & (1 << i)) != 0);
      }
    }
  }

  // #pragma unroll
  //   for (int i = 0; i < kNumUnroll; ++i) {
  //     CUDA_LONG li = id + i;
  //     if (li < N) {
  //       restored_output_mask[li] = masks[i];
  //       // printf("restored_output_mask[%d] = %d \n", static_cast<int>(li), restored_output_mask[li]);
  //     }
  //   }
}

__global__ void FillOutputFromMaskVectorizedKernel(const CUDA_LONG N,
                                                   const fast_divmod fdm_bits_per_element,
                                                   const BitmaskElementType* mask_data,
                                                   int* restored_output_mask) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG id = idx * kNumUnroll;

  using LoadT = aligned_vector<int, kNumUnroll>;

  if (id < N) {
    int bitmask_idx, bitmask_shift;
    fdm_bits_per_element.divmod(id, bitmask_idx, bitmask_shift);
    BitmaskElementType shifted_mask = mask_data[bitmask_idx] >> bitmask_shift;

    int masks[kNumUnroll];
#pragma unroll
    for (int i = 0; i < kNumUnroll; i++) {
      masks[i] = ((shifted_mask & (1 << i)) != 0);
    }

    // Vectorized writes for mask_data & Y_data
    *(reinterpret_cast<LoadT*>(&restored_output_mask[id])) = *reinterpret_cast<LoadT*>(&masks[0]);
  }
}

void FillOutputFromMaskImpl(cudaStream_t stream,
                            const int64_t total_element_count,
                            const BitmaskElementType* mask_data,
                            int* restored_output_mask) {
  const int blocksPerGrid = static_cast<int>(CeilDiv(total_element_count, kBlockSize * kNumUnroll));
  fast_divmod fdm_bits_per_element(kNumBitsPerBitmaskElement);
  if (total_element_count % kNumUnroll != 0) {
    FillOutputFromMaskKernel<<<blocksPerGrid, kBlockSize, 0, stream>>>(
        static_cast<CUDA_LONG>(total_element_count), fdm_bits_per_element, mask_data, restored_output_mask);
  } else {
    FillOutputFromMaskVectorizedKernel<<<blocksPerGrid, kBlockSize, 0, stream>>>(
        static_cast<CUDA_LONG>(total_element_count), fdm_bits_per_element, mask_data, restored_output_mask);
  }
}

template <typename T>
__global__ void RestoreFromMaskKernel(const CUDA_LONG N,
                                      const fast_divmod fdm_bits_per_element,
                                      const float zero_point_value,
                                      const T* input_data,
                                      const int* output_idx_to_input_idx_map_buffer,
                                      T* output_data) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG id = idx * kNumUnroll;

  int maps[kNumUnroll + 1];
  if (id < N) {
    if (id == 0) {
      maps[0] = 0;
    } else {
      maps[0] = output_idx_to_input_idx_map_buffer[id - 1];
    }

#pragma unroll
    for (int i = 0; i < kNumUnroll; ++i) {
      CUDA_LONG li = id + i;
      if (li < N) {
        maps[i + 1] = output_idx_to_input_idx_map_buffer[li];
      }
    }
  }

#pragma unroll
  for (int i = 0; i < kNumUnroll; ++i) {
    CUDA_LONG li = id + i;
    if (li < N) {
      int map_value = maps[i + 1] - maps[i];
      output_data[li] = map_value == 1 ? input_data[maps[i]] : static_cast<T>(zero_point_value);
      // printf("output_data[%d] = %f, map_value: %d, maps[i + 1]: %d \n", static_cast<int>(li), static_cast<float>(output_data[li]), map_value, maps[i + 1]);
    }
  }
}

template <typename T>
__global__ void RestoreFromMaskVectorizedKernel(const CUDA_LONG N,
                                                const fast_divmod fdm_bits_per_element,
                                                const float zero_point_value,
                                                const T* input_data,
                                                const int* output_idx_to_input_idx_map_buffer,
                                                T* output_data) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;

  using LoadT = aligned_vector<int, kNumUnroll>;
  using LoadT2 = aligned_vector<T, kNumUnroll>;

  CUDA_LONG id = idx * kNumUnroll;

  int maps[kNumUnroll + 1];
  if (id < N) {
    if (id == 0) {
      maps[0] = 0;
    } else {
      maps[0] = output_idx_to_input_idx_map_buffer[id - 1];
    }

    // vectorized load into storage
    LoadT* value = reinterpret_cast<LoadT*>(&maps[1]);
    *value = *reinterpret_cast<const LoadT*>(&output_idx_to_input_idx_map_buffer[id]);

    float cached_input[kNumUnroll];
#pragma unroll
    for (int i = 0; i < kNumUnroll; ++i) {
      cached_input[i] = static_cast<float>(input_data[maps[i]]);
    }

    T dest[kNumUnroll];
#pragma unroll
    for (int i = 0; i < kNumUnroll; ++i) {
      int map_value = maps[i + 1] - maps[i];
      dest[i] = static_cast<T>(onnxruntime::cuda::fma(static_cast<float>(map_value != 0), cached_input[i], zero_point_value));
      // printf("output_data[%d] = %f, map_value: %d, maps[i + 1]: %d \n", static_cast<int>(li), static_cast<float>(output_data[li]), map_value, maps[i + 1]);
    }

    // Vectorized writes for mask_data & Y_data
    *(reinterpret_cast<LoadT2*>(&output_data[id])) = *reinterpret_cast<LoadT2*>(&dest[0]);
  }
}

template <typename T>
void RestoreFromMaskImpl(const cudaDeviceProp& prop,
                         cudaStream_t stream,
                         const int64_t total_element_count,
                         const float zero_point_value,
                         const T* input_data,
                         const int* output_idx_to_input_idx_map_buffer,
                         T* output_data) {
  const int blocksPerGrid = static_cast<int>(CeilDiv(total_element_count, kBlockSize * kNumUnroll));
  fast_divmod fdm_bits_per_element(kNumBitsPerBitmaskElement);
  if (total_element_count % kNumUnroll != 0) {
    RestoreFromMaskKernel<T><<<blocksPerGrid, kBlockSize, 0, stream>>>(
        static_cast<CUDA_LONG>(total_element_count),
        fdm_bits_per_element,
        zero_point_value,
        input_data,
        output_idx_to_input_idx_map_buffer,
        output_data);
  } else {
    RestoreFromMaskVectorizedKernel<T><<<blocksPerGrid, kBlockSize, 0, stream>>>(
        static_cast<CUDA_LONG>(total_element_count),
        fdm_bits_per_element,
        zero_point_value,
        input_data,
        output_idx_to_input_idx_map_buffer,
        output_data);
  }
}

#define SPECIALIZED_RESTORE_FROM_MASK_IMPL(T)                                         \
  template void RestoreFromMaskImpl<T>(const cudaDeviceProp& prop,                    \
                                       cudaStream_t stream,                           \
                                       const int64_t total_element_count,             \
                                       const float zero_point_value,                  \
                                       const T* input_data,                           \
                                       const int* output_idx_to_input_idx_map_buffer, \
                                       T* output_data);

SPECIALIZED_RESTORE_FROM_MASK_IMPL(float)
SPECIALIZED_RESTORE_FROM_MASK_IMPL(double)
SPECIALIZED_RESTORE_FROM_MASK_IMPL(half)
SPECIALIZED_RESTORE_FROM_MASK_IMPL(BFloat16)

#undef SPECIALIZED_RESTORE_FROM_MASK_IMPL

}  // namespace cuda
}  // namespace onnxruntime
