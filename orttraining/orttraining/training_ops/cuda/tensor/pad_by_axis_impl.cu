// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/pad_by_axis_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

constexpr int kBlockSize = 256;
constexpr int kNumUnroll = 4;

template <typename T>
__global__ void RestoreFromMaskKernel(const CUDA_LONG N,
                                      const fast_divmod output_element_stride_fdm,
                                      const T* input_data,
                                      const int64_t* indices_data,
                                      T* output_data) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG id = idx * kNumUnroll;

  T input[kNumUnroll];
  if (id < N) {
    // int64_t indices[kNumUnroll];
    // int col_indices[kNumUnroll];

#pragma unroll
    for (int i = 0; i < kNumUnroll; ++i) {
      CUDA_LONG li = id + i;
      if (li < N) {
        input[i] = input_data[li];
        // indices[i] = indices_data[row_index];
      }
    }
  }

#pragma unroll
  for (int i = 0; i < kNumUnroll; ++i) {
    CUDA_LONG li = id + i;
    if (li < N) {
      int row_index, col_index;
      output_element_stride_fdm.divmod(li, row_index, col_index);
      output_data[indices_data[row_index] * output_element_stride_fdm.d_ + col_index] = input[i];
    }
  }
}

// template <typename T>
// __global__ void RestoreFromMaskVectorizedKernel(const CUDA_LONG N,
//                                                 const fast_divmod output_element_stride_fdm,
//                                                 const T* input_data,
//                                                 const int64_t* indices_data,
//                                                 T* output_data) {
//   CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;

//   using LoadT = aligned_vector<int64_t, kNumUnroll>;
//   using LoadT2 = aligned_vector<T, kNumUnroll>;

//   CUDA_LONG id = idx * kNumUnroll;

//   if (id < N) {
//     int row_index;
//     int col_index;
//     output_element_stride_fdm.div(id, row_index, col_index);

//     T input[kNumUnroll];
//     int64_t indices[kNumUnroll];
//     T output[kNumUnroll];

//     // vectorized load into storage
//     LoadT* value = reinterpret_cast<LoadT*>(&indices[0]);
//     *value = *reinterpret_cast<const LoadT*>(&indices_data[row_index]);

//     LoadT2* value2 = reinterpret_cast<LoadT2*>(&input[0]);
//     *value2 = *reinterpret_cast<const LoadT2*>(&input_data[row_index] + col_index);

// #pragma unroll
//     for (int i = 0; i < kNumUnroll; ++i) {
//       output_data[indices[i] * output_element_stride_fdm.d_ + col_index] = input[i];
//     }
//   }
// }

template <typename T>
void PadByAxisImpl(cudaStream_t stream,
                   const int64_t total_element_count,
                   const fast_divmod output_element_stride_fdm,
                   const T* input_data,
                   const int64_t* indices_data,
                   T* output_data) {
  const int blocksPerGrid = static_cast<int>(CeilDiv(total_element_count, kBlockSize * kNumUnroll));

  //   if (total_element_count % kNumUnroll != 0) {
  RestoreFromMaskKernel<T><<<blocksPerGrid, kBlockSize, 0, stream>>>(
      static_cast<CUDA_LONG>(total_element_count),
      output_element_stride_fdm,
      input_data,
      indices_data,
      output_data);
  //   } else {
  //     RestoreFromMaskVectorizedKernel<T><<<blocksPerGrid, kBlockSize, 0, stream>>>(
  //         static_cast<CUDA_LONG>(total_element_count),
  //         output_element_stride_fdm,
  //         input_data,
  //         indices_data,
  //         output_data);
  //   }
}

#define SPECIALIZED_RESTORE_FROM_MASK_IMPL(T)                                 \
  template void PadByAxisImpl<T>(cudaStream_t stream,                         \
                                 const int64_t total_element_count,           \
                                 const fast_divmod output_element_stride_fdm, \
                                 const T* input_data,                         \
                                 const int64_t* indices_data,                 \
                                 T* output_data);

SPECIALIZED_RESTORE_FROM_MASK_IMPL(float)
SPECIALIZED_RESTORE_FROM_MASK_IMPL(double)
SPECIALIZED_RESTORE_FROM_MASK_IMPL(half)
SPECIALIZED_RESTORE_FROM_MASK_IMPL(BFloat16)

#undef SPECIALIZED_RESTORE_FROM_MASK_IMPL

}  // namespace cuda
}  // namespace onnxruntime
