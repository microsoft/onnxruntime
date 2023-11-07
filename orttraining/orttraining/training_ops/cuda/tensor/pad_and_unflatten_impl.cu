// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/pad_and_unflatten_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

constexpr int kBlockSize = 256;
constexpr int kNumUnroll = 4;

template <typename T>
__global__ void FillOutputWithIndexKernel(const CUDA_LONG N,
                                          const fast_divmod output_element_stride_fdm,
                                          const int64_t index_value_upper_bound,
                                          const T* input_data,
                                          const int64_t* indices_data,
                                          T* output_data) {
  CUDA_LONG idx = blockDim.x * blockIdx.x + threadIdx.x;
  CUDA_LONG id = idx * kNumUnroll;

  T input[kNumUnroll];
  if (id < N) {
#pragma unroll
    for (int i = 0; i < kNumUnroll; ++i) {
      CUDA_LONG li = id + i;
      if (li < N) {
        input[i] = input_data[li];
      }
    }
  }

#pragma unroll
  for (int i = 0; i < kNumUnroll; ++i) {
    CUDA_LONG li = id + i;
    if (li < N) {
      int row_index, col_index;
      output_element_stride_fdm.divmod(li, row_index, col_index);
      assert(indices_data[row_index] < index_value_upper_bound);
      output_data[indices_data[row_index] * output_element_stride_fdm.d_ + col_index] = input[i];
    }
  }
}

template <typename T>
void PadAndUnflattenImpl(cudaStream_t stream,
                         const int64_t total_element_count,
                         const fast_divmod output_element_stride_fdm,
                         const int64_t index_value_upper_bound,
                         const T* input_data,
                         const int64_t* indices_data,
                         T* output_data) {
  const int blocksPerGrid = static_cast<int>(CeilDiv(total_element_count, kBlockSize * kNumUnroll));
  FillOutputWithIndexKernel<T><<<blocksPerGrid, kBlockSize, 0, stream>>>(
      static_cast<CUDA_LONG>(total_element_count),
      output_element_stride_fdm,
      index_value_upper_bound,
      input_data,
      indices_data,
      output_data);
}

#define PAD_AND_UNFLATTEN_FROM_MASK_IMPL(T)                                       \
  template void PadAndUnflattenImpl<T>(cudaStream_t stream,                         \
                                       const int64_t total_element_count,           \
                                       const fast_divmod output_element_stride_fdm, \
                                       const int64_t index_value_upper_bound,       \
                                       const T* input_data,                         \
                                       const int64_t* indices_data,                 \
                                       T* output_data);

PAD_AND_UNFLATTEN_FROM_MASK_IMPL(float)
PAD_AND_UNFLATTEN_FROM_MASK_IMPL(double)
PAD_AND_UNFLATTEN_FROM_MASK_IMPL(half)
PAD_AND_UNFLATTEN_FROM_MASK_IMPL(BFloat16)

#undef PAD_AND_UNFLATTEN_FROM_MASK_IMPL

}  // namespace cuda
}  // namespace onnxruntime
