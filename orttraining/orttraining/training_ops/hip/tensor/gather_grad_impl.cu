#include "hip/hip_runtime.h"
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/hip/tensor/gather_grad_impl.h"
#include "core/providers/hip/cu_inc/common.cuh"
#include "core/providers/hip/shared_inc/hip_call.h"

#include <hipcub/hipcub.hpp>


namespace onnxruntime {
namespace hip {

template <typename T>
__global__ void _Iota(
    hipcub::CountingInputIterator<T> input,
    size_t length,
    T* output) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(idx, length);
  output[idx] = input[idx];
}

template <typename T, typename Tin>
__global__ void _GatherGradImpl(
    const Tin* input,
    const Tin* indices,
    const T* grad_output,
    T* grad_weight,
    int64_t numel,
    int64_t input_numel,
    int64_t param_itrs,
    int64_t stride) {
  int idx = blockIdx.x * 4 + threadIdx.y;

  const int SZ = 4;
  if (idx < numel && (idx == 0 || input[idx] != input[idx - 1])) {
    do {
      for (int itr = 0; itr < param_itrs; ++itr) {
        const int start_feature = threadIdx.x + blockIdx.y * blockDim.x * SZ;
        const int weight_row = itr * input_numel + ((int)input[idx]) * stride;  //the offset of the input
        const int grad_row = (itr * numel + ((int)indices[idx])) * stride;      //the offset of the gradient

        T gradient[SZ];
        T weight[SZ];

#pragma unroll
        for (int ii = 0; ii < SZ; ii++) {
          int feature_dim = start_feature + ii * GPU_WARP_SIZE;
          if (feature_dim < stride) {
            gradient[ii] = static_cast<T>(grad_output[grad_row + feature_dim]);
            weight[ii] = static_cast<T>(grad_weight[weight_row + feature_dim]);
          }
        }

#pragma unroll
        for (int ii = 0; ii < SZ; ii++) {
          weight[ii] += gradient[ii];
        }

#pragma unroll
        for (int ii = 0; ii < SZ; ii++) {
          int feature_dim = start_feature + ii * GPU_WARP_SIZE;
          if (feature_dim < stride) {
            grad_weight[weight_row + feature_dim] = static_cast<T>(weight[ii]);
          }
        }
      }
      idx++;
    } while (idx < numel && input[idx] == input[idx - 1]);
  }
}

template <typename T, typename Tin>
void GatherGradImpl(
    const HipKernel& hip_kernel,
    const T* grad_data,
    const Tin* indices_data,
    const int64_t num_indices,
    const int64_t num_weights,
    const int64_t stride,
    T* output_data,
    const int64_t num_inputs,  //The number of input elements starting from the gathering dimension
    const int64_t param_itrs   //The size of dimensions of the data before gathering dimension
    ) {
  // allocate intermediate buffers
  auto original_indices = hip_kernel.template GetScratchBuffer<Tin>(num_indices);

  // initialize original_indices with [0, num_indices)
  {
    const auto blocks_per_grid = CeilDiv(num_indices, GridDim::maxThreadsPerBlock);
    hipcub::CountingInputIterator<Tin> counting_input(Tin{});
    hipLaunchKernelGGL(_Iota, dim3(blocks_per_grid), dim3(GridDim::maxThreadsPerBlock), 0, 0, 
        counting_input, num_indices, original_indices.get());
  }

  auto indices_data_sorted = hip_kernel.template GetScratchBuffer<Tin>(num_indices);
  auto original_indices_sorted = hip_kernel.template GetScratchBuffer<Tin>(num_indices);

  // sort indices and original indices
  size_t sort_temp_storage_size_bytes = 0;
  HIP_CALL_THROW(hipcub::DeviceRadixSort::SortPairs(
      nullptr, sort_temp_storage_size_bytes,
      indices_data, indices_data_sorted.get(),
      original_indices.get(), original_indices_sorted.get(),
      num_indices));

  auto sort_temp_storage = hip_kernel.GetScratchBuffer<void>(sort_temp_storage_size_bytes);

  HIP_CALL_THROW(hipcub::DeviceRadixSort::SortPairs(
      sort_temp_storage.get(), sort_temp_storage_size_bytes,
      indices_data, indices_data_sorted.get(),
      original_indices.get(), original_indices_sorted.get(),
      num_indices));

  dim3 block(GPU_WARP_SIZE, 4);
  dim3 grid(CeilDiv(num_indices, 4), CeilDiv(stride, 128));

  hipLaunchKernelGGL(_GatherGradImpl, dim3(grid), dim3(block), 0, 0, 
      indices_data_sorted.get(),
      original_indices_sorted.get(),
      grad_data,
      output_data,
      num_indices,
      num_inputs,
      param_itrs,
      stride);
}

#define SPECIALIZED_GRAD_IMPL2(T)           \
  template void GatherGradImpl<T, int64_t>( \
      const HipKernel& hip_kernel,        \
      const T* grad_data,                   \
      const int64_t* indices_data,          \
      const int64_t num_indices,            \
      const int64_t num_weights,            \
      const int64_t stride,                 \
      T* output_data,                       \
      const int64_t num_inputs,             \
      const int64_t param_itrs);            \
  template void GatherGradImpl<T, int32_t>( \
      const HipKernel& hip_kernel,        \
      const T* grad_data,                   \
      const int32_t* indices_data,          \
      const int64_t num_indices,            \
      const int64_t num_weights,            \
      const int64_t stride,                 \
      T* output_data,                       \
      const int64_t num_inputs,             \
      const int64_t param_itrs);

SPECIALIZED_GRAD_IMPL2(float)
SPECIALIZED_GRAD_IMPL2(half)

}  // namespace hip
}  // namespace onnxruntime
