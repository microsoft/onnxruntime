// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "trilu_impl.h"
#include <stdio.h>
namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void TriluKernel(
    bool upper,
    int64_t k,
    const TArray<int64_t>& input_dims,
    const T* input_data,
    T* output_data,
    const CUDA_LONG N,
    const fast_divmod devmod_indices) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  int i;
  int j;
  devmod_indices.divmod(id, j, i);
  output_data[id] = ((((i + k) >= j) && upper) || (((i + k) <= j) && !upper)) ? input_data[id] : 0;
}

Status TriluImpl(
    cudaStream_t stream,
    bool upper,
    size_t element_size,
    int64_t k,
    const TArray<int64_t>& input_dims,
    const void* input_data,
    void* output_data,
    int N,
    const fast_divmod& divmod_indices) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  switch (element_size) {
    case sizeof(int8_t):
      TriluKernel<int8_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          upper, k,
          input_dims,
          reinterpret_cast<const ToCudaType<int8_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int8_t>::MappedType*>(output_data),
          (CUDA_LONG)N,
          divmod_indices);
      break;
    case sizeof(int16_t):
      TriluKernel<int16_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          upper, k,
          input_dims,
          reinterpret_cast<const ToCudaType<int16_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int16_t>::MappedType*>(output_data),
          (CUDA_LONG)N,
          divmod_indices);
      break;
    case sizeof(int32_t):
      TriluKernel<int32_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          upper, k,
          input_dims,
          reinterpret_cast<const ToCudaType<int32_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int32_t>::MappedType*>(output_data),
          (CUDA_LONG)N,
          divmod_indices);
      break;
    case sizeof(int64_t):
      TriluKernel<int64_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
          upper, k,
          input_dims,
          reinterpret_cast<const ToCudaType<int64_t>::MappedType*>(input_data),
          reinterpret_cast<ToCudaType<int64_t>::MappedType*>(output_data),
          (CUDA_LONG)N,
          divmod_indices);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
                             element_size);
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime