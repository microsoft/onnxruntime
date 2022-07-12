// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "trilu_impl.h"
#include <stdio.h>
namespace onnxruntime {
namespace cuda {

template <typename T, bool upper>
__global__ void TriluKernel(
    int64_t k,
    const T* input_data,
    T* output_data,
    const CUDA_LONG N,
    const fast_divmod batch_divmod_indices,
    const fast_divmod row_col_divmod_indices) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);

  int row, col;

  row_col_divmod_indices.divmod(batch_divmod_indices.mod(id), row, col);
  output_data[id] = upper ? (((row + k) <= col) ? input_data[id] : 0) : (((row + k) >= col) ? input_data[id] : 0);
}

Status TriluImpl(
    cudaStream_t stream,
    bool upper,
    size_t element_size,
    int64_t k,
    const void* input_data,
    void* output_data,
    int N,
    const fast_divmod& batch_divmod_indices,
    const fast_divmod& row_col_divmod_indices) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  switch (element_size) {
    case sizeof(int8_t):
      if (upper) {
        TriluKernel<int8_t, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            k,
            reinterpret_cast<const ToCudaType<int8_t>::MappedType*>(input_data),
            reinterpret_cast<ToCudaType<int8_t>::MappedType*>(output_data),
            (CUDA_LONG)N,
            batch_divmod_indices,
            row_col_divmod_indices);
      } else {
        TriluKernel<int8_t, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            k,
            reinterpret_cast<const ToCudaType<int8_t>::MappedType*>(input_data),
            reinterpret_cast<ToCudaType<int8_t>::MappedType*>(output_data),
            (CUDA_LONG)N,
            batch_divmod_indices,
            row_col_divmod_indices);
      }
      break;
    case sizeof(int16_t):
      if (upper) {
        TriluKernel<int16_t, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            k,
            reinterpret_cast<const ToCudaType<int16_t>::MappedType*>(input_data),
            reinterpret_cast<ToCudaType<int16_t>::MappedType*>(output_data),
            (CUDA_LONG)N,
            batch_divmod_indices,
            row_col_divmod_indices);
      } else {
        TriluKernel<int16_t, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            k,
            reinterpret_cast<const ToCudaType<int16_t>::MappedType*>(input_data),
            reinterpret_cast<ToCudaType<int16_t>::MappedType*>(output_data),
            (CUDA_LONG)N,
            batch_divmod_indices,
            row_col_divmod_indices);
      }
      break;
    case sizeof(int32_t):
      if (upper) {
        TriluKernel<int32_t, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            k,
            reinterpret_cast<const ToCudaType<int32_t>::MappedType*>(input_data),
            reinterpret_cast<ToCudaType<int32_t>::MappedType*>(output_data),
            (CUDA_LONG)N,
            batch_divmod_indices,
            row_col_divmod_indices);
      } else {
        TriluKernel<int32_t, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            k,
            reinterpret_cast<const ToCudaType<int32_t>::MappedType*>(input_data),
            reinterpret_cast<ToCudaType<int32_t>::MappedType*>(output_data),
            (CUDA_LONG)N,
            batch_divmod_indices,
            row_col_divmod_indices);
      }
      break;
    case sizeof(int64_t):
      if (upper) {
        TriluKernel<int64_t, true><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            k,
            reinterpret_cast<const ToCudaType<int64_t>::MappedType*>(input_data),
            reinterpret_cast<ToCudaType<int64_t>::MappedType*>(output_data),
            (CUDA_LONG)N,
            batch_divmod_indices,
            row_col_divmod_indices);
      } else {
        TriluKernel<int64_t, false><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0, stream>>>(
            k,
            reinterpret_cast<const ToCudaType<int64_t>::MappedType*>(input_data),
            reinterpret_cast<ToCudaType<int64_t>::MappedType*>(output_data),
            (CUDA_LONG)N,
            batch_divmod_indices,
            row_col_divmod_indices);
      }
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
                             element_size);
  }
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime