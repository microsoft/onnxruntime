// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "transpose_impl.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void _TransposeKernel(int32_t rank, CUDA_LONG N,
                                 const TArray<int64_t> input_strides, const T* input_data,
                                 const TArray<fast_divmod> output_strides, T* output_data) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
  CUDA_LONG input_index = 0;
  CUDA_LONG output_index = id;

  #pragma unroll
  for (auto dim = 0; dim < MAX_ARRAY_SIZE; ++dim) {
    if (dim >= rank) {
      break;
    }
    int out_coord, r;
    output_strides.data_[dim].divmod(output_index, out_coord, r);
    output_index = r;
    input_index += input_strides.data_[dim] * out_coord;
  }
  output_data[id] = input_data[input_index];
}

<<<<<<< HEAD
template <typename T>
void TransposeImpl(int32_t rank, int64_t N,
                   const TArray<int64_t>& input_strides, const T* input_data,
                   const TArray<fast_divmod>& output_strides, T* output_data) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  _TransposeKernel<T><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
      rank, (CUDA_LONG)N, input_strides, input_data, output_strides, output_data);
}

#define SPECIALIZED_IMPL(T)                                                           \
  template void TransposeImpl<T>(int32_t rank, int64_t N,                             \
                      const TArray<int64_t>& input_strides, const T* input_data,      \
                      const TArray<fast_divmod>& output_strides, T* output_data);
=======
Status TransposeImpl(size_t element_size, size_t shape_rank, const int64_t* input_strides, const size_t* perm,
                     const void* input_data, const fast_divmod* fdm_output_strides, void* output_data, size_t N) {
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  switch (element_size) {
    case sizeof(int8_t):
      _TransposeKernel<int8_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          shape_rank, input_strides, perm,
          reinterpret_cast<const ToCudaType<int8_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToCudaType<int8_t>::MappedType*>(output_data),
          N);
      break;
    case sizeof(int16_t):
      _TransposeKernel<int16_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          shape_rank, input_strides, perm,
          reinterpret_cast<const ToCudaType<int16_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToCudaType<int16_t>::MappedType*>(output_data),
          N);
      break;
    case sizeof(int32_t):
      _TransposeKernel<int32_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          shape_rank, input_strides, perm,
          reinterpret_cast<const ToCudaType<int32_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToCudaType<int32_t>::MappedType*>(output_data),
          N);
      break;
    case sizeof(int64_t):
      _TransposeKernel<int64_t><<<blocksPerGrid, GridDim::maxThreadsPerBlock, 0>>>(
          shape_rank, input_strides, perm,
          reinterpret_cast<const ToCudaType<int64_t>::MappedType*>(input_data),
          fdm_output_strides,
          reinterpret_cast<ToCudaType<int64_t>::MappedType*>(output_data),
          N);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Type not supported for transpose on CUDA. Element size was ",
                             element_size);
  }
>>>>>>> c767e264c52c3bac2c319b630d37f541f4d2a677

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
