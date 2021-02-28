// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cu_inc/common.cuh"
#include "gather_elements_impl.h"

namespace onnxruntime {
namespace cuda {

namespace {
constexpr int threads_per_block = GridDim::maxThreadsPerBlock;
constexpr int thread_worksize = 16;
}  // namespace

__host__ __device__ inline int64_t GetIndexValue(const void* index_data, size_t index_element_size, size_t offset) {
  switch (index_element_size) {
    case sizeof(int32_t):
      return *(reinterpret_cast<const int32_t*>(index_data) + offset);
      break;
    case sizeof(int64_t):
      return *(reinterpret_cast<const int64_t*>(index_data) + offset);
      break;
    default:
      break;
  }
  // What is a sensible thing to do here?
  assert(false);
  return std::numeric_limits<int64_t>::max();
}

template <typename T>
__global__ void _GatherElementsKernel(
    const int64_t rank,
    const T* input_data,
    const int64_t input_dim_along_axis,
    const TArray<int64_t> input_strides,
    const void* indices_data,
    const int64_t indices_size,
    const size_t index_element_size,
    const TArray<fast_divmod> indices_strides,
    const int64_t axis,
    T* output_data) {

  CUDA_LONG indices_index = threads_per_block * thread_worksize * blockIdx.x + threadIdx.x;

  #pragma unroll
  for (int work = 0; work < thread_worksize; ++work) {
    if (indices_index < indices_size) {

      int dim = 0;
      int remain = indices_index;
      int64_t data_idx = 0;

      int i = 0;
      for (; i < axis && remain > 0; ++i) {
        indices_strides[i].divmod(remain, dim, remain);
        data_idx += input_strides[i] * dim;
      }

      i = axis;
      indices_strides[i].divmod(remain, dim, remain);
      dim = GetIndexValue(indices_data, index_element_size, indices_index);
      if (dim < -input_dim_along_axis || dim >= input_dim_along_axis) {
        return;  // Invalid index
      }

      if (dim < 0) {
        dim += input_dim_along_axis;
      }

      data_idx += input_strides[i] * dim;

      ++i;  // past axis
      for (; i < rank && remain > 0; ++i) {
        indices_strides[i].divmod(remain, dim, remain);
        data_idx += input_strides[i] * dim;
      }
      output_data[indices_index] = input_data[data_idx];

      indices_index += threads_per_block;
    }
  }
}

void GatherElementsImpl(
    cudaStream_t stream,
    const int64_t rank,
    const void* input_data,
    const int64_t input_dim_along_axis,
    const TArray<int64_t>& input_strides,
    const void* indices_data,
    const int64_t indices_size,
    const TArray<fast_divmod>& indices_strides,
    const int64_t axis,
    void* output_data,
    size_t element_size,
    size_t index_element_size) {

  if (indices_size > 0) {

    dim3 block(threads_per_block);
    dim3 blocksPerGrid((static_cast<int>(indices_size + block.x * thread_worksize - 1) / (block.x * thread_worksize)));

    switch (element_size) {
      case sizeof(int8_t): {
        using CudaType = typename ToCudaType<int8_t>::MappedType;
        _GatherElementsKernel<<<blocksPerGrid, block, 0, stream>>>(
            rank, reinterpret_cast<const CudaType*>(input_data), input_dim_along_axis, input_strides,
            indices_data, indices_size, index_element_size, indices_strides,
            axis, reinterpret_cast<CudaType*>(output_data));
      } break;

      case sizeof(int16_t): {
        using CudaType = typename ToCudaType<int16_t>::MappedType;
        _GatherElementsKernel<<<blocksPerGrid, block, 0, stream>>>(
            rank, reinterpret_cast<const CudaType*>(input_data), input_dim_along_axis, input_strides,
            indices_data, indices_size, index_element_size, indices_strides,
            axis, reinterpret_cast<CudaType*>(output_data));
      } break;

      case sizeof(int32_t): {
        using CudaType = typename ToCudaType<int32_t>::MappedType;
        _GatherElementsKernel<<<blocksPerGrid, block, 0, stream>>>(
            rank, reinterpret_cast<const CudaType*>(input_data), input_dim_along_axis, input_strides,
            indices_data, indices_size, index_element_size, indices_strides,
            axis, reinterpret_cast<CudaType*>(output_data));
      } break;

      case sizeof(int64_t): {
        using CudaType = typename ToCudaType<int64_t>::MappedType;
        _GatherElementsKernel<<<blocksPerGrid, block, 0, stream>>>(
            rank, reinterpret_cast<const CudaType*>(input_data), input_dim_along_axis, input_strides,
            indices_data, indices_size, index_element_size, indices_strides,
            axis, reinterpret_cast<CudaType*>(output_data));
      } break;

      // should not reach here as we validate if the all relevant types are supported in the Compute method
      default:
        ORT_THROW("Unsupported element size by the GatherElements CUDA kernel");
    }
  }
}  // namespace cuda

}  // namespace cuda
}  // namespace onnxruntime
