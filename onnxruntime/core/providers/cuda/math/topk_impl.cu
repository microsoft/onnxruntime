// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "cub/cub.cuh"
#include <limits>

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void FillInput(const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t offset, int64_t dimension) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, dimension);
  auto left = offset / (axis == size - 1 ? 1 : elem_nums[axis + 1]) * elem_nums[axis];
  auto right = axis == size - 1 ? 0 : offset % elem_nums[axis + 1];
  auto input_offset = left + id * (axis == size - 1 ? 1 : elem_nums[axis + 1]) + right;
  output_v[id] = input_x[input_offset];
  output_i[id] = id;
}

template <typename T>
__global__ void FillOutput(const T* input_v, const int64_t* input_i, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t offset, int64_t dimension) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, dimension);
  auto left = offset / (axis == size - 1 ? 1 : elem_nums[axis + 1]) * elem_nums[axis] * K / dimension;
  auto right = axis == size - 1 ? 0 : offset % elem_nums[axis + 1];
  auto output_offset = left + id * (axis == size - 1 ? 1 : elem_nums[axis + 1]) + right;
  output_v[output_offset] = input_v[id];
  output_i[output_offset] = input_i[id];
}

__global__ void ExcludeOutput(int64_t* output_i, int64_t K, int64_t dimension) {
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, dimension);
  if (id >= K) {
    output_i[id] = dimension;
  }
}

template <typename T>
Status TopKImpl(const CudaKernel* kernel, const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t N, int64_t dimension) {
  CudaKernel::CudaAsyncBuffer<T> input_key_buffer(kernel, dimension);
  CudaKernel::CudaAsyncBuffer<T> output_key_buffer(kernel, dimension);
  CudaKernel::CudaAsyncBuffer<int64_t> input_value_buffer(kernel, dimension);
  CudaKernel::CudaAsyncBuffer<int64_t> output_value_buffer(kernel, dimension);
  ORT_RETURN_IF_ERROR(input_key_buffer.CopyToGpu());
  ORT_RETURN_IF_ERROR(output_key_buffer.CopyToGpu());
  ORT_RETURN_IF_ERROR(input_value_buffer.CopyToGpu());
  ORT_RETURN_IF_ERROR(output_value_buffer.CopyToGpu());
  auto input_key = input_key_buffer.GpuPtr();
  auto output_key = output_key_buffer.GpuPtr();
  auto input_value = input_value_buffer.GpuPtr();
  auto output_value = output_value_buffer.GpuPtr();
  size_t temp_bytes = 0;
  CUDA_RETURN_IF_ERROR(cub::DeviceRadixSort::SortPairs(nullptr, temp_bytes, input_key, output_key, input_value, output_value, dimension));
  CudaKernel::CudaAsyncBuffer<char> temp_storage_buffer(kernel, temp_bytes);
  ORT_RETURN_IF_ERROR(temp_storage_buffer.CopyToGpu());
  auto temp_storage = temp_storage_buffer.GpuPtr();
  auto blocksPerGridD = (int)(ceil(static_cast<float>(dimension) / GridDim::maxThreadsPerBlock));
  auto blocksPerGridK = (int)(ceil(static_cast<float>(K) / GridDim::maxThreadsPerBlock));
  for (int64_t i = 0; i < N; i++) {
    FillInput<T><<<blocksPerGridD, GridDim::maxThreadsPerBlock, 0>>>(input_x, input_key, input_value, elem_nums, size, axis, K, i, dimension);
    CUDA_RETURN_IF_ERROR(1 == largest ? cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_bytes, input_key, output_key, input_value, output_value, dimension) : cub::DeviceRadixSort::SortPairs(temp_storage, temp_bytes, input_key, output_key, input_value, output_value, dimension));
    if (1 == sorted) {
      FillOutput<T><<<blocksPerGridK, GridDim::maxThreadsPerBlock, 0>>>(output_key, output_value, output_v, output_i, elem_nums, size, axis, K, i, dimension);
    } else {  //reorder by ascending index
      ExcludeOutput<<<blocksPerGridD, GridDim::maxThreadsPerBlock, 0>>>(output_value, K, dimension);
      CUDA_RETURN_IF_ERROR(cub::DeviceRadixSort::SortPairs(temp_storage, temp_bytes, output_value, input_value, output_key, input_key, dimension));
      FillOutput<T><<<blocksPerGridK, GridDim::maxThreadsPerBlock, 0>>>(input_key, input_value, output_v, output_i, elem_nums, size, axis, K, i, dimension);
    }
  }
  return Status::OK();
}

#define TOPKIMPLE(T) template Status TopKImpl<T>(const CudaKernel* kernel, \
                                                 const T* input_x,         \
                                                 T* output_v,              \
                                                 int64_t* output_i,        \
                                                 const int64_t* elem_nums, \
                                                 size_t size,              \
                                                 int64_t axis,             \
                                                 int64_t K,                \
                                                 int64_t largest,          \
                                                 int64_t sorted,           \
                                                 int64_t N,                \
                                                 int64_t dimension)

TOPKIMPLE(uint8_t);
TOPKIMPLE(uint16_t);
TOPKIMPLE(uint32_t);
TOPKIMPLE(uint64_t);
TOPKIMPLE(int8_t);
TOPKIMPLE(int16_t);
TOPKIMPLE(int32_t);
TOPKIMPLE(int64_t);
TOPKIMPLE(float);
TOPKIMPLE(double);

}  // namespace cuda
}  // namespace onnxruntime