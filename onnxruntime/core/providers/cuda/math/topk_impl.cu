// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"
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
Status TopKImpl(const T* input_x, T* output_v, int64_t* output_i, const int64_t* elem_nums, size_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t N, int64_t dimension) {
  auto err = cudaSuccess;
  T* input_key;
  T* output_key;
  int64_t* input_value;
  int64_t* output_value;
  err = cudaMalloc(&input_key, sizeof(T) * dimension);
  err = cudaSuccess == err ? cudaMalloc(&output_key, sizeof(T) * dimension) : err;
  err = cudaSuccess == err ? cudaMalloc(&input_value, sizeof(int64_t) * dimension) : err;
  err = cudaSuccess == err ? cudaMalloc(&output_value, sizeof(int64_t) * dimension) : err;
  if (cudaSuccess != err) {
    return Status(StatusCategory::ONNXRUNTIME, StatusCode::EP_FAIL, "Faild to alloc cuda memory");
  }
  void* temp_storage = nullptr;
  size_t temp_bytes = 0;
  err = cub::DeviceRadixSort::SortPairs(temp_storage, temp_bytes, input_key, output_key, input_value, output_value, dimension);
  if (cudaSuccess != err) {
    return Status(StatusCategory::ONNXRUNTIME, StatusCode::EP_FAIL, "cub::DeviceRadixSort::SortPairs call failed");
  }
  err = cudaMalloc(&temp_storage, temp_bytes);
  if (cudaSuccess != err) {
    return Status(StatusCategory::ONNXRUNTIME, StatusCode::EP_FAIL, "Faild to alloc cuda memory");
  }
  auto blocksPerGridD = (int)(ceil(static_cast<float>(dimension) / GridDim::maxThreadsPerBlock));
  auto blocksPerGridK = (int)(ceil(static_cast<float>(K) / GridDim::maxThreadsPerBlock));
  for (int64_t i = 0; i < N; i++) {
    FillInput<T><<<blocksPerGridD, GridDim::maxThreadsPerBlock, 0>>>(input_x, input_key, input_value, elem_nums, size, axis, K, i, dimension);
    err = 1 == largest ? cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_bytes, input_key, output_key, input_value, output_value, dimension) : cub::DeviceRadixSort::SortPairs(temp_storage, temp_bytes, input_key, output_key, input_value, output_value, dimension);
    if (cudaSuccess != err) {
      return Status(StatusCategory::ONNXRUNTIME, StatusCode::EP_FAIL, "cub::DeviceRadixSort::SortPairs call failed");
    }
    if (1 == sorted) {
      FillOutput<T><<<blocksPerGridK, GridDim::maxThreadsPerBlock, 0>>>(output_key, output_value, output_v, output_i, elem_nums, size, axis, K, i, dimension);
    } else {  //reorder by ascending index
      ExcludeOutput<<<blocksPerGridD, GridDim::maxThreadsPerBlock, 0>>>(output_value, K, dimension);
      err = cub::DeviceRadixSort::SortPairs(temp_storage, temp_bytes, output_value, input_value, output_key, input_key, dimension);
      if (cudaSuccess != err) {
        return Status(StatusCategory::ONNXRUNTIME, StatusCode::EP_FAIL, "cub::DeviceRadixSort::SortPairs call failed");
      }
      FillOutput<T><<<blocksPerGridK, GridDim::maxThreadsPerBlock, 0>>>(input_key, input_value, output_v, output_i, elem_nums, size, axis, K, i, dimension);
    }
  }
  cudaFree(temp_storage);
  cudaFree(input_key);
  cudaFree(output_key);
  cudaFree(input_value);
  cudaFree(output_value);
  return Status::OK();
}

#define TOPKIMPLE(T) template Status TopKImpl<T>(const T* input_x,         \
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