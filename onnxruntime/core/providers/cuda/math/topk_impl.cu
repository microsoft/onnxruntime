// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "topk_impl.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
__global__ void TopKKernel(const T* input_x, void* output_v, void* output_i, const int64_t* input_shape, int64_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted)
{
  auto N = size / input_shape[axis];
  CALCULATE_ELEMENTWISE_INDEX_OR_EXIT(id, N);
}

template <typename T>
Status TopKImpl(const T* input_x, void* output_v, void* output_i, const int64_t* input_shape, int64_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted)
{
  auto N = size / input_shape[axis];
  int blocksPerGrid = (int)(ceil(static_cast<float>(N) / GridDim::maxThreadsPerBlock));
  TopKKernel <<<blocksPerGrid, GridDim::maxThreadsPerBlock,0>>> (input_x, output_v, output_i, input_shape, size, axis, K, largest, sorted);
  return Status::OK();
}

template Status TopKImpl<int32_t> (const int32_t* input_x, void* output_v, void* output_i, const int64_t* input_shape, int64_t size, int64_t axis, int64_t K, int64_t largest, int64_t sorted);

}
}