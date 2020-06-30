/*
 The implementation of this file is based on embLayerNorm plugin in TensorRT demo:
 https://github.com/NVIDIA/TensorRT/tree/release/5.1/demo/BERT/
 
Copyright 2019 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "mask_index_impl.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include <cub/cub.cuh>

using namespace cub;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T, unsigned TPB>
__global__ void MaskIndexKernelSmall(int sequence_length, const T* mask, int* mask_index) {
  using BlockReduce = cub::BlockReduce<int, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage_min;
  __shared__ typename BlockReduce::TempStorage temp_storage_max;

  const int& batch_size = gridDim.x;
  const int& batch_index = blockIdx.x;

  // Mask is optional for EmbedLayerNormalization operator
  if (mask == nullptr) {
    if (threadIdx.x == 0) {
      mask_index[batch_index] = sequence_length;
      mask_index[batch_size + batch_index] = 0;
    }
    return;
  }

  // Find min and max positions of mask value > 0
  int min_position(sequence_length);
  int max_position(-1);

  if (threadIdx.x < sequence_length) {
    const T val = mask[batch_index * sequence_length + threadIdx.x];
    if (val > static_cast<T>(0))  // mask could be 0 or 1
    {
      min_position = threadIdx.x;
      max_position = threadIdx.x;
    }
  }

  const int min_valid = BlockReduce(temp_storage_min).Reduce(min_position, cub::Min());
  const int max_valid = BlockReduce(temp_storage_max).Reduce(max_position, cub::Max());

  if (threadIdx.x == 0) {
    mask_index[batch_index] = max_valid + 1;
    mask_index[batch_size + batch_index] = min_valid;
  }
}

template <typename T, unsigned TPB>
__global__ void MaskIndexKernel(int sequence_length, const T* mask, int* mask_index) {
  using BlockReduce = cub::BlockReduce<int, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage_min;
  __shared__ typename BlockReduce::TempStorage temp_storage_max;

  const int& batch_size = gridDim.x;
  const int& batch_index = blockIdx.x;

  // Mask is optional for EmbedLayerNormalization operator
  if (mask == nullptr) {
    if (threadIdx.x == 0) {
      mask_index[batch_index] = sequence_length;
      mask_index[batch_size + batch_index] = 0;
    }
    return;
  }

  // Find min and max positions of mask value > 0
  int min_position(sequence_length);
  int max_position(-1);

  const int offset = blockIdx.x * sequence_length;

  for (int i = threadIdx.x; i < sequence_length; i += TPB) {
    const T val = mask[offset + i];
    if (val > static_cast<T>(0)) {  // mask could be 0 or 1
      min_position = min(min_position, i);
      max_position = max(max_position, i);
    }
  }

  const int min_valid = BlockReduce(temp_storage_min).Reduce(min_position, cub::Min());
  const int max_valid = BlockReduce(temp_storage_max).Reduce(max_position, cub::Max());

  if (threadIdx.x == 0) {
    mask_index[batch_index] = max_valid + 1;
    mask_index[batch_size + batch_index] = min_valid;
  }
}

template <typename T>
bool LaunchMaskIndexKernel(
    cudaStream_t stream,
    const T* mask,
    int* mask_index,
    int batch_size,
    int sequence_length) {
  // Assume input mask total elements n = batch_size x sequence_length.
  if (sequence_length <= 32) {
    MaskIndexKernelSmall<T, 32><<<batch_size, 32, 0, stream>>>(sequence_length, mask, mask_index);
  } else if (sequence_length <= 64) {
    MaskIndexKernelSmall<T, 64><<<batch_size, 64, 0, stream>>>(sequence_length, mask, mask_index);
  } else if (sequence_length <= 128) {
    MaskIndexKernelSmall<T, 128><<<batch_size, 128, 0, stream>>>(sequence_length, mask, mask_index);
  } else if (sequence_length <= 256) {
    MaskIndexKernelSmall<T, 256><<<batch_size, 256, 0, stream>>>(sequence_length, mask, mask_index);
  } else if (sequence_length <= 512) {
    MaskIndexKernelSmall<T, 512><<<batch_size, 512, 0, stream>>>(sequence_length, mask, mask_index);
  } else {
    MaskIndexKernel<T, 512><<<batch_size, 512, 0, stream>>>(sequence_length, mask, mask_index);
  }

  return CUDA_CALL(cudaPeekAtLastError());
}

#define SPECIALIZED_IMPL(T) \
  template bool LaunchMaskIndexKernel<T>(cudaStream_t stream, const T* input_mask, int* mask_index, int batch_size, int sequence_length);

SPECIALIZED_IMPL(int32_t)  // Needed by EmbedLayerNormalization.
SPECIALIZED_IMPL(int64_t)
SPECIALIZED_IMPL(float)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
