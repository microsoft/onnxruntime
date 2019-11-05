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
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // blockIdx.x is b
  const int offset = blockIdx.x * sequence_length;  // batch strides of sequence_length

  cub::Min min;
  int thread_data(sequence_length);

  const int idx = offset + threadIdx.x;
  if (threadIdx.x < sequence_length) {
    const T val = mask[idx];
    if (val == 0)  // masked position: report thread idx
    {
      thread_data = threadIdx.x;
    }
  }

  const auto min_index = BlockReduce(temp_storage).Reduce(thread_data, min);

  if (threadIdx.x == 0) {
    mask_index[blockIdx.x] = min_index;
  }
}

template <typename T, unsigned TPB>
__global__ void MaskIndexKernel(int sequence_length, const T* mask, int* mask_index) {
  using BlockReduce = cub::BlockReduce<int, TPB>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  // blockIdx.x is b
  const int offset = blockIdx.x * sequence_length;  // batch strides of sequence_length

  cub::Min min;
  int thread_data(sequence_length);

  for (int i = threadIdx.x; i < sequence_length; i += TPB) {
    const int idx = offset + i;
    const T val = mask[idx];
    if (val == 0)
    {
      thread_data = min(thread_data, i);
    }
  }

  const auto min_index = BlockReduce(temp_storage).Reduce(thread_data, min);

  if (threadIdx.x == 0) {
    mask_index[blockIdx.x] = min_index;
  }
}

template <typename T>
bool LaunchMaskIndexKernel(
    cudaStream_t stream,
    const T* mask,
    int* mask_index,
    int batch_size,
    int sequence_length)
{
  // Assume input mask total elements n = batch_size x sequence_length.
  // mask_index is of length batch_size, and the value is the index of first mask (0) within each batch.
  if (sequence_length <= 32) {
    MaskIndexKernelSmall<T, 32><<<batch_size, 32, 0, stream>>>(sequence_length, mask, mask_index);
  } else if (sequence_length <= 128) {
    MaskIndexKernelSmall<T, 128><<<batch_size, 128, 0, stream>>>(sequence_length, mask, mask_index);
  } else if (sequence_length == 384) {
    MaskIndexKernelSmall<T, 384><<<batch_size, 384, 0, stream>>>(sequence_length, mask, mask_index);
  } else {
    MaskIndexKernel<T, 256><<<batch_size, 256, 0, stream>>>(sequence_length, mask, mask_index);
  }

  return CUDA_CALL(cudaPeekAtLastError());
}

#define SPECIALIZED_IMPL(T) \
  template bool LaunchMaskIndexKernel<T>(cudaStream_t stream, const T * input_mask, int* mask_index, int batch_size, int sequence_length);

SPECIALIZED_IMPL(int32_t)
SPECIALIZED_IMPL(int64_t)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
