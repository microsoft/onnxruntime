// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sequence_pooling_impl.h"
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void SequencePoolingKernel(const T* input, T* output, const int* sentence_lengthes_prefixsum) {
  const int sequence_length_for_split = sentence_lengthes_prefixsum[blockDim.x - 1];
  const int input_offset =
  const int output_offset =
  const int stride = gridDim.x;
  // 1. lookup word and segment of the block
  // blockIdx.x = position in the sequence
  // blockIdx.y = batch
  // gridDim.x = sequence_length
  // gridDim.y = batch_size
  //int n = threadIdx.y;
  //int s = blockIdx.x;
  //int b = blockIdx.y;
  //int num_heads = blockDim.y;
  //int sequence_length = gridDim.x;

}

template <typename T>
bool SequencePooling(
  cudaStream_t stream,
  const T* input,
  T* output,
  const int batch_size,
  const int hidden_size,
  const int num_sequences) {

  const dim3 grid(hidden_size, batch_size, 1);
  const dim3 block(num_sequences, 1, 1);

  // bugbug
  SequencePoolingKernel<T><<<grid, block, 0, stream>>>(input, output, sentence_lengthes_prefixsum);

  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchSequencePoolingKernel(
  void* output,
  const void* input,
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const size_t element_size) {
  // use default stream
  const cudaStream_t stream = nullptr;

  if (element_size == 2) {
    return SequencePooling<half>(
      stream,
      reinterpret_cast<const half*>(input),
      reinterpret_cast<const half*>(output),
      batch_size,
      hidden_size,
      num_sequences);
  } else {
    return SequencePooling<float>(
      stream,
      reinterpret_cast<const float*>(input),
      reinterpret_cast<const float*>(output),
      batch_size,
      hidden_size,
      num_sequences);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
