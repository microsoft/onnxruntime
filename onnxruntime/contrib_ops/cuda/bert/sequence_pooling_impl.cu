// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sequence_pooling_impl.h"
#include <cuda_fp16.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void SequencePoolingKernel(const T* input, T* output, const int* sentence_lengthes_prefixsum) {
  // blockDim.x -> num_sequences
  // gridDim.x -> hidden_size
  // gridDim.y -> batch_size
  const int batch_size = gridDim.y;
  const int hidden_size = gridDim.x;
  const int num_sequences = blockDim.x;
  const int sequence_length_for_split = sentence_lengthes_prefixsum[blockDim.x - 1];
  const int past_sequence_length = (threadIdx.x == 0) ? 0 : sentence_lengthes_prefixsum[threadIdx.x - 1];

  const int input_offset = batch_size * hidden_size * sequence_length_for_split + hidden_size * past_sequence_length + blockIdx.x;
  const int output_offset = batch_size * hidden_size * num_sequences + hidden_size * threadIdx.x + blockIdx.x;
  const int stride = hidden_size;

  T local_max;
  const int sequence_length = sentence_lengthes_prefixsum[threadIdx.x] - past_sequence_length;
  for (int i = 0; i < sequence_length; ++i) {
    if (i == 0) {
      local_max = *(input + input_offset);
    } else {
      local_max = max(local_max, *(input + input_offset + i * stride));
    }
  }

  *(output + output_offset) = local_max;
}

template <typename T>
bool SequencePooling(
  cudaStream_t stream,
  const T* input,
  const int* sentence_lengthes_prefixsum,
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  T* output,) {

  const dim3 grid(hidden_size, batch_size, 1);
  const dim3 block(num_sequences, 1, 1);

  // bugbug
  SequencePoolingKernel<T><<<grid, block, 0, stream>>>(input, output, sentence_lengthes_prefixsum);

  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchSequencePoolingKernel(
  void* output,
  const void* input,
  const int* sentence_lengthes_prefixsum,
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
      sentence_lengthes_prefixsum,
      batch_size,
      hidden_size,
      num_sequences,
      reinterpret_cast<const half*>(output)
    );
  } else {
    return SequencePooling<float>(
      stream,
      reinterpret_cast<const float*>(input),
      sentence_lengthes_prefixsum,
      batch_size,
      hidden_size,
      num_sequences,
      reinterpret_cast<const float*>(output)
    );
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
