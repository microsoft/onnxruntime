// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "sequence_pooling_impl.h"
#include <cuda_fp16.h>
#include <stdio.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename InputIt, typename OutputIt>
__device__ inline void PrefixSumLinear(const InputIt* first, const InputIt* last, OutputIt* d_first) {
  if (first == last) {
    return;
  }
  OutputIt sum = *first;
  *d_first = sum;
  while (++first != last) {
    sum += *first;
    *(++d_first) = sum;
  }
}

template <typename T>
__global__ void SequencePoolingKernel(const T* input, const int64_t* sentence_lengthes, const int num_sequences, T* output) {

  const int hidden_size = gridDim.z;
  const int num_sequences_max = blockDim.y;
  const int batch_id = blockIdx.x;
  const int hidden_id = blockIdx.z;
  const int seq_id_per_batch = blockIdx.y;

  int sentence_lengthes_prefixsum[512]; // suppose num_sequences <= 256

  const int offset = batch_id * num_sequences;

  PrefixSumLinear(sentence_lengthes + offset, sentence_lengthes + offset + num_sequences, sentence_lengthes_prefixsum);

  const int sequence_length_for_split = sentence_lengthes_prefixsum[num_sequences - 1];
  const int past_sequence_length = (seq_id_per_batch == 0) ? 0 : sentence_lengthes_prefixsum[seq_id_per_batch - 1];

  const int input_offset = batch_id * hidden_size * sequence_length_for_split + hidden_size * past_sequence_length + hidden_id;
  const int output_offset = batch_id * hidden_size * num_sequences_max + hidden_size * seq_id_per_batch + hidden_id;

  if (seq_id_per_batch >= num_sequences) {
    output[output_offset] = 0;
  } else {
    T local_max = 0;
    output[output_offset] = local_max;
    const int sequence_length = sentence_lengthes_prefixsum[seq_id_per_batch] - past_sequence_length;
    //if (hidden_id == 767) {
    //  printf("input_offset is %d, output_offset is %d\n, input_val= %f\n", input_offset, output_offset, *(input + input_offset));
    //}
    for (int i = 0; i < sequence_length; ++i) {
      if (i == 0) {
        local_max = input[input_offset];
      } else {
        T value = input[input_offset + i * hidden_size];
        local_max = (float)value > (float)local_max ? value : local_max;
      }
    }
    //if ((float)local_max < -1.0f){
    //  printf("seq_id is %d, hidden_id is %d, local max is %f\n", seq_id_per_batch, hidden_id, local_max);
    //}
    output[output_offset] = local_max;
    //T result = *(output + output_offset);
    //if ((float)result < -1.0f){
    //  printf("seq_id is %d, hidden_id is %d, local max is %f\n", seq_id_per_batch, hidden_id, result);
    //}
    //if (seq_id_per_batch == 46) {
    //  printf("seq_id is %d, hidden_id is %d, local max is %f\n", seq_id_per_batch, hidden_id, result);
    //}
  }

}

template <typename T>
bool SequencePooling(
  cudaStream_t stream,
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const T* input,
  const int64_t* sentence_lengthes,
  T* output) {
  //T* masks) {
  const int num_sequences_max = 256;
  const dim3 grid(batch_size, num_sequences_max, hidden_size);
  const dim3 block(1, 1, 1);

  SequencePoolingKernel<T><<<grid, block, 0, stream>>>(input, sentence_lengthes, num_sequences, output);
  cudaDeviceSynchronize();
  return CUDA_CALL(cudaPeekAtLastError());
}

bool LaunchSequencePoolingKernel(
  void* output,
  //void* masks,
  const void* input,
  const void* sentence_lengthes,
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const size_t element_size) {
  // use default stream
  const cudaStream_t stream = nullptr;

  if (element_size == 2) {
    return SequencePooling<half>(
      stream,
      batch_size,
      hidden_size,
      num_sequences,
      reinterpret_cast<const half*>(input),
      reinterpret_cast<const int64_t*>(sentence_lengthes),
      reinterpret_cast<half*>(output)
      //reinterpret_cast<half*>(masks)
    );
  } else {
    return SequencePooling<float>(
      stream,
      batch_size,
      hidden_size,
      num_sequences,
      reinterpret_cast<const float*>(input),
      reinterpret_cast<const int64_t*>(sentence_lengthes),
      reinterpret_cast<float*>(output)
      //reinterpret_cast<float*>(masks)
    );
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
