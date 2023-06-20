// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {
namespace attention_softmax_cuda {

template <typename T>
Status ComputeSoftmax(cudaStream_t stream, const int all_sequence_length, const int sequence_length,
                      const int batch_size, const int num_heads, const T* rel_pos_bias,
                      const bool broadcast_rel_pos_bias, T* input, T* output, bool causal);

template <typename T>
Status ComputeSoftmaxWithCumSeqLength(
    const T* input,
    const T* rel_pos_bias,
    const bool broadcast_rel_pos_bias,
    const int32_t* cum_seq_length,
    const int batch_size,
    const int sequence_length,
    const int num_heads,
    T* output, cudaStream_t stream);

template <typename T>
Status ComputeSoftmaxWithMask1D(cudaStream_t stream,
                                const int all_sequence_length,
                                const int sequence_length,
                                const int batch_size,
                                const int num_heads,
                                const int* mask_index,
                                const int* mask_start,
                                const T* rel_pos_bias,
                                const bool broadcast_rel_pos_bias,
                                const T* input,
                                T* output,
                                const bool causal);

template <typename T>
Status ComputeSoftmaxWithRawMask(cudaStream_t stream,
                                 const int all_sequence_length,
                                 const int sequence_length,
                                 const int batch_size,
                                 const int num_heads,
                                 const int* attention_mask,
                                 const bool* key_padding_mask,
                                 const T* rel_pos_bias,
                                 const bool broadcast_rel_pos_bias,
                                 const T* input,
                                 T* output,
                                 const bool causal,
                                 const float rsqrt_head_size,
                                 const int mask_dimension,
                                 const int max_sequence_length,
                                 const bool use_persistent_softmax,
                                 T* persistent_softmax_workspace,
                                 const float mask_filter_value);

}  // namespace attention_softmax_cuda
}  // namespace contrib
}  // namespace onnxruntime
