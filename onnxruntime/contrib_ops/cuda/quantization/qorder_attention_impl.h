// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "contrib_ops/cuda/bert/attention_impl.h"
#include <cublas_v2.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

bool LaunchQOrderAttentionKernel(
    const cudaDeviceProp& prop,                   // Device Properties
    cudaStream_t stream,                          // cuda stream
    const void* input,                            // Input tensor
    const int* mask_index,                        // Attention mask raw data or index (end position of each sequence, or end positions and start positions). NULL means no mask.
    gsl::span<const int64_t> mask_index_dims,     // Mask index shape
    void* output,                                 // Output tensor
    int batch_size,                               // Batch size (B)
    int sequence_length,                          // Sequence length (S)
    int num_heads,                                // Number of attention heads (N)
    int head_size,                                // Hidden layer size per head (H)
    void* workspace,                              // Temporary buffer
    cublasHandle_t& cublas,                       // Cublas handle
    const size_t element_size,                    // Element size of input tensor
    bool is_unidirectional,                       // Whether there is unidirecitonal mask.
    int past_sequence_length,                     // Sequence length in past state
    const void* past,                             // Past state input
    const void* extra_add_qk,                     // Additional Add
    void* present                                 // Present state output
);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
