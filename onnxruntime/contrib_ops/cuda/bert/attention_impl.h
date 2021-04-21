// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cublas_v2.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {
size_t GetAttentionScratchSize(size_t element_size, int batch_size, int num_heads, int sequence_length, int all_sequence_length);

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    int batchsize,
    int num_heads,
    int head_size,
    int sequence_length,
    int past_sequence_length);

bool LaunchAttentionKernel(
    const cudaDeviceProp& prop,                   // Device Properties
    cudaStream_t stream,                          // cuda stream
    const void* input,                            // Input tensor
    const int* mask_index,                        // Attention mask raw data or index (end position of each sequence, or end positions and start positions). NULL means no mask.
    const std::vector<int64_t>* mask_index_dims,  // Mask index shape
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
    void* present                                 // Present state output
);

bool LaunchTransCtx(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const int max_threads_per_block, const float* input, float* output);

bool LaunchTransCtx(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const int max_threads_per_block, const half* input, half* output);

bool LaunchTransQkv(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const int max_threads_per_block, const float* input, float* output);

bool LaunchTransQkv(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const int max_threads_per_block, const half* input, half* output);

bool LaunchConcatPastToPresent(cudaStream_t stream,
                               const int all_sequence_length,
                               const int sequence_length,
                               const int batch_size,
                               const int head_size,
                               const int num_heads,
                               const int max_threads_per_block,
                               const float* past,
                               const float* k_v,
                               float* present);

bool LaunchConcatPastToPresent(cudaStream_t stream,
                               const int all_sequence_length,
                               const int sequence_length,
                               const int batch_size,
                               const int head_size,
                               const int num_heads,
                               const int max_threads_per_block,
                               const half* past,
                               const half* k_v,
                               half* present);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
