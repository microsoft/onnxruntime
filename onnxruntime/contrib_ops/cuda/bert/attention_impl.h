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

bool LaunchDecoderAttentionKernel(
    const cudaDeviceProp& prop,                   // Device Properties
    cudaStream_t stream,                          // Cuda stream
    cublasHandle_t& cublas,                       // Cublas handle
    const size_t element_size,                    // Element size of input tensor
    const int batch_size,                         // Batch size (B)
    const int sequence_length,                    // Sequence length (S)
    const int kv_sequence_length,                 // Key/Value/Cache sequence length
    const int num_heads,                          // Number of attention heads (N)
    const int head_size,                          // Hidden layer size per head (H)
    const bool static_kv,                         // Whether cross attention or not
    const bool use_past,                          // Whether use cache or not
    const bool has_layer_state,                   // Whether output cache or not
    const bool has_key_padding_mask,              // Whether use key_padding_mask or not
    const void* gemm_query_buffer,                // Query buffer
    const void* gemm_kv_buffer,                   // Key and value buffer
    const bool* key_padding_mask,                 // Key padding mask
    const void* key_cache,                        // Input key cache
    const void* value_cache,                      // Input value cache
    void* qkv_buffer,                             // Temporary buffer
    void* workspace_buffer,                       // Temporary buffer
    void* output,                                 // Output tensor
    void* new_key_cache,                          // New_key_cache tensor
    void* new_value_cache                         // New_value_cache tensor
);

bool LaunchTransCtx(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const int max_threads_per_block, const bool reversed_bs, const float* input, float* output);

bool LaunchTransCtx(cudaStream_t stream,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const int max_threads_per_block, const bool reversed_bs, const half* input, half* output);

bool LaunchTransQkv(cudaStream_t stream, const int matrix_num,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const int max_threads_per_block, const bool reversed_bs, const float* input, float* output);

bool LaunchTransQkv(cudaStream_t stream, const int matrix_num,
                    const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                    const int max_threads_per_block, const bool reversed_bs, const half* input, half* output);

bool LaunchConcatTensorToTensor(cudaStream_t stream,
                                const int all_sequence_length,
                                const int sequence_length,
                                const int batch_size,
                                const int head_size,
                                const int num_heads,
                                const int max_threads_per_block,
                                const int matrix_num,
                                const float* tensor_in,
                                const float* tensor_add,
                                float* tensor_out);

bool LaunchConcatTensorToTensor(cudaStream_t stream,
                                const int all_sequence_length,
                                const int sequence_length,
                                const int batch_size,
                                const int head_size,
                                const int num_heads,
                                const int max_threads_per_block,
                                const int matrix_num,
                                const half* tensor_in,
                                const half* tensor_add,
                                half* tensor_out);

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
