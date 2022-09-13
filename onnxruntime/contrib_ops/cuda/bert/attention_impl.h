// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

size_t GetAttentionScratchSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t sequence_length,
    size_t all_sequence_length);

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    size_t batchsize,
    size_t num_heads,
    size_t head_size,
    size_t sequence_length,
    size_t past_sequence_length,
    void* fused_runner);

Status LaunchAttentionKernel(
    const cudaDeviceProp& prop,                // Device Properties
    cudaStream_t stream,                       // cuda stream
    cublasHandle_t& cublas,                    // Cublas handle
    const size_t element_size,                 // Element size of input tensor
    int batch_size,                            // Batch size (B)
    int sequence_length,                       // Sequence length (S)
    int num_heads,                             // Number of attention heads (N)
    int head_size,                             // Hidden layer size per head (H)
    int past_sequence_length,                  // Sequence length in past state
    bool is_unidirectional,                    // Whether there is unidirecitonal mask.
    const void* input,                         // Input tensor
    const void* bias,                          // Bias tensor
    const int* mask_index,                     // Attention mask raw data or index. NULL means no mask.
    gsl::span<const int64_t> mask_index_dims,  // Mask index shape
    const void* past,                          // Past state input
    const void* extra_add_qk,                  // Additional Add
    void* workspace,                           // Temporary buffer
    void* output,                              // Output tensor
    void* present,                             // Present state output
    void* fused_runner                         // Fused multi-head attention
);

Status LaunchDecoderAttentionKernel(
    const cudaDeviceProp& prop,       // Device Properties
    cudaStream_t stream,              // Cuda stream
    cublasHandle_t& cublas,           // Cublas handle
    const size_t element_size,        // Element size of input tensor
    const int batch_size,             // Batch size (B)
    const int sequence_length,        // Sequence length (S)
    const int kv_sequence_length,     // Key/Value/Cache sequence length
    const int num_heads,              // Number of attention heads (N)
    const int head_size,              // Hidden layer size per head (H)
    const bool static_kv,             // Whether cross attention or not
    const bool use_past,              // Whether use cache or not
    const bool has_layer_state,       // Whether output cache or not
    const bool has_key_padding_mask,  // Whether use key_padding_mask or not
    const void* gemm_query_buffer,    // Query buffer
    const void* gemm_kv_buffer,       // Key and value buffer
    const bool* key_padding_mask,     // Key padding mask
    const void* key_cache,            // Input key cache
    const void* value_cache,          // Input value cache
    void* qkv_buffer,                 // Temporary buffer
    void* workspace_buffer,           // Temporary buffer
    void* output,                     // Output tensor
    void* new_key_cache,              // New_key_cache tensor
    void* new_value_cache             // New_value_cache tensor
);

// BxNxSxH => BxSxNxH or SxBxNxH (reversed_bs is true)
Status LaunchTransCtx(cudaStream_t stream,
                      const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                      const int max_threads_per_block, const bool reversed_bs, const float* input, float* output);

Status LaunchTransCtx(cudaStream_t stream,
                      const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                      const int max_threads_per_block, const bool reversed_bs, const half* input, half* output);

// BxSxMxNxH or SxBxMxNxH (reversed_bs is true) => MxBxNxSxH
Status LaunchTransQkv(cudaStream_t stream, const int matrix_num,
                      const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                      const int max_threads_per_block, const bool reversed_bs, const float* input, float* output);

Status LaunchTransQkv(cudaStream_t stream, const int matrix_num,
                      const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                      const int max_threads_per_block, const bool reversed_bs, const half* input, half* output);

Status LaunchConcatTensorToTensor(cudaStream_t stream,
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

Status LaunchConcatTensorToTensor(cudaStream_t stream,
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

Status LaunchConcatPastToPresent(cudaStream_t stream,
                                 const int all_sequence_length,
                                 const int sequence_length,
                                 const int batch_size,
                                 const int head_size,
                                 const int num_heads,
                                 const int max_threads_per_block,
                                 const float* past,
                                 const float* k_v,
                                 float* present);

Status LaunchConcatPastToPresent(cudaStream_t stream,
                                 const int all_sequence_length,
                                 const int sequence_length,
                                 const int batch_size,
                                 const int head_size,
                                 const int num_heads,
                                 const int max_threads_per_block,
                                 const half* past,
                                 const half* k_v,
                                 half* present);

void LaunchTrtSequenceOffset(int* trt_mha_padding_offset,
                             const int* mask_index,
                             const int batch_size,
                             cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
