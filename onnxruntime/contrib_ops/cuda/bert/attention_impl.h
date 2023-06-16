// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/framework/allocator.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kCumulatedSequenceLengthCacheMaxBatchSize = 128;

struct CumulatedSequenceLengthCache {
  onnxruntime::IAllocatorUniquePtr<void> buffer;
  int32_t max_batch_size;
  int32_t sequence_length;

  CumulatedSequenceLengthCache() : max_batch_size(0), sequence_length(0) {}
  void Initialize(int32_t sequence_length, cudaStream_t stream);
};

size_t
GetAttentionScratchSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t sequence_length,
    size_t all_sequence_length);

size_t GetSequenceOffsetSize(int batch_size, bool has_padding);

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    size_t batchsize,
    size_t num_heads,
    size_t qk_head_size,
    size_t v_head_size,
    size_t sequence_length,
    size_t kv_sequence_length,
    size_t total_sequence_length,
    void* fused_runner,
    bool use_fused_cross_attention,
    bool use_memory_efficient_attention);

template <typename T>
struct AttentionData {
  T* gemm_buffer;
  const T* bias;

  const T* query;
  const T* key;
  const T* value;
  const int* mask_index;
  gsl::span<const int64_t> mask_index_dims;
  const T* past;
  const T* past_key;
  const T* past_value;
  const T* relative_position_bias;

  bool has_qkv_workspace;
  T* workspace;
  T* temp_k_workspace;
  T* temp_v_workspace;

  T* output;
  T* present;
  T* present_key;
  T* present_value;

  void* fused_runner;
  const void* fused_cross_attention_kernel;

  bool use_memory_efficient_attention;

  mutable CumulatedSequenceLengthCache* cumulated_sequence_length_q_cache;
  mutable CumulatedSequenceLengthCache* cumulated_sequence_length_kv_cache;
};

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data);

Status LaunchDecoderAttentionKernel(
    const cudaDeviceProp& prop,       // Device Properties
    cudaStream_t stream,              // Cuda stream
    cublasHandle_t& cublas,           // Cublas handle
    const size_t element_size,        // Element size of input tensor
    const int batch_size,             // Batch size (B)
    const int sequence_length,        // Sequence length (S)
    const int kv_sequence_length,     // Key/Value/Cache sequence length
    const int num_heads,              // Number of attention heads (N)
    const int head_size,              // Hidden size per head (H)
    const bool static_kv,             // Whether cross attention or not
    const bool use_past,              // Whether use cache or not
    const bool has_layer_state,       // Whether output cache or not
    const bool has_key_padding_mask,  // Whether use key_padding_mask or not
    const float mask_filter_value,    // Mask filter value
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
                      const int max_threads_per_block, const bool reversed_bs, const float* input, float* output,
                      int total_matrix_count = -1);

Status LaunchTransQkv(cudaStream_t stream, const int matrix_num,
                      const int sequence_length, const int batch_size, const int head_size, const int num_heads,
                      const int max_threads_per_block, const bool reversed_bs, const half* input, half* output,
                      int total_matrix_count = -1);

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

template <typename T>
Status LaunchStridedCopy(cudaStream_t stream,
                         const T* in, int4 in_shape, longlong4 in_strides,  // coord (b,n,s,h)
                         T* out, longlong4 out_strides,                     // coord (b,n,s,h)
                         int max_threads_per_block);
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
