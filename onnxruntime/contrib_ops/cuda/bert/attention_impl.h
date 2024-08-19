// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <gsl/gsl>
#include <iostream>
#include <mutex>
#include "core/framework/allocator.h"
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kCumulatedSequenceLengthCacheMaxBatchSize = 128;

// A cache for cumulated sequence length. It will be initialized in the first request, then become read-only after that.
struct CumulatedSequenceLengthCache {
  onnxruntime::IAllocatorUniquePtr<void> buffer;
  int32_t max_batch_size;
  int32_t sequence_length;

  CumulatedSequenceLengthCache() : max_batch_size(kCumulatedSequenceLengthCacheMaxBatchSize), sequence_length(0) {}

  const int32_t* TryGet(int batch_size, int32_t sequence_length, cudaStream_t stream);

  // Use this flag to guard the initializaton only once in multi-threading.
  mutable std::once_flag init_once_flag_;
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
    bool use_flash_attention,
    bool use_fused_cross_attention,
    bool use_memory_efficient_attention,
    bool no_qkv_workspace);

template <typename T>
struct AttentionData {
  T* gemm_buffer = nullptr;
  const T* bias = nullptr;

  const T* query = nullptr;
  const T* key = nullptr;
  const T* value = nullptr;
  const int* mask_index = nullptr;
  gsl::span<const int64_t> mask_index_dims;
  const T* past = nullptr;
  const T* past_key = nullptr;
  const T* past_value = nullptr;
  const T* attention_bias = nullptr;

  bool has_qkv_workspace = false;
  T* workspace = nullptr;

  T* output = nullptr;
  T* present = nullptr;
  T* present_key = nullptr;
  T* present_value = nullptr;

  void* fused_runner = nullptr;
  const void* fused_cross_attention_kernel = nullptr;

  bool use_flash_attention = false;
  bool use_memory_efficient_attention = false;

  const int32_t* cumulated_sequence_length_q_cache = nullptr;
  const int32_t* cumulated_sequence_length_kv_cache = nullptr;

  // Intermediate data
  T* q = nullptr;
  T* k = nullptr;
  T* v = nullptr;
  T* scratch = nullptr;
  AttentionQkvFormat qkv_format = AttentionQkvFormat::UNKNOWN;

  // Flash buffers
  T* softmax_lse = nullptr;
  T* softmax_lse_accum = nullptr;
  T* out_accum = nullptr;

  // For Debugging
  size_t workspace_bytes = 0;
  bool allow_debug_info = false;

  bool IsUnfused() const {
    return !use_flash_attention && !use_memory_efficient_attention &&
           (fused_runner == nullptr) && (fused_cross_attention_kernel == nullptr);
  }

  void PrintDebugInfo() const {
    std::cout << "flash=" << use_flash_attention
              << ", efficient=" << use_memory_efficient_attention
              << ", fused_runner=" << (fused_runner != nullptr)
              << ", fused_cross=" << (fused_cross_attention_kernel != nullptr)
              << ", bias=" << (bias != nullptr)
              << ", attn_bias=" << (attention_bias != nullptr)
              << ", mask_dims=" << mask_index_dims.size()
              << ", has_qkv_workspace=" << has_qkv_workspace
              << ", workspace=" << workspace_bytes
              << ", past=" << (past != nullptr ? 1 : (past_key != nullptr ? 2 : 0))
              << ", present=" << (present != nullptr ? 1 : (present_key != nullptr ? 2 : 0))
              << std::endl;
  }
};

// Return true if it does not need qkv workspace, false otherwise.
template <typename T>
bool NoQkvWorkspace(contrib::AttentionParameters& parameters, AttentionData<T>& data);

template <typename T>
Status PrepareQkv(contrib::AttentionParameters& parameters,
                  AttentionData<T>& data,
                  cudaStream_t stream,
                  int max_threads_per_block);

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data);

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

Status Transpose_BSNH_to_BNSH(const int batch_size, const int sequence_length, const int num_heads, const int head_size,
                              const float* input, float* output, cudaStream_t stream, const int max_threads_per_block);

Status Transpose_BSNH_to_BNSH(const int batch_size, const int sequence_length, const int num_heads, const int head_size,
                              const half* input, half* output, cudaStream_t stream, const int max_threads_per_block);

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

template <typename T>
Status ConcatPastToPresent(int batch_size, int num_heads, int qk_head_size, int v_head_size,
                           int sequence_length, int total_sequence_length,
                           cudaStream_t stream,
                           int max_threads_per_block,
                           AttentionData<T>& data);

template <typename T>
Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                           const int max_sequence_length,
                                           const int past_sequence_length,
                                           const int sequence_length,
                                           const int batch_size,
                                           const int head_size,
                                           const int num_heads,
                                           const int max_threads_per_block,
                                           const T* biases,
                                           const T* qkv_buffer,
                                           T* present);

template <typename T>
Status LaunchStridedCopy(
    cudaStream_t stream,
    const T* in, int4 in_shape, longlong4 in_strides, const int* in_seqlens_offset,  // coord (b,n,s,h)
    T* out, longlong4 out_strides, const int* out_seqlens_offset,                    // coord (b,n,s,h)
    int max_threads_per_block);

template <typename T>
Status LaunchStridedCopy(cudaStream_t stream,
                         const T* in, int4 in_shape, longlong4 in_strides,  // coord (b,n,s,h)
                         T* out, longlong4 out_strides,                     // coord (b,n,s,h)
                         int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
