// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "core/common/gsl.h"
#include "core/framework/allocator.h"
#include "contrib_ops/cpu/bert/attention_common.h"

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
    bool use_flash_attention,
    bool use_fused_cross_attention,
    bool use_memory_efficient_attention);

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
  const T* relative_position_bias = nullptr;

  bool has_qkv_workspace = false;
  T* workspace = nullptr;
  T* temp_k_workspace = nullptr;
  T* temp_v_workspace = nullptr;

  T* output = nullptr;
  T* present = nullptr;
  T* present_key = nullptr;
  T* present_value = nullptr;

  void* fused_runner = nullptr;
  const void* fused_cross_attention_kernel = nullptr;

  bool use_flash_attention = false;
  bool use_memory_efficient_attention = false;

  mutable CumulatedSequenceLengthCache* cumulated_sequence_length_q_cache = nullptr;
  mutable CumulatedSequenceLengthCache* cumulated_sequence_length_kv_cache = nullptr;

  // Intermediate data
  T* q = nullptr;
  T* k = nullptr;
  T* v = nullptr;
  T* scratch = nullptr;
  AttentionQkvFormat qkv_format = AttentionQkvFormat::Q_K_V_BSNH;

  // Flash buffers
  T* softmax_lse = nullptr;
  T* softmax_lse_accum = nullptr;
  T* out_accum = nullptr;
};

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
                           int sequence_length, int total_sequence_length, bool pass_past_in_kv,
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
Status LaunchStridedCopy(cudaStream_t stream,
                         const T* in, int4 in_shape, longlong4 in_strides,  // coord (b,n,s,h)
                         T* out, longlong4 out_strides,                     // coord (b,n,s,h)
                         int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
