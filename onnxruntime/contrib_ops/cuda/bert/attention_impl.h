// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <gsl/gsl>
#include <iostream>
#include <mutex>
#include "core/framework/allocator.h"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cuda/bert/attention_data.h"
#include "contrib_ops/cuda/bert/fastertransformer_decoder_attention/decoder_masked_multihead_attention_impl.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

constexpr int kCumulatedSequenceLengthCacheMaxBatchSize = 128;

// longlong4 is deprecated in cuda 13.
// LongLong4 is similar to longlong4_32a, except this is also visible in Host compiler (longlong4_32a is only visible to nvcc);
typedef struct __align__(32) {
    long long int  x, y, z, w;
} LongLong4;

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
    bool use_lean_attention,
    bool use_fused_cross_attention,
    bool use_memory_efficient_attention,
    bool use_cudnn_flash_attention,
    bool no_qkv_workspace);

// Return true if it does not need qkv workspace, false otherwise.
template <typename T>
bool NoQkvWorkspace(contrib::AttentionParameters& parameters, AttentionData<T>& data);

template <typename T>
Status PrepareQkv(contrib::AttentionParameters& parameters,
                  AttentionData<T>& data,
                  cudaStream_t stream,
                  int max_threads_per_block);

template <typename T, typename QK = T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudnnHandle_t& cudnn,
    Stream* stream,
    contrib::AttentionParameters& parameters,
    AttentionData<T>& data);

template <typename T, typename QK>
Status LaunchDecoderMaskedMultiHeadAttention(
    const DecoderMaskedMultiHeadAttentionParameters& parameters,
    cudaStream_t stream,
    const int head_size);

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

template <typename T>
Status ConcatPastToPresent(int batch_size, int num_heads, int qk_head_size, int v_head_size,
                           int sequence_length, int total_sequence_length,
                           cudaStream_t stream,
                           int max_threads_per_block,
                           AttentionData<T>& data);

template <typename T>
Status PastPresentBufferShare(int batch_size, int num_heads, int qk_head_size, int v_head_size,
                              int sequence_length, void* fused_runner,
                              contrib::AttentionParameters& parameters,
                              AttentionData<T>& data,
                              cudaStream_t stream,
                              int max_threads_per_block);

template <typename T>
Status LaunchStridedCopy(
    cudaStream_t stream,
    const T* in, int4 in_shape, LongLong4 in_strides, const int* in_seqlens_offset,  // coord (b,n,s,h)
    T* out, LongLong4 out_strides, const int* out_seqlens_offset,                    // coord (b,n,s,h)
    int max_threads_per_block);

template <typename T>
Status LaunchStridedCopy(cudaStream_t stream,
                         const T* in, int4 in_shape, LongLong4 in_strides,  // coord (b,n,s,h)
                         T* out, LongLong4 out_strides,                     // coord (b,n,s,h)
                         int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
