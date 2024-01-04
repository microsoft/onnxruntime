// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "contrib_ops/cpu/bert/attention_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

size_t GetAttentionScratchSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t sequence_length);

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t qk_head_size,
    size_t v_head_size,
    size_t sequence_length,
    void* fused_runner,
    bool use_flash_attention,
    bool use_memory_efficient_attention,
    bool no_qkv_workspace);

template <typename T>
struct PackedAttentionData {
  T* gemm_buffer;
  const T* bias;
  const T* relative_position_bias;
  const int32_t* token_offset;
  const int32_t* cumulative_sequence_length;

  T* workspace;
  T* output;

  void* fused_runner;

  bool use_memory_efficient_attention;
};

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    contrib::PackedAttentionParameters& parameters,
    PackedAttentionData<T>& data);

template <typename T>
Status LaunchTransposeRemovePadding(
    T* output, const T* input,
    const int* token_offset, const int token_count,
    const int batch_size, const int seq_len, const int number_heads, const int head_size,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
