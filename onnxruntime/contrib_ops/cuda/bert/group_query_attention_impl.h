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

size_t GetAttentionWorkspaceSize(
    size_t element_size,
    size_t batch_size,
    size_t num_heads,
    size_t kv_num_heads,
    size_t head_size,
    size_t sequence_length,
    size_t kv_sequence_length,
    size_t total_sequence_length,
    bool use_flash_attention);

template <typename T>
struct GroupQueryAttentionData {
  const T* query;
  const T* key;
  const T* value;
  const T* past;
  const T* past_key;
  const T* past_value;
  bool has_qkv_workspace;
  T* workspace;
  T* temp_k_workspace;
  T* temp_v_workspace;
  T* softmax_lse;
  T* softmax_lse_accum;
  T* out_accum;
  int* seqlens_k;
  T* output;
  T* present;
  T* present_key;
  T* present_value;
  bool use_flash_attention;
};

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* stream,
    contrib::GroupQueryAttentionParameters& parameters,
    GroupQueryAttentionData<T>& data);


}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
