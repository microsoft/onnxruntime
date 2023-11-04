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

template <typename T>
struct GroupQueryAttentionData {
  // Input Tensors
  const T* query = nullptr;
  const T* key = nullptr;
  const T* value = nullptr;
  const T* past_key = nullptr;
  const T* past_value = nullptr;
  const int64_t* attention_mask = nullptr;
  int* seqlens_k = nullptr;
  // Flash buffers
  T* softmax_lse = nullptr;
  T* softmax_lse_accum = nullptr;
  T* out_accum = nullptr;
  // Memory Efficient buffers
  T* fmha_buffer = nullptr;
  T* k = nullptr;
  T* v = nullptr;
  int32_t* seqstart_q = nullptr;
  int32_t* seqstart_k = nullptr;
  // Output Tensors
  T* output = nullptr;
  T* present_key = nullptr;
  T* present_value = nullptr;
  // Kernel Flags
  bool use_flash_attention = false;
  bool use_memory_efficient_attention = false;
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
