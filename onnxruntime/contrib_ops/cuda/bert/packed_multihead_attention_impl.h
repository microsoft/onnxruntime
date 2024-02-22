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

template <typename T>
struct PackedMultiHeadAttentionData {
  const T* query;
  const T* key;
  const T* value;
  const T* bias;
  const T* relative_position_bias;
  const int32_t* token_offset;
  const int32_t* cumulative_sequence_length;

  AttentionQkvFormat source_qkv_format;

  bool no_qkv_workspace;
  T* workspace;
  T* output;

  void* fused_runner;

  bool use_flash_attention;
  bool use_memory_efficient_attention;
};

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    cudaStream_t stream,
    contrib::PackedAttentionParameters& parameters,
    PackedMultiHeadAttentionData<T>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
