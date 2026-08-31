// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cpu/bert/attention_parameters.h"
#include "contrib_ops/cuda/bert/packed_attention_data.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
T* PackedAttentionWorkspaceAt(T* workspace, size_t offset_bytes) {
  if (offset_bytes == 0) {
    return workspace;
  }

  return reinterpret_cast<T*>(reinterpret_cast<uint8_t*>(workspace) + offset_bytes);
}

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
