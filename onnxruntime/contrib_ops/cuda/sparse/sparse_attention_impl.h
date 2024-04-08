// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "core/framework/allocator.h"
#include "core/providers/cuda/tunable/cuda_tunable.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
struct SparseAttentionData {
  // Input Tensors
  const T* query = nullptr;
  const T* key = nullptr;
  const T* value = nullptr;
  const T* past_key = nullptr;
  const T* past_value = nullptr;
  const T* cos_cache = nullptr;
  const T* sin_cache = nullptr;

  const int32_t* block_mask = nullptr;
  const int32_t* seqlens_k_total = nullptr;

  // Output Tensors
  T* output = nullptr;
  T* present_key = nullptr;
  T* present_value = nullptr;
};

template <typename T>
Status QkvToContext(
    const cudaDeviceProp& device_prop,
    cublasHandle_t& cublas,
    Stream* stream,
    contrib::SparseAttentionParameters& parameters,
    SparseAttentionData<T>& data);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
