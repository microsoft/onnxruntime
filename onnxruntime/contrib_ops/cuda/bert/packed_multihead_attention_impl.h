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

__host__ __device__ constexpr int64_t AdvanceTokenOffsetValidationIndex(
    int64_t index,
    int32_t grid_dimension,
    int32_t block_dimension) {
  return index + static_cast<int64_t>(grid_dimension) * block_dimension;
}

Status ValidatePackedMultiHeadAttentionTokenOffset(
    const int32_t* token_offset,
    int32_t token_offset_count,
    int32_t* validation_flag,
    cudaStream_t stream);

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
