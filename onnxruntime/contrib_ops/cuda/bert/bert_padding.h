// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include <cuda_fp16.h>
#include <cublas_v2.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

__host__ __device__ constexpr int64_t AdvancePaddingTokenOffsetValidationIndex(
    int64_t index,
    int32_t grid_dimension,
    int32_t block_dimension) {
  return index + static_cast<int64_t>(grid_dimension) * block_dimension;
}

// Build token indice for non-padding tokens and padding tokens.
void LaunchGetTokenOffset(int* token_count_buffer,
                          int* token_offset,
                          int* cumulated_token_count,
                          const int* sequence_token_count,
                          const int batch_size,
                          const int sequence_length,
                          cudaStream_t stream);

Status ValidateTokenOffset(
    const int* token_offset,
    int token_offset_count,
    int* validation_flag,
    cudaStream_t stream);

// Remove paddings from input.
template <typename T>
void LaunchRemovePadding(
    T* output, const T* input, const int* token_offset, const int token_count, const int hidden_size,
    cudaStream_t stream);

// Rebuild paddings to restore output shape.
template <typename T>
void LaunchRestorePadding(
    T* output, const T* input, const int* token_offset, const int token_count, const int hidden_size,
    const int batch_size, const int sequence_length,
    cudaStream_t stream);

// Padding offset for TensorRT fused attention kernel
void LaunchTrtSequenceOffset(int* trt_mha_padding_offset,
                             const int* mask_index,
                             const int batch_size,
                             cudaStream_t stream);

void LaunchTrtSequenceOffset(int* trt_mha_padding_offset,
                             const int* mask_index,
                             const int batch_size,
                             const int sequence_length,
                             cudaStream_t stream);

void LaunchTrtSequenceOffset2d(int* trt_mha_padding_offset,
                               const int* mask_index,
                               const int batch_size,
                               const int sequence_length,
                               cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
