// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"

namespace onnxruntime {
namespace cuda {
template <typename T>
void ImplDivGradSimple(
    cudaStream_t stream,
    SimpleBroadcast simpleBroadcast,
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    size_t count,
    T* da_output_data,
    T* db_output_data);

template <typename T>
void ImplDivGradRhsPerChannelBatch1(
    cudaStream_t stream,
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    size_t count,
    const fast_divmod& fdm_H,
    T* da_output_data,
    T* db_output_data);

template <typename T>
void ImplDivGradRhsPerChannelBatchN(
    cudaStream_t stream,
    const T* a_data,
    const T* b_data,
    const T* dy_data,
    size_t count,
    const fast_divmod& fdm_H,
    const fast_divmod& fdm_C,
    T* da_output_data,
    T* db_output_data);

template <typename T>
void ImplDivGrad(
    cudaStream_t stream,
    int32_t output_rank,
    const TArray<int64_t>* a_padded_strides,
    const T* a_data,
    const TArray<int64_t>* b_padded_strides,
    const T* b_data,
    const T* dy_data,
    size_t count,
    const TArray<fast_divmod>* fdm_output_strides,
    T* da_output_data,
    T* db_output_data);
}  // namespace cuda
}  // namespace onnxruntime
