// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

int compute_reduction_buffer_size(int element_size, int size);

template <typename TIn, typename TOut>
void reduce_sum(hipStream_t stream, const TIn* input, TOut* output, int size, TOut* buffer);

template <typename TIn, typename TOut>
void reduce_square_sum(hipStream_t stream, const TIn* input, TOut* output, int size, TOut* buffer);

template <typename TIn, typename TOut>
void reduce_l2_norm(hipStream_t stream, const TIn* input, TOut* output, int size, TOut* buffer);

template <typename TIn, typename TOut>
void reduce_mean(hipStream_t stream, const TIn* data, TOut* output, int size, TOut* buffer);

template <typename TIn, typename TOut>
void reduce_matrix_rows(hipStream_t stream, const TIn* data, TOut* output, int m, int n);

}  // namespace rocm
}  // namespace onnxruntime