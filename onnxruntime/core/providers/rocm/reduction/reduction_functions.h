// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/rocm/rocm_common.h"

namespace onnxruntime {
namespace rocm {

int compute_reduction_buffer_size(int element_size, int size);

template <typename TIn, typename TOut>
void reduce_sum(const TIn* input, TOut* output, int size, TOut* buffer);

template <typename TIn, typename TOut>
void reduce_square_sum(const TIn* input, TOut* output, int size, TOut* buffer);

template <typename TIn, typename TOut>
void reduce_l2_norm(const TIn* input, TOut* output, int size, TOut* buffer);

template <typename TIn, typename TOut>
void reduce_mean(const TIn* data, TOut* output, int size, TOut* buffer);

template <typename TIn, typename TOut>
void reduce_matrix_rows(const TIn* data, TOut* output, int m, int n);

}  // namespace rocm
}  // namespace onnxruntime