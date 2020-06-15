// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

int compute_reduction_buffer_size(int element_size, int size);

template <typename TIn, typename TOut>
void reduce_sum(const TIn* input, TOut* output, int size, TOut* buffer);

template <typename TIn, typename TOut>
void reduce_square_sum(const TIn* input, TOut* output, int size, TOut* buffer);

template <typename TIn, typename TOut>
void reduce_l2_norm(const TIn* input, TOut* output, int size, TOut* buffer);

template <typename TIn, typename TOut>
void reduce_mean(const TIn* data, TOut* output, int size, TOut* buffer);

// Determine if a CUDNN reduction can be computed by reduce_matrix_rows.
bool is_matrix_row_reduction(
    const cudnnReduceTensorOp_t cudnn_reduce_op,
    const int m,
    const int n,
    const size_t rank,
    std::vector<int64_t> axes);

template <typename TIn, typename TOut>
void reduce_matrix_rows(const TIn* data, TOut* output, int m, int n);

}  // namespace cuda
}  // namespace onnxruntime