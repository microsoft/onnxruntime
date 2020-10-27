// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace cuda {

namespace detail {
size_t compute_reduce_all_in_rows_buffer_size(int element_size, int element_alignment, int row_size, int num_rows);
}

template <typename TIn>
size_t compute_reduce_matrix_columns_buffer_size(int m, int n) {
  using TBuf = AccumulationType_t<TIn>;
  return detail::compute_reduce_all_in_rows_buffer_size(
      sizeof(TBuf), alignof(TBuf), n, m);
}

template <typename TIn>
size_t compute_reduction_buffer_size(int size) {
  using TBuf = AccumulationType_t<TIn>;
  return detail::compute_reduce_all_in_rows_buffer_size(
      sizeof(TBuf), alignof(TBuf), size, 1);
}

template <typename TIn, typename TOut>
Status reduce_sum(const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size);

template <typename TIn, typename TOut>
Status reduce_square_sum(const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size);

template <typename TIn, typename TOut>
Status reduce_l2_norm(const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size);

template <typename TIn, typename TOut>
Status reduce_mean(const TIn* data, TOut* output, int size, void* buffer, size_t buffer_size);

// Determine if a CUDNN reduction can be computed by reduce_matrix_rows.
bool is_matrix_row_reduction(
    const cudnnReduceTensorOp_t cudnn_reduce_op,
    const int m,
    const int n,
    const size_t rank,
    std::vector<int64_t> axes);

template <typename TIn, typename TOut>
Status reduce_matrix_rows(const TIn* data, TOut* output, int m, int n);

template <typename TIn, typename TOut>
Status reduce_matrix_columns(const TIn* data, TOut* output, int m, int n, void* buffer, size_t buffer_size);

}  // namespace cuda
}  // namespace onnxruntime