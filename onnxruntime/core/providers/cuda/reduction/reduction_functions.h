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

/**
 * Computes the size in bytes of the intermediate buffer needed by reduce_matrix_columns().
 */
template <typename TIn>
size_t compute_reduce_matrix_columns_buffer_size(int m, int n) {
  using TBuf = AccumulationType_t<TIn>;
  return detail::compute_reduce_all_in_rows_buffer_size(
      sizeof(TBuf), alignof(TBuf), n, m);
}

/**
 * Computes the size in bytes of the intermediate buffer needed by the reduce_x() functions.
 */
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

enum class ApplicableMatrixReduction {
  // can use reduce_matrix_rows()
  Rows,
  // can use reduce_matrix_columns()
  Columns,
  // no optimized matrix reduction function applies
  None,
};

/**
 * Determines whether a cuDNN reduction can be computed by an optimized matrix reduction function.
 * @param cudnn_reduce_op The cuDNN reduction op type.
 * @param dims The input dimensions.
 * @param axes The reduction axes.
 * @param[out] m If matrix reduction is possible, the number of matrix rows to use.
 * @param[out] n If matrix reduction is possible, the number of matrix columns to use.
 * @return The type of matrix reduction that can be done.
 */
ApplicableMatrixReduction get_applicable_matrix_reduction(
    const cudnnReduceTensorOp_t cudnn_reduce_op,
    const std::vector<int64_t>& dims, const std::vector<int64_t>& axes,
    int& m, int& n);

template <typename TIn, typename TOut>
Status reduce_matrix_rows(const TIn* data, TOut* output, int m, int n, bool reset_initial_output = true);

template <typename TIn, typename TOut>
Status reduce_matrix_columns(const TIn* data, TOut* output, int m, int n, void* buffer, size_t buffer_size);

}  // namespace cuda
}  // namespace onnxruntime
