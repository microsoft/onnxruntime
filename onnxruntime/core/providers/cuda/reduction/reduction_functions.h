// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/accumulation_type.h"

namespace onnxruntime {
namespace cuda {

namespace detail {
size_t compute_reduce_matrix_columns_intermediate_buffer_size(
    int element_size, int num_rows, int num_cols);
}  // namespace detail

/**
 * Computes the size in bytes of the intermediate buffer needed by reduce_matrix_columns().
 * @tparam TIn The input data type.
 * @param m The number of matrix rows.
 * @param n The number of matrix columns.
 * @return The size of the intermediate buffer.
 */
template <typename TIn>
size_t compute_reduce_matrix_columns_buffer_size(int m, int n) {
  using TBuf = AccumulationType_t<TIn>;
  return detail::compute_reduce_matrix_columns_intermediate_buffer_size(
      sizeof(TBuf), m, n);
}

/**
 * Computes the size in bytes of the intermediate buffer needed by the reduce_x() functions.
 * @tparam TIn The input data type.
 * @param size The number of elements.
 * @return The size of the intermediate buffer.
 */
template <typename TIn>
size_t compute_reduction_buffer_size(int size) {
  using TBuf = AccumulationType_t<TIn>;
  return detail::compute_reduce_matrix_columns_intermediate_buffer_size(
      sizeof(TBuf), 1, size);
}

/** Computes the sum of the given elements. */
template <typename TIn, typename TOut>
Status reduce_sum(cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size);

/** Computes the sum of the squares of the given elements. */
template <typename TIn, typename TOut>
Status reduce_square_sum(cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size);

/** Computes the L2 norm of the given elements. */
template <typename TIn, typename TOut>
Status reduce_l2_norm(cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size);

/** Computes the mean of the given elements. */
template <typename TIn, typename TOut>
Status reduce_mean(cudaStream_t stream, const TIn* input, TOut* output, int size, void* buffer, size_t buffer_size);

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

/**
 * Reduces the rows in a row-major matrix to a single row containing the sum of each column.
 * @param input The input data.
 * @param output The output data.
 * @param m The number of matrix rows.
 * @param n The number of matrix columns.
 * @param reset_initial_output Whether to reset (i.e., zero) the output values first.
 */
template <typename TIn, typename TOut>
Status reduce_matrix_rows(cudaStream_t stream, const TIn* input, TOut* output, int m, int n, bool reset_initial_output = true);

/**
 * Reduces the columns in a row-major matrix to a single column containing the sum of each row.
 * @param input The input data.
 * @param output The output data.
 * @param m The number of matrix rows.
 * @param n The number of matrix columns.
 * @param buffer The intermediate buffer.
 * @param buffer_size The size of the intermediate buffer in bytes.
 */
template <typename TIn, typename TOut>
Status reduce_matrix_columns(cudaStream_t stream, const TIn* input, TOut* output, int m, int n, void* buffer, size_t buffer_size);

}  // namespace cuda
}  // namespace onnxruntime
