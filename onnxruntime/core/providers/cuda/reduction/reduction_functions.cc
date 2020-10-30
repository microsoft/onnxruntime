// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/reduction/reduction_functions.h"

#include <algorithm>
#include <utility>

#include "core/common/optional.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

ApplicableMatrixReduction get_applicable_matrix_reduction(
    const cudnnReduceTensorOp_t cudnn_reduce_op,
    const std::vector<int64_t>& dims, const std::vector<int64_t>& original_axes,
    int& m_out, int& n_out) {
  if (cudnn_reduce_op != CUDNN_REDUCE_TENSOR_ADD) {
    return ApplicableMatrixReduction::None;
  }

  const auto rank = gsl::narrow<int64_t>(dims.size());

  // min and max of single contiguous range of axes
  const auto minmax_axes = [&]() -> optional<std::pair<int64_t, int64_t>> {
    // empty axes means reduce all dimensions
    if (original_axes.empty()) {
      return std::make_pair(int64_t{0}, rank - 1);
    }

    // normalize axis values, sort, and remove duplicates
    const std::vector<int64_t> axes = [&original_axes, rank]() {
      std::vector<int64_t> result(original_axes);
      std::for_each(
          result.begin(), result.end(),
          [rank](int64_t& axis) { axis = HandleNegativeAxis(axis, rank); });
      std::sort(result.begin(), result.end());
      const auto last = std::unique(result.begin(), result.end());
      result.erase(last, result.end());
      return result;
    }();

    const bool are_axes_contiguous =
        std::adjacent_find(
            axes.begin(), axes.end(),
            [](int64_t a, int64_t b) { return a + 1 != b; }) == axes.end();

    if (!are_axes_contiguous) {
      return {};
    }

    return std::make_pair(axes.front(), axes.back());
  }();

  if (!minmax_axes.has_value()) {
    return ApplicableMatrixReduction::None;
  }

  const auto& min_axis = minmax_axes.value().first;
  const auto& max_axis = minmax_axes.value().second;

  const bool axes_from_beginning = min_axis == 0;
  const bool axes_to_end = max_axis == rank - 1;

  // handle axes anchored to one of beginning or end, not both
  if (axes_from_beginning == axes_to_end) {
    return ApplicableMatrixReduction::None;
  }

  const int64_t m_end_axis = axes_from_beginning ? max_axis + 1 : min_axis;

  const TensorShape& shape = TensorShape::ReinterpretBaseType(dims);

  const auto m = shape.SizeToDimension(m_end_axis);
  const auto n = shape.SizeFromDimension(m_end_axis);

  ORT_ENFORCE(m > 0 && n > 0, "shape must not have negative dimensions: ", shape);

  if (m > std::numeric_limits<int>::max() ||
      n > std::numeric_limits<int>::max()) {
    return ApplicableMatrixReduction::None;
  }

  m_out = gsl::narrow_cast<int>(m);
  n_out = gsl::narrow_cast<int>(n);

  return axes_from_beginning
             ? ApplicableMatrixReduction::Rows
             : ApplicableMatrixReduction::Columns;
}

}  // namespace cuda
}  // namespace onnxruntime
