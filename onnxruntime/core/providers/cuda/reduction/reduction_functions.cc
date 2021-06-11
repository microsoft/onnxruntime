// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/shared_library/provider_api.h"
#include "core/providers/cuda/reduction/reduction_functions.h"

#include <algorithm>
#include <cassert>
#include <iterator>
#include <utility>

#include "core/common/optional.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace cuda {

namespace {
// std::make_reverse_iterator is not implemented in older versions of GCC
#if !defined(__GNUC__) || __GNUC__ >= 5
using std::make_reverse_iterator;
#else
template <typename It>
std::reverse_iterator<It> make_reverse_iterator(It it) {
  return std::reverse_iterator<It>(it);
}
#endif

// gets min and max of single contiguous range of axes if available
optional<std::pair<int64_t, int64_t>> GetMinAndMaxContiguousAxes(
    int64_t rank,
    const std::vector<int64_t>& dims,
    const std::vector<int64_t>& original_axes) {
  assert(rank == static_cast<int64_t>(dims.size()));

  // empty axes means reduce all dimensions
  if (original_axes.empty()) {
    return std::make_pair(int64_t{0}, rank - 1);
  }

  // normalize axis values and sort
  const std::vector<int64_t> axes = [&original_axes, rank]() {
    std::vector<int64_t> result(original_axes);
    std::for_each(
        result.begin(), result.end(),
        [rank](int64_t& axis) { axis = HandleNegativeAxis(axis, rank); });
    std::sort(result.begin(), result.end());
    return result;
  }();

  assert(!axes.empty());

  const auto is_dim_one = [](int64_t dim) { return dim == 1; };

  for (auto a = axes.begin(), b = axes.begin() + 1;
       b != axes.end();
       ++a, ++b) {
    ORT_ENFORCE(*a != *b, "axes must not contain duplicate values");

    // if axis values are adjacent, the axes are contiguous
    if (*a + 1 == *b) {
      continue;
    }

    // if all dimension values between adjacent axes are 1,
    // treat the axes as contiguous
    if (std::all_of(dims.begin() + *a + 1, dims.begin() + *b, is_dim_one)) {
      continue;
    }

    // otherwise, not contiguous
    return std::nullopt;
  }

  // expand axes over surrounding dimensions with value of 1
  const int64_t min_axis = [&dims, &axes, &is_dim_one]() {
    const auto& min_given_axis = axes.front();
    // note that std::reverse_iterator(it) refers to the element at (it-1)
    // it -> reverse it: element offset of -1
    const auto before_min_given_axis_rit =
        make_reverse_iterator(dims.begin() + min_given_axis);
    const auto before_min_axis_rit =
        std::find_if_not(before_min_given_axis_rit, dims.rend(), is_dim_one);
    // reverse it -> it: element offset of +1
    return std::distance(dims.begin(), before_min_axis_rit.base());
  }();

  const int64_t max_axis = [&dims, &axes, &is_dim_one]() {
    const auto& max_given_axis = axes.back();
    const auto after_max_given_axis_it = dims.begin() + max_given_axis + 1;
    const auto after_max_axis_it =
        std::find_if_not(after_max_given_axis_it, dims.end(), is_dim_one);
    return std::distance(dims.begin(), after_max_axis_it - 1);
  }();

  return std::make_pair(min_axis, max_axis);
}
}  // namespace

ApplicableMatrixReduction get_applicable_matrix_reduction(
    const cudnnReduceTensorOp_t cudnn_reduce_op,
    const std::vector<int64_t>& dims, const std::vector<int64_t>& original_axes,
    int& m_out, int& n_out) {
  if (cudnn_reduce_op != CUDNN_REDUCE_TENSOR_ADD) {
    return ApplicableMatrixReduction::None;
  }

  const auto rank = gsl::narrow<int64_t>(dims.size());
  const auto min_and_max_axes = GetMinAndMaxContiguousAxes(rank, dims, original_axes);
  if (!min_and_max_axes.has_value()) {
    return ApplicableMatrixReduction::None;
  }

  const auto& min_axis = min_and_max_axes.value().first;
  const auto& max_axis = min_and_max_axes.value().second;

  // axes from beginning means row reduction, axes to end means column reduction
  // for axes from beginning to end, either works and we do row reduction
  const bool axes_from_beginning = min_axis == 0;
  const bool axes_to_end = max_axis == rank - 1;

  // handle axes anchored to beginning or end
  if (!axes_from_beginning && !axes_to_end) {
    return ApplicableMatrixReduction::None;
  }

  // the axis index right after the last flattened into matrix rows
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
