// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <numeric>

#include "core/common/inlined_containers.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/common.h"

namespace onnxruntime {

struct MeanVarianceNormalizationPlan {
  InlinedVector<size_t> permutation;
  InlinedVector<size_t> inverse_permutation;
  TensorShape shape;
  int64_t rows;
  int64_t columns;
  size_t unreduced_rank;
  bool transpose;
};

inline Status PrepareMeanVarianceNormalization(const TensorShape& shape, gsl::span<const int64_t> axes,
                                               MeanVarianceNormalizationPlan& plan) {
  const size_t rank = shape.NumDimensions();
  InlinedVector<size_t> normalized_axes;
  if (axes.empty()) {
    normalized_axes.resize(rank);
    std::iota(normalized_axes.begin(), normalized_axes.end(), size_t{0});
  } else {
    normalized_axes.reserve(axes.size());
    for (int64_t axis : axes) {
      ORT_RETURN_IF_NOT(IsAxisInRange(axis, static_cast<int64_t>(rank)),
                        "MeanVarianceNormalization axis ", axis, " is invalid for rank ", rank,
                        ". Set axes to values in [-rank, rank-1], or use an empty list to reduce all dimensions.");
      normalized_axes.push_back(static_cast<size_t>(HandleNegativeAxis(axis, static_cast<int64_t>(rank))));
    }
    std::sort(normalized_axes.begin(), normalized_axes.end());
    ORT_RETURN_IF_NOT(std::adjacent_find(normalized_axes.begin(), normalized_axes.end()) == normalized_axes.end(),
                      "MeanVarianceNormalization axes contains duplicates. Specify each reduction axis only once.");
  }

  plan.permutation.clear();
  plan.permutation.reserve(rank);
  for (size_t axis = 0; axis < rank; ++axis) {
    if (!std::binary_search(normalized_axes.begin(), normalized_axes.end(), axis)) {
      plan.permutation.push_back(axis);
    }
  }
  plan.permutation.insert(plan.permutation.end(), normalized_axes.begin(), normalized_axes.end());
  plan.inverse_permutation.resize(rank);
  TensorShapeVector dims;
  dims.reserve(rank);
  plan.transpose = false;
  for (size_t axis = 0; axis < rank; ++axis) {
    plan.inverse_permutation[plan.permutation[axis]] = axis;
    dims.push_back(shape[plan.permutation[axis]]);
    plan.transpose |= plan.permutation[axis] != axis;
  }
  plan.shape = TensorShape(dims);
  const size_t outer_rank = rank - normalized_axes.size();
  plan.unreduced_rank = outer_rank;
  plan.rows = plan.shape.SizeToDimension(outer_rank);
  plan.columns = plan.shape.SizeFromDimension(outer_rank);
  return Status::OK();
}

}  // namespace onnxruntime
