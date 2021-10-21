// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file contains the functions compute the starts, steps (strides) and output shape
// for Slice op, which can be called from other ops or EPs.
#pragma once
#include "core/providers/cpu/tensor/slice_compute_metadata.h"

namespace onnxruntime {

namespace SliceOp {
// compute output_dims without steps (Slice V1-9 & DynamicSlice)
// Please note this will not Flatten the output shape
inline Status PrepareForComputeHelper(const std::vector<int64_t>& raw_starts,
                                      const std::vector<int64_t>& raw_ends,
                                      const std::vector<int64_t>& raw_axes,
                                      SliceOp::PrepareForComputeMetadata& compute_metadata) {
  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<int64_t> axes(raw_axes);
  if (axes.empty()) {
    //axes are omitted, they are set to[0, ..., ndim - 1]
    axes.resize(raw_starts.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  // Iterate through the provided axes and override the start/end ranges
  std::unordered_set<int64_t> unique_axes;
  const auto& dimension_count = compute_metadata.input_dimensions_.size();
  for (size_t axis_index = 0, axes_count = axes.size(); axis_index < axes_count; ++axis_index) {
    const auto axis = HandleNegativeAxis(axes[axis_index], dimension_count);  // handle negative and enforce axis is valid
    if (axis >= static_cast<int64_t>(dimension_count) || axis < 0)
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "'axes' has an axis outside of the tensor dimension count");
    if (unique_axes.find(axis) != unique_axes.end())
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "'axes' has duplicates");
    unique_axes.insert(axis);
    const auto dim_value = compute_metadata.input_dimensions_[axis];

    // process start
    auto start = raw_starts[axis_index];
    if (start < 0)
      start += dim_value;
    compute_metadata.starts_[axis] = std::clamp(start, int64_t{0}, dim_value);

    // process end
    auto end = raw_ends[axis_index];
    if (end < 0)
      end += dim_value;
    compute_metadata.ends_[axis] = std::clamp(end, int64_t{0}, dim_value);

    // find output dim value for this axis
    const auto temp = compute_metadata.ends_[axis] - compute_metadata.starts_[axis];
    if (temp < 0)
      compute_metadata.output_dims_[axis] = 0;
    else
      compute_metadata.output_dims_[axis] = temp;
  }

  return Status::OK();
}

// compute output_dims with steps (Slice V10)
// Please note this will not Flatten the output shape
inline Status PrepareForComputeHelper(const std::vector<int64_t>& raw_starts,
                                      const std::vector<int64_t>& raw_ends,
                                      const std::vector<int64_t>& raw_axes,
                                      const std::vector<int64_t>& raw_steps,
                                      SliceOp::PrepareForComputeMetadata& compute_metadata) {
  // Initialize axes to the provided axes attribute or to the default sequence
  std::vector<int64_t> axes(raw_axes);

  if (axes.empty()) {
    // axes are omitted, they are set to[0, ..., ndim - 1]
    axes.resize(raw_starts.size());
    std::iota(axes.begin(), axes.end(), 0);
  }

  // Iterate through the provided axes and override the start/end/steps ranges
  std::unordered_set<int64_t> unique_axes;
  const auto& dimension_count = compute_metadata.input_dimensions_.size();
  for (size_t axis_index = 0, axes_count = axes.size(); axis_index < axes_count; ++axis_index) {
    const auto axis = axes[axis_index] < 0 ? axes[axis_index] + static_cast<int64_t>(dimension_count) : axes[axis_index];
    if (axis >= static_cast<int64_t>(dimension_count) || axis < 0)
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "'axes' has an axis outside of the tensor dimension count");
    if (unique_axes.find(axis) != unique_axes.end())
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "'axes' has duplicates");
    unique_axes.insert(axis);
    const auto dim_value = compute_metadata.input_dimensions_[axis];

    // process step
    auto step = axis_index < raw_steps.size() ? raw_steps[axis_index] : 1;
    // clamp step to avoid overflow if there's a stupidly large value (which will be multiplied in SliceImpl)
    // as long as the clamped value is >= the size of the dimension a single step will push us past the end
    step = std::clamp(step, -dim_value, dim_value);
    if (step == 0)
      return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "'step' value cannot be 0");
    compute_metadata.steps_[axis] = step;

    // process start
    auto start = raw_starts[axis_index];
    if (start < 0)
      start += dim_value;
    if (step < 0)
      compute_metadata.starts_[axis] = std::clamp(start, int64_t{0}, dim_value - 1);
    else
      compute_metadata.starts_[axis] = std::clamp(start, int64_t{0}, dim_value);

    // process end
    auto end = raw_ends[axis_index];
    // INT_MAX has a special meaning for end according to spec
    // equivalent to 'None' in numpy
    // it represent slicing to the end of the dimension
    if (end == std::numeric_limits<int32_t>::max() ||
        end == std::numeric_limits<int64_t>::max()) {
      end = step < 0 ? -1 : dim_value;
    } else {
      if (end < 0)
        end += dim_value;
      if (step < 0)
        end = std::clamp(end, int64_t{-1}, dim_value);
      else
        end = std::clamp(end, int64_t{0}, dim_value);
    }

    compute_metadata.ends_[axis] = end;

    // find output dim value for this axis
    const auto temp = static_cast<int64_t>(ceil(1.0 * (compute_metadata.ends_[axis] - compute_metadata.starts_[axis]) / step));
    if (temp < 0)
      compute_metadata.output_dims_[axis] = 0;
    else
      compute_metadata.output_dims_[axis] = temp;
  }

  return Status::OK();
}

}  // namespace SliceOp
}  // namespace onnxruntime
