// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file contains the defnition of the PrepareForComputeMetadata for Slice operator
#pragma once

namespace onnxruntime {

namespace SliceOp {
struct PrepareForComputeMetadata {
  PrepareForComputeMetadata() = delete;
  PrepareForComputeMetadata(const std::vector<int64_t>& input_dimensions)
      : input_dimensions_(input_dimensions) {
    size_t dimension_count = input_dimensions.size();
    starts_.resize(dimension_count, 0);
    ends_ = input_dimensions;
    steps_.resize(dimension_count, 1);
    output_dims_ = input_dimensions;
  }

  const std::vector<int64_t>& input_dimensions_;
  std::vector<int64_t> starts_;
  std::vector<int64_t> ends_;
  std::vector<int64_t> steps_;
  std::vector<int64_t> output_dims_;
  std::vector<int64_t> flattened_output_dims_;
  std::vector<int64_t>* p_flattened_output_dims_ = &flattened_output_dims_;
};

}  // namespace SliceOp
}  // namespace onnxruntime
