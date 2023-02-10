// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file contains the definition of the PrepareForComputeMetadata for Slice operator
#pragma once

#include <cstdint>
#include <vector>
#include "core/common/gsl.h"
#include "core/framework/tensor_shape.h"

namespace onnxruntime {

namespace SliceOp {
struct PrepareForComputeMetadata {
  explicit PrepareForComputeMetadata(gsl::span<const int64_t> input_dimensions)
      : input_dimensions_(input_dimensions),
        ends_(input_dimensions.begin(), input_dimensions.end()),
        output_dims_(input_dimensions.begin(), input_dimensions.end()) {
    size_t dimension_count = input_dimensions.size();
    starts_.resize(dimension_count, 0);
    steps_.resize(dimension_count, 1);
  }

  gsl::span<const int64_t> input_dimensions_;
  TensorShapeVector starts_;
  TensorShapeVector ends_;
  TensorShapeVector steps_;
  TensorShapeVector output_dims_;
  TensorShapeVector flattened_input_dims_;
  TensorShapeVector* p_flattened_input_dims_ = &flattened_input_dims_;
  TensorShapeVector flattened_output_dims_;
  TensorShapeVector* p_flattened_output_dims_ = &flattened_output_dims_;
};

}  // namespace SliceOp
}  // namespace onnxruntime
