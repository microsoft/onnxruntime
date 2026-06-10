// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/tensor_shape.h"

namespace onnxruntime {
namespace webgpu {

// Counts the trailing dimensions that are equal in BOTH operands and can therefore be merged
// into a single "shared" (non-broadcast) dimension for the vectorized broadcast path.
//
// Counting stops as soon as either operand runs out of real dimensions: an exhausted operand
// contributes an implicit size-1 dimension, which is a broadcast, not a shared dimension, and
// must not extend the shared run. As a result the returned count is bounded by
// min(lhs rank, rhs rank) (and by output_rank - 1, matching the loop that leaves at least one
// outer dimension). This bound is what prevents the downstream
// SizeFromDimension(rank - num_shared_dimension) from underflowing (size_t wrap to SIZE_MAX)
// when the operands have unequal ranks. See issue #28969.
//
// The product of the shared dimensions is returned via `shared_dimension_product`, which the
// caller uses to decide whether the shared run is divisible by 4.
//
// Defined inline in this lightweight, Dawn-free header so that any translation unit can use it
// (including the deviceless unit test) without taking a link dependency on the webgpu provider
// library, which is not linked into onnxruntime_provider_test in every build configuration
// (e.g. the plugin build), and without pulling Dawn/WebGPU headers into a CPU test translation
// unit. See issue #28969.
inline size_t CountSharedTrailingDimensions(const TensorShape& lhs_shape,
                                            const TensorShape& rhs_shape,
                                            size_t output_rank,
                                            int64_t& shared_dimension_product) {
  shared_dimension_product = 1;
  size_t num_shared_dimension = 0;
  for (size_t i = 1; i < output_rank; i++) {
    // Stop once either operand runs out of real dimensions. An exhausted operand contributes an
    // implicit size-1 broadcast dimension, which must not be merged into the shared (non-broadcast)
    // run; otherwise num_shared_dimension can exceed an operand's rank and the downstream
    // SizeFromDimension(rank - num_shared_dimension) underflows (size_t wrap). See issue #28969.
    if (lhs_shape.NumDimensions() < i || rhs_shape.NumDimensions() < i) {
      break;
    }
    int64_t dimA = lhs_shape[lhs_shape.NumDimensions() - i];
    int64_t dimB = rhs_shape[rhs_shape.NumDimensions() - i];
    if (dimA != dimB) {
      break;
    }
    shared_dimension_product *= dimA;
    num_shared_dimension++;
  }
  return num_shared_dimension;
}

}  // namespace webgpu
}  // namespace onnxruntime
