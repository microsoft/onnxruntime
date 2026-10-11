// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include "core/graph/node_arg.h"

namespace onnxruntime {

// Data propagation carries a small "shape value" in one of two non-interchangeable channels
// on a NodeArg: a rank-0 scalar (inferred_scalar_value_) or a rank>=1 list of values
// (inferred_shape_values_). The helpers below let custom data-propagation ops read and write a
// single-element value while preserving its rank (rank-0 scalar vs rank-1 [1]), so a producer
// and its consumers cannot silently disagree on rank (e.g. Gather feeding Mul feeding TopK).

// Reads a single int64 shape value carried by a NodeArg's data propagation, accepting either a
// rank-0 scalar value or a rank-1 single-element value. On success, sets `value`, sets
// `is_rank1` to false for a scalar source or true for a rank-1 [1] source, and returns true.
// Returns false if the NodeArg carries no usable single-element shape value.
inline bool TryGetSinglePropagatedShapeValue(const NodeArg& input_def, int64_t& value, bool& is_rank1) {
  if (input_def.GetInferredShapeScalarValue().has_value()) {
    value = input_def.GetInferredShapeScalarValue().value();
    is_rank1 = false;
    return true;
  }

  const auto& inferred_values = input_def.GetInferredShapeValues();
  if (inferred_values.has_value() &&
      inferred_values->dim_size() == 1 &&
      inferred_values->dim(0).has_dim_value()) {
    value = inferred_values->dim(0).dim_value();
    is_rank1 = true;
    return true;
  }

  return false;
}

// Stores a single int64 shape value on `output_def`, as a rank-0 scalar when `is_rank1` is false
// or as a rank-1 single-element value when `is_rank1` is true. The rank-1 representation mirrors
// how Graph::getInputData() reconstructs a TensorProto (dims=[1]) from inferred_shape_values_.
// The setter is correct-by-construction: it populates exactly one channel and clears the other, so
// the scalar-first reader (TryGetSinglePropagatedShapeValue) and the values-first getInputData()
// can never disagree on rank even if `output_def` carried a stale value from another channel.
inline void SetSinglePropagatedShapeValue(NodeArg& output_def, int64_t value, bool is_rank1) {
  if (!is_rank1) {
    output_def.SetInferredShapeScalarValue(value);
    // Keep exactly one channel populated: drop any stale values channel that getInputData() would
    // otherwise prefer over this scalar.
    output_def.GetMutableInferredShapeValues().reset();
    return;
  }

  auto& inferred_values = output_def.GetMutableInferredShapeValues();
  if (!inferred_values.has_value()) {
    inferred_values.emplace();
  }
  inferred_values->clear_dim();
  inferred_values->add_dim()->set_dim_value(value);
  // Keep exactly one channel populated: drop any stale scalar that the scalar-first reader would
  // otherwise return ahead of this rank-1 value.
  output_def.ClearInferredShapeScalarValue();
}

}  // namespace onnxruntime
