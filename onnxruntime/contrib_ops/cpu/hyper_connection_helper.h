// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/tensor.h"

namespace onnxruntime::contrib::hyper_connection {

struct StreamShape {
  int64_t rows;
  int64_t branches;
  int64_t hidden;
  size_t prefix_rank;
  bool flattened;
  TensorShapeVector reduced_shape;
};

inline Status ResolveStreamShape(const TensorShape& shape, int64_t num_branches,
                                 StreamShape& result) {
  const auto rank = shape.NumDimensions();
  result.flattened = num_branches != 0;
  if (result.flattened) {
    ORT_RETURN_IF_NOT(num_branches > 0, "num_branches must be positive");
    ORT_RETURN_IF_NOT(rank >= 1, "flattened streams must have rank at least 1");
    const int64_t width = shape[rank - 1];
    ORT_RETURN_IF_NOT(width > 0 && width % num_branches == 0,
                      "flattened stream width must be positive and divisible by num_branches");
    result.branches = num_branches;
    result.hidden = width / num_branches;
    result.prefix_rank = rank - 1;
  } else {
    ORT_RETURN_IF_NOT(rank >= 2, "grouped streams must have rank at least 2");
    result.branches = shape[rank - 2];
    result.hidden = shape[rank - 1];
    ORT_RETURN_IF_NOT(result.branches > 0 && result.hidden > 0,
                      "branch and hidden dimensions must be positive");
    result.prefix_rank = rank - 2;
  }

  result.rows = shape.SizeToDimension(result.prefix_rank);
  result.reduced_shape.assign(shape.GetDims().begin(),
                              shape.GetDims().begin() + result.prefix_rank);
  result.reduced_shape.push_back(result.hidden);
  return Status::OK();
}

inline bool PrefixMatches(const TensorShape& shape, const TensorShape& streams,
                          size_t prefix_rank) {
  if (shape.NumDimensions() < prefix_rank) {
    return false;
  }
  for (size_t i = 0; i < prefix_rank; ++i) {
    if (shape[i] != streams[i]) {
      return false;
    }
  }
  return true;
}

enum class GateLayout {
  Scalar,
  Branch,
  BranchSingleton,
  Feature,
};

inline Status ResolveGateShape(const TensorShape& gate_shape,
                               const TensorShape& stream_shape,
                               const StreamShape& params, bool allow_scalar,
                               GateLayout& layout,
                               bool allow_flattened_feature = false) {
  if (allow_scalar && gate_shape.NumDimensions() == 0) {
    layout = GateLayout::Scalar;
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(PrefixMatches(gate_shape, stream_shape, params.prefix_rank),
                    "gate leading dimensions must match streams");
  const size_t suffix_rank = gate_shape.NumDimensions() - params.prefix_rank;
  if (suffix_rank == 1 && gate_shape[params.prefix_rank] == params.branches) {
    layout = GateLayout::Branch;
    return Status::OK();
  }
  if (allow_flattened_feature && params.flattened && suffix_rank == 1 &&
      gate_shape[params.prefix_rank] == params.branches * params.hidden) {
    layout = GateLayout::Feature;
    return Status::OK();
  }
  if (suffix_rank == 2 && gate_shape[params.prefix_rank] == params.branches &&
      gate_shape[params.prefix_rank + 1] == 1) {
    layout = GateLayout::BranchSingleton;
    return Status::OK();
  }
  if (suffix_rank == 2 && gate_shape[params.prefix_rank] == params.branches &&
      gate_shape[params.prefix_rank + 1] == params.hidden) {
    layout = GateLayout::Feature;
    return Status::OK();
  }
  const char* expected_shapes;
  if (allow_scalar) {
    expected_shapes = allow_flattened_feature && params.flattened
                          ? "gate must be scalar or have suffix (C), (C, 1), (C, H), or (C * H)"
                          : "gate must be scalar or have suffix (C), (C, 1), or (C, H)";
  } else {
    expected_shapes = allow_flattened_feature && params.flattened
                          ? "gate must have suffix (C), (C, 1), (C, H), or (C * H)"
                          : "gate must have suffix (C), (C, 1), or (C, H)";
  }
  return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, expected_shapes);
}

inline Status ValidateScale(const TensorShape& scale_shape,
                            const StreamShape& params) {
  const auto rank = scale_shape.NumDimensions();
  const bool valid = (rank == 1 &&
                      (scale_shape[0] == params.hidden ||
                       (scale_shape[0] % params.branches == 0 &&
                        scale_shape[0] / params.branches == params.hidden))) ||
                     (rank == 2 && scale_shape[0] == params.branches &&
                      scale_shape[1] == params.hidden);
  ORT_RETURN_IF_NOT(valid, "scale must have shape (H), (C * H), or (C, H)");
  return Status::OK();
}

inline Status ValidateReduced(const TensorShape& shape,
                              const StreamShape& params) {
  ORT_RETURN_IF_NOT(shape == TensorShape(params.reduced_shape),
                    "block_output must have shape (..., H)");
  return Status::OK();
}

inline Status ValidateStreamMix(const TensorShape& shape,
                                const TensorShape& streams,
                                const StreamShape& params) {
  ORT_RETURN_IF_NOT(PrefixMatches(shape, streams, params.prefix_rank),
                    "stream_mix leading dimensions must match streams");
  ORT_RETURN_IF_NOT(shape.NumDimensions() == params.prefix_rank + 2 &&
                        shape[params.prefix_rank] == params.branches &&
                        shape[params.prefix_rank + 1] == params.branches,
                    "stream_mix must have shape (..., C, C)");
  return Status::OK();
}

inline int64_t GateOffset(GateLayout layout, int64_t row, int64_t branch,
                          int64_t feature, int64_t branches, int64_t hidden) {
  switch (layout) {
    case GateLayout::Scalar:
      return 0;
    case GateLayout::Branch:
    case GateLayout::BranchSingleton:
      return row * branches + branch;
    case GateLayout::Feature:
      return (row * branches + branch) * hidden + feature;
  }
  return 0;
}

}  // namespace onnxruntime::contrib::hyper_connection
