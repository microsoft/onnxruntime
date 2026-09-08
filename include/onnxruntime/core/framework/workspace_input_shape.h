// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <utility>

#include <gsl/span>

#include "core/framework/tensor_shape.h"

namespace onnxruntime {

// Shape metadata for one positional node input used when declaring workspace requirements.
enum class WorkspaceInputShapeState {
  // The optional input is omitted at this position.
  Missing,
  // The input is present and rank/dimension metadata is available. This includes rank-0 scalars,
  // zero extents, and partial shapes. Unknown dimensions are encoded as -1; multiple -1 dimensions
  // are independent unknowns and do not imply symbolic equality. TensorShape::Size() returns -1
  // for a partial shape, so consumers must inspect dimensions rather than convert Size() to an
  // unsigned byte/count value.
  PresentWithShape,
  // The input is present, but no rank or dimension metadata is available.
  PresentWithoutShape,
};

// A PresentWithShape value may come from executable-graph metadata or max-shape inference. The
// latter is an estimation hint, not a proven upper bound, so consumers must retain runtime checks
// and allocation fallbacks. This Phase-A type intentionally carries no provenance state; deciding
// whether Phase-B allocation planning needs it is deferred until that runtime fallback contract is
// designed.
class WorkspaceInputShape {
 public:
  WorkspaceInputShape() = default;

  static WorkspaceInputShape PresentWithShape(const TensorShape& shape) {
    // TensorShape can be a non-owning view created by FromExistingBuffer(). Always copy the
    // dimensions so this value remains valid independently of the caller's storage.
    return WorkspaceInputShape{
        WorkspaceInputShapeState::PresentWithShape, TensorShape{shape.GetDims()}};
  }

  static WorkspaceInputShape PresentWithoutShape() {
    return WorkspaceInputShape{WorkspaceInputShapeState::PresentWithoutShape, TensorShape{}};
  }

  WorkspaceInputShapeState GetState() const noexcept {
    return state_;
  }

  const TensorShape* GetShape() const& noexcept {
    return state_ == WorkspaceInputShapeState::PresentWithShape ? &shape_ : nullptr;
  }

  // A pointer into a temporary WorkspaceInputShape would dangle immediately.
  const TensorShape* GetShape() const&& = delete;

 private:
  WorkspaceInputShape(WorkspaceInputShapeState state, TensorShape shape)
      : state_{state}, shape_{std::move(shape)} {
  }

  WorkspaceInputShapeState state_{WorkspaceInputShapeState::Missing};
  TensorShape shape_;
};

// Node inputs are positional. A caller may omit trailing optional inputs from the span; querying one
// of those positions has the same meaning as an explicit Missing entry.
inline const WorkspaceInputShape& GetWorkspaceInputShape(
    gsl::span<const WorkspaceInputShape> input_shapes, size_t input_index) {
  if (input_index < input_shapes.size()) {
    return input_shapes[input_index];
  }

  static const WorkspaceInputShape missing_input;
  return missing_input;
}

}  // namespace onnxruntime
