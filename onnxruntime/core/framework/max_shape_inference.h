// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/max_shape_override.h"

namespace onnxruntime {

class Graph;

/// Computes estimation-only maximum shapes without changing the executable graph.
///
/// A disposable shadow model is created from `graph`, including model-local functions,
/// generated schemas, initializers, and nested control-flow graphs. `input_overrides`
/// are validated against the source graph's explicit inputs and applied to the
/// corresponding shadow inputs. Values assigned to symbolic dimensions are also
/// propagated to other graph inputs that use the same symbols.
///
/// Resolving the shadow graph runs normal ORT shape inference recursively. Every fully
/// concrete NodeArg shape is then recorded against the identity of its corresponding
/// source graph, including nested subgraphs. `result` is cleared before inference, and
/// the source graph's shape metadata is never modified. The inferred downstream shapes
/// are estimation hints, not proven upper bounds: shape inference does not establish that
/// every operator's output shape is monotonic with respect to its input dimensions.
Status InferMaxShapes(const Graph& graph,
                      const MaxShapeOverrideMap& input_overrides,
                      MaxShapeInferenceResult& result);

}  // namespace onnxruntime
