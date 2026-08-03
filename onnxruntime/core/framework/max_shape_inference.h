// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/max_shape_override.h"

namespace onnxruntime {

class Graph;

/// Applies graph-input overrides to a shadow copy of graph, runs ORT shape inference,
/// and captures every concrete NodeArg shape without modifying graph.
Status InferMaxShapes(const Graph& graph,
                      const MaxShapeOverrideMap& input_overrides,
                      MaxShapeInferenceResult& result);

}  // namespace onnxruntime
