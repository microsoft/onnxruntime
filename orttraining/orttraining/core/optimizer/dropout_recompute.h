// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"

namespace onnxruntime {

Node& InsertDropoutRecompute(Graph& graph, Node& node, bool use_original_input);

}  // namespace onnxruntime
