// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph.h"

namespace onnxruntime {
namespace training {

Status TransformGraphForMixedPrecision(
    Graph& graph,
    const std::unordered_set<std::string>& weights_to_train);
};
}  // namespace onnxruntime
