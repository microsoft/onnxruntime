// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <string>
#include <vector>
#include <utility>
#include "core/graph/graph.h"
#include "core/training/loss_function_registry.h"

namespace onnxruntime {
namespace training {

class LossFunctionBuilder {
 public:
  GraphAugmenter::GraphDefs Build(const Graph& graph, const LossFunctionInfo& loss_func_info) const;
};
}  // namespace training
}  // namespace onnxruntime
