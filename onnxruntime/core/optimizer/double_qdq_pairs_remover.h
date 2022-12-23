// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {
class DoubleQDQPairsRemover : public GraphTransformer {
public:
  DoubleQDQPairsRemover() : GraphTransformer("DoubleQDQPairsRemover", {}) {}

private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level,
                   const logging::Logger& logger) const override;
};
} // namespace onnxruntime
