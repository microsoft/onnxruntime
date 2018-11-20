// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_transformer.h"

namespace onnxruntime {

class UnsqueezeElimination : public onnxruntime::GraphTransformer {
 public:
  UnsqueezeElimination() noexcept : onnxruntime::GraphTransformer("EliminateUnsqueeze", "Eliminate unsequeeze node") {}
  Status Apply(onnxruntime::Graph& graph, bool& modified) const override;
};

}  // namespace onnxruntime
