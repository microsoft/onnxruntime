// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class UnsqueezeElimination : public onnxruntime::GraphTransformer {
 public:
  UnsqueezeElimination() noexcept : onnxruntime::GraphTransformer("EliminateUnsqueeze", "Eliminate unsqueeze node") {}

 private:
  Status ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level) const override;
};

}  // namespace onnxruntime
