// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class GemmActivationFusion : public onnxruntime::GraphTransformer {
 public:
  GemmActivationFusion() noexcept
      : onnxruntime::GraphTransformer("GemmActivationFusion", "Fusing Activation into Gemm") {}
  Status ApplyImpl(onnxruntime::Graph& graph, bool& modified, int graph_level) const override;
};

}  // namespace onnxruntime
