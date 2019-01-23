// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_transformer.h"

namespace onnxruntime {

class GemmActivationFusion : public onnxruntime::GraphTransformer {
 public:
  GemmActivationFusion() noexcept : onnxruntime::GraphTransformer("GemmActivationFusion", "Fusing Activation into Gemm") {}
  Status Apply(onnxruntime::Graph& graph, bool& modified) const override;
};

}  // namespace onnxruntime
