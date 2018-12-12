// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_transformer.h"

namespace onnxruntime {

class ConvActivationFusion : public onnxruntime::GraphTransformer {
 public:
  ConvActivationFusion() noexcept : onnxruntime::GraphTransformer("ConvActivationFusion", "Fusing Activation into Conv") {}
  Status Apply(onnxruntime::Graph& graph, bool& modified) const override;
};

}  // namespace onnxruntime
