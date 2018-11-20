// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_transformer.h"

namespace onnxruntime {

class ConvAddFusion : public onnxruntime::GraphTransformer {
public:
  ConvAddFusion() noexcept : onnxruntime::GraphTransformer("ConvAddFusion", "Fusing Add into Conv") {}
  Status Apply(onnxruntime::Graph& graph, bool& modified) const override;
};

}  // namespace onnxruntime
