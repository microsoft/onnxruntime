// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class ConvAddFusion : public onnxruntime::GraphTransformer {
 public:
  ConvAddFusion() noexcept : onnxruntime::GraphTransformer("ConvAddFusion") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;
};

}  // namespace onnxruntime
