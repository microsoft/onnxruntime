// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class ConvMulFusion : public onnxruntime::GraphTransformer {
 public:
  ConvMulFusion() noexcept : onnxruntime::GraphTransformer("ConvMulFusion") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level) const override;
};

}  // namespace onnxruntime
