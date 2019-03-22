// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

class ConvMulFusion : public onnxruntime::GraphTransformer {
 public:
  ConvMulFusion() noexcept : onnxruntime::GraphTransformer("ConvMulFusion", "Fusing Mul into Conv") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, 
                   const std::vector<std::string>& provider_types, int graph_level) const override;
};

}  // namespace onnxruntime
