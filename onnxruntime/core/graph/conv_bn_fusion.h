// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/graph/graph_transformer.h"

namespace onnxruntime {

class ConvBNFusion : public onnxruntime::GraphTransformer {
 public:
  ConvBNFusion() noexcept : onnxruntime::GraphTransformer("ConvBNFusion", "Fusing BN into Conv") {}

 private:
  Status ApplyImpl(onnxruntime::Graph& graph, bool& modified) const override;
};
}  // namespace onnxruntime
