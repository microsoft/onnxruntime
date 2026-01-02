// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class MulAddFusion

Rewrite rule that fuses two Mul+Add nodes to a single Batchnorm node.

Determines whether a Mul followed by an Add can be safely fused into a
BatchNormalization node. The fusion is based on the observation that:

  Y = (X * scale) + bias

is mathematically equivalent to a BatchNormalization operation when the
BatchNorm parameters are set to:

  mean    = 0
  var     = 1
  epsilon = 0

with

  BatchNorm(X) = (X - mean) / sqrt(var + epsilon) * scale + bias
               = (X - 0) / sqrt(1 + 0) * scale + bias
               = X * scale + bias

*/
class MulAddFusion : public GraphTransformer {
 public:
  MulAddFusion() noexcept : GraphTransformer("MulAddFusion") {}

 private:
  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

  bool SatisfyCondition(const Graph& graph, const Node& node, const logging::Logger&) const;
  Status FuseMulAdd(Node& node, Graph& graph, bool& modified, const logging::Logger&) const;
};

}  // namespace onnxruntime
