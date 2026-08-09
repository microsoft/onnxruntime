// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class SceLossGradBiasFusion
Fuse SoftmaxCrossEntropyLossInternalGrad + Reshape(optional) + Sum/Add to SoftmaxCrossEntropyLossInternalGrad.
If it's Sum Op, it requires that it has only 2 inputs. Sum/Add must be non-broadcasting computation.
*/
class SceLossGradBiasFusion : public GraphTransformer {
 public:
  explicit SceLossGradBiasFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("SceLossGradBiasFusion", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
