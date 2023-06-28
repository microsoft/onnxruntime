// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class SceLossGradBiasFusion
Fuse SoftmaxCrossEntropyLossInternalGrad + Reshape(optional) + Sum/Add to SoftmaxCrossEntropyLossInternalGrad.
If it's Sum Op, it requires that it has only 2 inputs. Sum/Add must be non-broadcasting computation.

 input_0  scale_0  input_1  scale_1
      \     /          \   /
        Div             Div
          \            /
           \          /
            \        /
             \      /
    input_2    Add
         \     /
           Add
            |



*/
class ScaledSumFusion : public GraphTransformer {
 public:
  explicit ScaledSumFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("ScaledSumFusion", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
