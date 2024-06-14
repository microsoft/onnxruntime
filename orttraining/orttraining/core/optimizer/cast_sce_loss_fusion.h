// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class CastSceLossFusion
Fuse Cast + SoftmaxCrossEntropyLossInternal to SoftmaxCrossEntropyLossInternal.
*/
class CastSceLossFusion : public GraphTransformer {
 public:
  explicit CastSceLossFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("CastSceLossFusion", compatible_execution_providers) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
