// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class LoraWeightsFolding

Folds LoRA weights into the original model's weights in order to remove all runtime overhead.

*/
class LoraWeightsFolding : public GraphTransformer {
 public:
  LoraWeightsFolding(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("LoraWeightsFolding", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
