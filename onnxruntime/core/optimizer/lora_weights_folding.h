// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

/**
@Class LoraWeightsFolding

Folds LoRA weights into the original model's weights in order to remove all runtime overhead.

*/
class LoraWeightsFolding : public GraphTransformer {
 public:
  LoraWeightsFolding(
    const IExecutionProvider& cpu_execution_provider,
    const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map,
    const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("LoraWeightsFolding", compatible_execution_providers),
        cpu_execution_provider_(cpu_execution_provider),
        initializers_to_share_map_(initializers_to_share_map) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  const IExecutionProvider& cpu_execution_provider_;
  const std::unordered_map<std::string, const OrtValue*>& initializers_to_share_map_;
};

}  // namespace onnxruntime
