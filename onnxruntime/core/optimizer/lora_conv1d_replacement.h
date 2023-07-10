// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

#define DEBUG_LOG(x) LOGS(logger, VERBOSE) << x
namespace onnxruntime {

class LoRAConv1dReplacement : public GraphTransformer {
 public:
  LoRAConv1dReplacement(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("LoRAConv1dReplacement", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
