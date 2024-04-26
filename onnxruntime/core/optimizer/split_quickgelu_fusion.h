// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"

#define DEBUG_LOG(x) LOGS(logger, VERBOSE) << x

namespace onnxruntime {

/**
@Class SplitQuickGeluFusion

Fuse Split followed by QuickGelu and Multiply
*/
class SplitQuickGeluFusion : public GraphTransformer {
 public:
  SplitQuickGeluFusion(const InlinedHashSet<std::string_view>& compatible_execution_providers = {}) noexcept
      : GraphTransformer("SplitQuickGeluFusion", compatible_execution_providers) {}

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;
};

}  // namespace onnxruntime
