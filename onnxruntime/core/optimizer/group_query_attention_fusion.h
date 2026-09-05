// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <utility>

#include "core/optimizer/graph_transformer.h"

namespace onnxruntime {

/**
@Class GroupQueryAttention
*/
class GroupQueryAttentionFusion : public GraphTransformer {
 public:
  explicit GroupQueryAttentionFusion(
      const InlinedHashSet<std::string_view>& compatible_execution_providers = {},
      NodeKernelSupportChecker node_kernel_support_checker = {}) noexcept
      : GraphTransformer("GroupQueryAttentionFusion", compatible_execution_providers),
        node_kernel_support_checker_(std::move(node_kernel_support_checker)) {
  }

  Status ApplyImpl(Graph& graph, bool& modified, int graph_level, const logging::Logger& logger) const override;

 private:
  const NodeKernelSupportChecker node_kernel_support_checker_;
};

}  // namespace onnxruntime
