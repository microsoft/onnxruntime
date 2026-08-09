// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/constant_folding_dq_node.h"
#include "core/optimizer/graph_optimizer_registry.h"
#include "core/graph/graph_utils.h"

namespace onnxruntime {

ConstantFoldingDQ::ConstantFoldingDQ(const IExecutionProvider& execution_provider,
                                     bool skip_dequantize_linear,
                                     const ConfigOptions& config_options,
                                     const InlinedHashSet<NodeIndex>& node_index_set,
                                     const InlinedHashSet<std::string_view>& compatible_execution_providers,
                                     const InlinedHashSet<std::string>& excluded_initializers) noexcept
    : ConstantFolding("ConstantFoldingDQ", execution_provider, skip_dequantize_linear, config_options, compatible_execution_providers, excluded_initializers),
      node_index_set_(node_index_set) {}

bool ConstantFoldingDQ::AllowConstantFolding(const Node& node) const {
  if (node_index_set_.find(node.Index()) != node_index_set_.end()) {
    return true;
  }
  return false;
}

}  // namespace onnxruntime
