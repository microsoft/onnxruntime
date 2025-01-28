// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>

#include "core/optimizer/qdq_transformer/constant_folding_dq_node.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/utils.h"
#include "core/graph/graph_utils.h"
#include "core/optimizer/optimizer_execution_frame.h"
#include "core/optimizer/utils.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"

using namespace onnxruntime::common;

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

Status ConstantFoldingDQ::UpdateNodeIndexSet(InlinedHashSet<NodeIndex>& node_index_set) {
  node_index_set_ = node_index_set;
  return Status::OK();
}

}  // namespace onnxruntime
