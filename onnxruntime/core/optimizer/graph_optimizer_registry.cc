// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_optimizer_registry.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/selection_and_optimization_func.h"
#include "core/optimizer/qdq_transformer/constant_folding_dq_node.h"

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

GraphOptimizerRegistry::GraphOptimizerRegistry() {}

std::optional<SelectionFunc> GraphOptimizerRegistry::GetSelectionFunc(std::string& name) const {
  auto lookup = transformer_name_to_selection_func_.find(name);
  if (lookup != transformer_name_to_selection_func_.end()) {
    return transformer_name_to_selection_func_.at(name);
  }
  LOGS(*logger_, WARNING) << "Can't find selection function of " << name;
  return std::nullopt;
}

common::Status GraphOptimizerRegistry::Create(
    const onnxruntime::SessionOptions* sess_options,
    const onnxruntime::IExecutionProvider* cpu_ep,
    const logging::Logger* logger) {
  session_options_ = sess_options;
  cpu_ep_ = cpu_ep;
  logger_ = logger;

  // Add predefined transformer names and their selection functions
  transformer_name_to_selection_func_[kConstantFoldingDQ] = ConstantFoldingDQFuncs::Select;

  return Status::OK();
}

// Initialize static members
std::shared_ptr<GraphOptimizerRegistry> onnxruntime::GraphOptimizerRegistry::graph_optimizer_registry = nullptr;
std::mutex GraphOptimizerRegistry::registry_mutex;
}  // namespace onnxruntime
