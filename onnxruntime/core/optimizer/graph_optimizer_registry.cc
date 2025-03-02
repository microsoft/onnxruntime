// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_optimizer_registry.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/selection_and_optimization_func.h"
#include "core/optimizer/qdq_transformer/constant_folding_dq_node.h"

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {
GraphOptimizerRegistry::GraphOptimizerRegistry(const onnxruntime::SessionOptions* sess_options,
                                               const onnxruntime::IExecutionProvider* cpu_ep,
                                               const logging::Logger* logger) : session_options_(sess_options),
                                                                                cpu_ep_(cpu_ep),
                                                                                logger_(logger) {}

Status GraphOptimizerRegistry::CreatePredefinedSelectionFuncs() {
  transformer_name_to_selection_func_[kConstantFoldingDQ] = ConstantFoldingDQFuncs::Select;

  return Status::OK();
}

// Create and initialize the graph optimizer registry instance as a singleton.
Status GraphOptimizerRegistry::Create(const onnxruntime::SessionOptions* sess_options,
                                      const onnxruntime::IExecutionProvider* cpu_ep,
                                      const logging::Logger* logger) {
  if (!graph_optimizer_registry_) {  // First Check (without locking)
    std::lock_guard<std::mutex> lock(registry_mutex_);
    if (!graph_optimizer_registry_) {  // Second Check (with locking)
      graph_optimizer_registry_ = std::unique_ptr<GraphOptimizerRegistry>(new GraphOptimizerRegistry(sess_options, cpu_ep, logger));
      ORT_RETURN_IF_ERROR(graph_optimizer_registry_->CreatePredefinedSelectionFuncs());
    }
  } else {
    LOGS(*graph_optimizer_registry_->GetLogger(), INFO) << "The GraphOptimizerRegistry instance has been created before.";
  }

  return Status::OK();
}

std::optional<SelectionFunc> GraphOptimizerRegistry::GetSelectionFunc(std::string& name) const {
  auto lookup = transformer_name_to_selection_func_.find(name);
  if (lookup != transformer_name_to_selection_func_.end()) {
    return transformer_name_to_selection_func_.at(name);
  }
  LOGS(*logger_, WARNING) << "Can't find selection function of " << name;
  return std::nullopt;
}

// Initialize static members
std::unique_ptr<GraphOptimizerRegistry> onnxruntime::GraphOptimizerRegistry::graph_optimizer_registry_ = nullptr;
std::mutex GraphOptimizerRegistry::registry_mutex_;
}  // namespace onnxruntime
