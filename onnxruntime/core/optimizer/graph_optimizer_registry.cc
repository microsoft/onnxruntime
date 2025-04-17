// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_optimizer_registry.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/selection_and_optimization_func.h"
#include "core/optimizer/qdq_transformer/constant_folding_dq_node.h"

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {
#if !defined(ORT_MINIMAL_BUILD)
GraphOptimizerRegistry::GraphOptimizerRegistry(const onnxruntime::SessionOptions* sess_options,
                                               const onnxruntime::IExecutionProvider* cpu_ep,
                                               const logging::Logger* logger) : session_options_(sess_options),
                                                                                cpu_ep_(cpu_ep),
                                                                                logger_(logger) {
  auto status = CreatePredefinedSelectionFuncs();
  ORT_ENFORCE(status.IsOK(), "Could not create pre-defined selection functions. Error Message: ",
              status.ErrorMessage());
}

Status GraphOptimizerRegistry::CreatePredefinedSelectionFuncs() {
  transformer_name_to_selection_func_[kConstantFoldingDQ] = ConstantFoldingDQFuncs::Select;

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
#else
GraphOptimizerRegistry::GraphOptimizerRegistry(const onnxruntime::SessionOptions* sess_options,
                                               const onnxruntime::IExecutionProvider* cpu_ep,
                                               const logging::Logger* logger) : session_options_(sess_options),
                                                                                cpu_ep_(cpu_ep),
                                                                                logger_(logger) {}

std::optional<SelectionFunc> GraphOptimizerRegistry::GetSelectionFunc(std::string& /*name*/) const {
  return std::nullopt;
}
#endif
}  // namespace onnxruntime
