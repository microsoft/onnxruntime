// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_optimizer_registry.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/selection_and_optimization_func.h"
#include "core/optimizer/qdq_transformer/constant_folding_dq_node.h"

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

GraphOptimizerRegistry::GraphOptimizerRegistry() {
  logger_ = &logging::LoggingManager::DefaultLogger();
}

common::Status GraphOptimizerRegistry::Register(std::unique_ptr<GraphTransformer> transformer) {
  const auto& name = transformer->Name();
  if (name_to_transformer_map_.find(name) != name_to_transformer_map_.end() &&
      name_to_transformer_map_.at(name)) {
    LOGS(*logger_, WARNING) << "This optimizer is already created and registered " << name;
    return Status::OK();
  }

  name_to_transformer_map_[name] = transformer.get();
  transformer_list_.push_back(std::move(transformer));

  if (name == kCONSTANT_FOLDING_DQ) {
    transformer_name_to_selection_func_[name] = ConstantFoldingDQ_selection;
  }

  return Status::OK();
}

common::Status GraphOptimizerRegistry::AddPredefinedOptimizerNames(std::vector<std::string>& optimizer_names) {
  for (auto name : optimizer_names) {
    if (name_to_transformer_map_.find(name) != name_to_transformer_map_.end()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "This transformer name is already added " + name);
    }
    name_to_transformer_map_[name] = nullptr;  // The transformer will be instantizted only when EP requests it
  }
  return Status::OK();
}

std::unique_ptr<GraphTransformer> GraphOptimizerRegistry::CreateOptimizer(std::string& name, std::unordered_map<std::string, std::string>& key_value_configs) {
  std::unique_ptr<GraphTransformer> transformer;
  if (name == kCONSTANT_FOLDING_DQ) {
    const InlinedHashSet<NodeIndex> node_index_set = {};
    return std::make_unique<ConstantFoldingDQ>(*cpu_ep_, false /*skip_dequantize_linear*/,
                                               session_options_->config_options, node_index_set);
  }
  LOGS(*logger_, WARNING) << "Can't create optimizer " << name;
  return transformer;
}

std::optional<std::function<std::vector<std::unique_ptr<ComputeCapability>>(const GraphViewer&)>> GraphOptimizerRegistry::GetSelectionFunc(std::string& name,
                                                                                                                                           std::unordered_map<std::string, std::string>& key_value_configs) const {
  if (name_to_transformer_map_.find(name) == name_to_transformer_map_.end()) {
    LOGS(*logger_, WARNING) << "Can't find optimizer " << name;
    return std::nullopt;
  }

  // Create and register if the transformer instance is not created.
  if (!name_to_transformer_map_.at(name)) {
    auto new_transformer = Get()->CreateOptimizer(name, key_value_configs);
    Get()->Register(std::move(new_transformer));
  }

  auto lookup = transformer_name_to_selection_func_.find(name);
  if (lookup != transformer_name_to_selection_func_.end()) {
    return transformer_name_to_selection_func_.at(name);
  }
  return std::nullopt;
}

GraphTransformer* GraphOptimizerRegistry::GetTransformerByName(std::string& name) const {
  if (name_to_transformer_map_.find(name) != name_to_transformer_map_.end()) {
    return name_to_transformer_map_.at(name);
  }
  return nullptr;
}

// Create and register all the predefined transformers for EP
common::Status GraphOptimizerRegistry::AddPredefinedOptimizers(
    const onnxruntime::SessionOptions& sess_options,
    const onnxruntime::IExecutionProvider& cpu_ep,
    const logging::Logger& logger) {
  // TODO: Apply optimization level here if we later decide to do so
  auto transformers_to_register = [&]() {
    return optimizer_utils::GenerateTransformersForEP(sess_options, cpu_ep, logger);
  }();

  for (auto& entry : transformers_to_register) {
    ORT_RETURN_IF_ERROR(Get()->Register(std::move(entry)));
  }
  return Status::OK();
}

common::Status GraphOptimizerRegistry::ApplyTransformer(Graph& graph, std::string& name,
                                                         const logging::Logger& logger) const {
  auto transformer = GetTransformerByName(name);
  if (!transformer) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "This transformer is not registered " + name);
  }

  bool modified = false;
  ORT_RETURN_IF_ERROR(transformer->Apply(graph, modified, logger));

  return Status::OK();
}

common::Status GraphOptimizerRegistry::AddCpuEpReference(onnxruntime::IExecutionProvider* cpu_ep) {
  cpu_ep_ = cpu_ep;
  return Status::OK();
}

common::Status GraphOptimizerRegistry::AddSessionOptionsReference(onnxruntime::SessionOptions* session_options) {
  session_options_ = session_options;
  return Status::OK();
}

// Initialize static members
std::shared_ptr<GraphOptimizerRegistry> onnxruntime::GraphOptimizerRegistry::graph_optimizer_registry = nullptr;
std::mutex GraphOptimizerRegistry::registry_mutex;
}  // namespace onnxruntime
