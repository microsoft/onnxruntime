// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_optimizer_registry.h"
#include "core/optimizer/graph_transformer_utils.h"

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

common::Status GraphOptimizerRegistry::Register(std::unique_ptr<GraphTransformer> transformer) {
  const auto& name = transformer->Name();
  if (name_to_transformer_map_.find(name) != name_to_transformer_map_.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "This transformer is already registered " + name);
  }

  name_to_transformer_map_[name] = transformer.get();
  transformer_list_.push_back(std::move(transformer));
  return Status::OK();
}

GraphTransformer* GraphOptimizerRegistry::GetTransformerByName(std::string& name) const {
  if (name_to_transformer_map_.find(name) != name_to_transformer_map_.end()) {
    return name_to_transformer_map_.at(name);
  }
  return nullptr;
}

// Registers all the predefined transformers for EP
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

// Initialize static members
std::shared_ptr<GraphOptimizerRegistry> onnxruntime::GraphOptimizerRegistry::graph_optimizer_registry = nullptr;
std::mutex GraphOptimizerRegistry::registry_mutex;
}  // namespace onnxruntime
