// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/rule_based_graph_transformer.h"
using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

common::Status GraphTransformerManager::ApplyTransformers(Graph& graph, const TransformerLevel& level) const {
  const auto& transformers = level_to_transformer_map_.find(level);
  if (transformers == level_to_transformer_map_.end()) {
    return Status::OK();
  }

  for (unsigned step = 0; step < steps_; ++step) {
    bool graph_changed = false;
    for (const auto& transformer : transformers->second) {
      bool modified = false;
      ORT_RETURN_IF_ERROR(transformer->Apply(graph, modified, 
                                             GetProvidersForTransformer(transformer->Name())));
      graph_changed = graph_changed || modified;
    }
    if (!graph_changed) {
      break;
    }
  }

  return Status::OK();
}

common::Status GraphTransformerManager::Register(std::unique_ptr<GraphTransformer> transformer, 
                                                 const TransformerLevel& level, 
                                                 const std::vector<std::string>& providers) {
  const auto& name = transformer->Name();
  if (transformers_info_.find(name) != transformers_info_.end()) {
    return Status(ONNXRUNTIME, FAIL, "This transformer is already registered " + name);
  }

  TransformerInfo transformer_info{level, providers, transformer.get()};
  transformers_info_[name] = std::move(transformer_info);
  level_to_transformer_map_[level].push_back(std::move(transformer));
  return Status::OK();
}

}  // namespace onnxruntime
