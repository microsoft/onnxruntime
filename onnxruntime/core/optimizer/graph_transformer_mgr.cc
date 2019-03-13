// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/rule_based_graph_transformer.h"
using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

common::Status GraphTransformerManager::ApplyTransformers(Graph& graph, std::vector<std::string> providers, const TransformerLevel& level) const {
  const auto& transformers = level_to_transformer_map_.find(level);
  if (transformers == level_to_transformer_map_.end()) {
    return Status::OK();
  }

  // sort the providers. This makes it easy to find intersection of requested providers for this run
  // and registered providers for a transformer
  std::sort(providers.begin(), providers.end());

  for (unsigned step = 0; step < steps_; ++step) {
    bool graph_changed = false;
    for (const auto& transformer : transformers->second) {
      const auto compatible_providers = GenerateCompatibleProvidersList(providers, transformer->Name());
      bool modified = false;
      ORT_RETURN_IF_ERROR(transformer->Apply(graph, modified, compatible_providers));
      graph_changed = graph_changed || modified;
    }
    if (!graph_changed) {
      break;
    }
  }

  return Status::OK();
}

common::Status GraphTransformerManager::Register(std::unique_ptr<GraphTransformer> transformer, const TransformerLevel& level, std::vector<std::string>&& provider) {
  const auto& name = transformer->Name();
  if (transformers_info_.find(name) != transformers_info_.end()) {
    return Status(ONNXRUNTIME, FAIL, "This transformer is already registered " + name);
  }

  std::sort(provider.begin(), provider.end());
  TransformerInfo transformer_info{level, std::move(provider), transformer.get()};
  transformers_info_[name] = std::move(transformer_info);
  level_to_transformer_map_[level].push_back(std::move(transformer));
  return Status::OK();
}

}  // namespace onnxruntime
