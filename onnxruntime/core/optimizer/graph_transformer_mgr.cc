// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/graph_transformer_factory.h"

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

Status GraphTransformerManager::ApplyTransformations(Graph& graph, const std::vector<TransformerLevel>& levels) const {  
  for (unsigned step = 0; step < steps_; ++step) {
    bool changed = false;
    // apply all transformations registered for every enabled level in the levels list
    for (const auto& level : levels) {
      if (!(optimizations_enabled_ & level)) {
        continue;
      }

      const auto& transformers = factory_->GetTransformers(level);
      ORT_ENFORCE(Apply(graph, transformers, changed).IsOK());
    }

    if (!changed) break;
  }

  return Status::OK();
}

Status GraphTransformerManager::ApplyTransformations(Graph& graph, const std::vector<std::string>& transformer_ids) const {  
  const auto& transformers = factory_->GetTransformers(transformer_ids);
  for (unsigned step = 0; step < steps_; ++step) {
    bool changed = false;    
    ORT_ENFORCE(Apply(graph, transformers, changed).IsOK());   

    if (!changed) break;
  }
  return Status::OK();
}

Status GraphTransformerManager::ApplyTransformations(Graph& graph, const GraphTransformer* transformer) const {
  bool modified = false;
  return transformer->Apply(graph, modified);
}

Status GraphTransformerManager::Apply(Graph& graph, const TransformersList& transformers, bool& graphModified) const {
  for (const auto& transformer : transformers) {
    bool modified = false;
    Status s = transformer->Apply(graph, modified);
    if (!s.IsOK()) {
      return s;
    }
    graphModified = graphModified || modified;
  }
  return Status::OK();
}

}  // namespace onnxruntime