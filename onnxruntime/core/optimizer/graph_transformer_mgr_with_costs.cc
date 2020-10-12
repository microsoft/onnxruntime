// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_mgr_with_costs.h"

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

common::Status GraphTransformerManagerWithCosts::SetSteps(unsigned steps) {
  steps_ = steps;
  return Status::OK();
}

common::Status GraphTransformerManagerWithCosts::GetSteps(unsigned& steps) const {
  steps = steps_;
  return Status::OK();
}

common::Status GraphTransformerManagerWithCosts::ApplyTransformers(Graph& graph, TransformerLevel level, const logging::Logger& logger) const {
  LOGS(logger, WARNING) << "Applying transformers with cost model";

  const auto& transformers = level_to_transformer_map_.find(level);
  if (transformers == level_to_transformer_map_.end()) {
    return Status::OK();
  }

  LOGS(logger, WARNING) << "Found " << transformers->second.size() << " possible transformers to apply";

  // TODO: Find the optimal sequence of transformations
  for (unsigned step = 0; step < steps_; ++step) {
    bool graph_changed = false;
    for (const auto& transformer : transformers->second) {
      if (step > 0 && transformer->ShouldOnlyApplyOnce())
        continue;

      bool modified = false;
      ORT_RETURN_IF_ERROR(transformer->Apply(graph, modified, logger));
      graph_changed = graph_changed || modified;
    }
    if (!graph_changed) {
      break;
    }
  }

  return Status::OK();
}

common::Status GraphTransformerManagerWithCosts::Register(std::unique_ptr<GraphTransformer> transformer, TransformerLevel level) {
  const auto& name = transformer->Name();
  if (transformers_info_.find(name) != transformers_info_.end()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "This transformer is already registered " + name);
  }

  transformers_info_[name] = transformer.get();
  level_to_transformer_map_[level].push_back(std::move(transformer));
  return Status::OK();

  return Status::OK();
}

} // namespace onnxruntime
