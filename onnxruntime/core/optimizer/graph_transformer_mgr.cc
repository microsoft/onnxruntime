// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/rule_based_graph_transformer.h"

#include <memory>
#include <string>
#include <utility>

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

common::Status GraphTransformerManager::SetSteps(unsigned steps) {
  steps_ = steps;
  return Status::OK();
}

common::Status GraphTransformerManager::GetSteps(unsigned& steps) const {
  steps = steps_;
  return Status::OK();
}

common::Status GraphTransformerManager::ApplyTransformers(Graph& graph, TransformerLevel level,
                                                          const logging::Logger& logger) const {
  _is_graph_modified = false;
  const auto& transformers = level_to_transformer_map_.find(level);
  if (transformers == level_to_transformer_map_.end()) {
    LOGS(logger, VERBOSE) << "No graph transformers registered for level " << static_cast<int>(level) << ".";
    return Status::OK();
  }

  LOGS(logger, VERBOSE) << "Applying " << transformers->second.size() << " graph transformer(s) for level "
                        << static_cast<int>(level) << " for up to " << steps_ << " step(s).";

  for (unsigned step = 0; step < steps_; ++step) {
    if (IsLoadCancellationFlagSet()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, MODEL_LOAD_CANCELED, "Graph transformation canceled due to user request.");
    }

    LOGS(logger, VERBOSE) << "Graph transformer step " << (step + 1) << " of " << steps_
                          << " for level " << static_cast<int>(level) << " started.";

    bool graph_changed = false;
    for (const auto& transformer : transformers->second) {
      if (step > 0 && transformer->ShouldOnlyApplyOnce()) {
        LOGS(logger, VERBOSE) << "Skipping graph transformer " << transformer->Name()
                              << " on step " << (step + 1) << " because it should only apply once.";
        continue;
      }

      bool modified = false;
      LOGS(logger, VERBOSE) << "Applying graph transformer " << transformer->Name()
                            << " on step " << (step + 1) << ".";
      ORT_RETURN_IF_ERROR(transformer->Apply(graph, modified, logger));
      graph_changed = graph_changed || modified;
      _is_graph_modified = _is_graph_modified || modified;
    }

    LOGS(logger, VERBOSE) << "Graph transformer step " << (step + 1) << " of " << steps_
                          << " for level " << static_cast<int>(level)
                          << " completed. graph_changed: " << graph_changed << ".";

    if (!graph_changed) {
      LOGS(logger, VERBOSE) << "Stopping graph transformer iteration for level " << static_cast<int>(level)
                            << " after step " << (step + 1) << " because the graph was not modified.";
      break;
    }
  }

  return Status::OK();
}

const bool& GraphTransformerManager::IsGraphModified(void) const {
  return _is_graph_modified;
}

void GraphTransformerManager::ClearGraphModified(void) {
  _is_graph_modified = false;
}

common::Status GraphTransformerManager::Register(std::unique_ptr<GraphTransformer> transformer,
                                                 TransformerLevel level) {
  const auto& name = transformer->Name();
  // Reject a duplicate transformer name at the same level. Keying transformers_info_ by (level, name)
  // makes this an O(1) lookup while preserving the per-level semantics: the same name may still be
  // registered at different levels (e.g. LayerNormFusion at both Level1 and Level2). The original
  // std::find compared the incoming unique_ptr against the stored ones, which could never match (it owns
  // a distinct, not-yet-inserted pointer), so duplicates slipped through and were applied twice per level.
  auto level_scoped_name = std::to_string(static_cast<int>(level)) + ":" + name;
  if (!transformers_info_.emplace(std::move(level_scoped_name), transformer.get()).second) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "This transformer is already registered " + name);
  }

  level_to_transformer_map_[level].push_back(std::move(transformer));
  return Status::OK();
}
}  // namespace onnxruntime
