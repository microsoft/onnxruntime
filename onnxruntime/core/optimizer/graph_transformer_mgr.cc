// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/graph_transformers_utils.h"

using namespace onnxruntime;
using namespace ::onnxruntime::common;

namespace onnxruntime {

Status GraphTransformerManager::ApplyTransformations(Graph& graph, TransformationStage stage) const {

  auto stage_pipeline_entry = stage_to_level_.find(stage);
  if (stage_pipeline_entry == stage_to_level_.end()) {
    return Status(ONNXRUNTIME, FAIL, "Specified stage not supported.");
  }

  for (unsigned step = 0; step < steps_; ++step) {
    bool changed = false;
    // apply all transformations registered for every enabled level in the stage pipeline    
    for (const auto& level : stage_pipeline_entry->second) {
      if (!(optimizations_enabled_ & level)) {
        continue;
      }        
      auto transformers_entry = transformers_map_.find(level);
      if (transformers_entry != transformers_map_.end()) {
        Status s = Apply(graph, transformers_entry->second, changed);
        if (!s.IsOK()) {
          return s;
        }            
      }
    }
    if (!changed) break;
  }
  return Status::OK();
}

Status GraphTransformerManager::ApplyTransformations(Graph& graph, const std::vector<std::string>& transformers) const {
  ORT_ENFORCE(false, "API not implemented yet");
  return Status::OK();
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

void GraphTransformerManager::InitTransformersMap(TransformersMap& transformers_map) {

  if (optimizations_enabled_ & TransformerLevel::Default_ProviderSpecific) {
    transformers_map[TransformerLevel::Default_ProviderSpecific] = std::move(GraphTransformerUtils::InitDefaultProviderSpecificTransformers());
  }
  if (optimizations_enabled_ & TransformerLevel::Optional_L1) {
    transformers_map[TransformerLevel::Optional_L1] = std::move(GraphTransformerUtils::InitL1Transformers());
  }
  if (optimizations_enabled_ & TransformerLevel::Optional_L2) {
    transformers_map[TransformerLevel::Optional_L2] = std::move(GraphTransformerUtils::InitL2Transformers());
  }
}

void GraphTransformerManager::InitStagePipeline(StageMap& stage_map) {

  stage_map[TransformationStage::PrePartition] = {TransformerLevel::Optional_L1};

  stage_map[TransformationStage::PostPartition] = {TransformerLevel::Default_ProviderSpecific,
                                                   TransformerLevel::Optional_L2, 
                                                   TransformerLevel::Optional_L1};  
}

}// namespace onnxruntime