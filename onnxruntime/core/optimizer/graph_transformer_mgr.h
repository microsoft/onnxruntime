// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
using namespace ::onnxruntime::common;
namespace onnxruntime {

enum class TransformationStage {
  PrePartition,
  PostPartition
};

// Manages graph transformation for a inference session. It is initialized with a list of graph
// transformers. Each inference session can further register additional ones.
class GraphTransformerManager {
 public:
  explicit GraphTransformerManager(unsigned steps, unsigned int level) : steps_(steps) {
    ORT_ENFORCE(ValidateOptimizationLevel(level).IsOK(),
        "Unsupported optimization level specified");

    optimizations_enabled_ |= TransformerLevel::Default_ProviderSpecific;
    if (level == 1) {
      optimizations_enabled_ |= TransformerLevel::Optional_L1;
    } else if (level == 2) {
      optimizations_enabled_ |= TransformerLevel::Optional_L1 | TransformerLevel::Optional_L2;
    }

    InitStagePipeline(stage_to_level_);

    InitTransformersMap(transformers_map_);
  }

  // Register a graph transformer.
  Status Register(std::unique_ptr<GraphTransformer> transformer, unsigned int level) {
    auto t_level = GetTransformerLevel(level);    
    transformers_map_[t_level].push_back(std::move(transformer));
    return common::Status::OK();
  }

  // Apply the list of graph transformers registered for the specified level on the specified graph
  // up to the given number of steps.
  Status ApplyTransformations(Graph& graph, const TransformationStage stage) const;

  // Apply all the transformers present in the transformers vector on the specified graph
  Status ApplyTransformations(Graph& graph, const std::vector<std::string>& transformers) const;

 private:
  using TransformersList = std::vector<std::unique_ptr<GraphTransformer>>;
  using TransformersMap = std::unordered_map<TransformerLevel, TransformersList>;
  using StageMap = std::unordered_map<TransformationStage, std::vector<TransformerLevel>>;

  GraphTransformerManager() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphTransformerManager);
  
  Status GraphTransformerManager::Apply(Graph& graph, const TransformersList& transformers, bool& graphModified) const;

  TransformerLevel GetTransformerLevel(unsigned int level) const {
    auto t_level = (unsigned int)std::pow(2, level);
    ORT_ENFORCE(t_level < TransformerLevel::MaxTransformerLevel, "Unexpected level.");
    return static_cast<TransformerLevel>(t_level);
  }

  Status ValidateOptimizationLevel(unsigned int level) const {
    auto t_level = (unsigned int)std::pow(2, level);
    if (t_level < TransformerLevel::MaxTransformerLevel) {
      return Status::OK();
    }
    return Status(ONNXRUNTIME, FAIL, "Unexpected level.");
  }

  void InitTransformersMap(TransformersMap& transformers_map);

  void InitStagePipeline(StageMap& stage_map);

  // Bit mask to hold all the enabled levels for this session
  unsigned int optimizations_enabled_ = 0;
  const unsigned steps_;
  TransformersMap transformers_map_;
  StageMap stage_to_level_;

};
}// namespace onnxruntime