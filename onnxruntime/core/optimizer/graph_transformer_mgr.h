// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_factory.h"
using namespace ::onnxruntime::common;
namespace onnxruntime {

// Manages graph transformation for a inference session. It is initialized with a list of pre-defined graph
// transformers. Each inference session can further register additional ones.
class GraphTransformerManager {
 public:
  explicit GraphTransformerManager(unsigned steps, unsigned int level) : steps_(steps), factory_{std::make_unique<GraphTransformerFactory>()} {
    ORT_ENFORCE(ValidateOptimizationLevel(level).IsOK(),
                "Unsupported optimization level specified");

    optimizations_enabled_ |= TransformerLevel::Default_ProviderSpecific;
    if (level == 1) {
      optimizations_enabled_ |= TransformerLevel::Optional_L1;
    } else if (level == 2) {
      optimizations_enabled_ |= TransformerLevel::Optional_L1 | TransformerLevel::Optional_L2;
    }    
  }

  // Registers a graph transformer.
  Status Register(std::unique_ptr<GraphTransformer> transformer) {
      return factory_->Register(std::move(transformer));
  }

  // Apply the list of graph transformers registered for the specified level on the specified graph
  // up to the given number of steps.
  Status ApplyTransformations(Graph& graph, const std::vector<TransformerLevel>& levels) const;
  
  Status ApplyTransformations(Graph& graph, const std::vector<std::string>& transformer_ids) const;

  Status ApplyTransformations(Graph& graph, const GraphTransformer* transformer) const;

 private:
  using TransformersList = std::vector<GraphTransformer*>;  

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphTransformerManager);

  Status Apply(Graph& graph, const TransformersList& transformers, bool& graphModified) const;

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

  // Bit mask to hold all the enabled levels for this session
  unsigned int optimizations_enabled_ = 0;
  const unsigned steps_;
  std::unique_ptr<GraphTransformerFactory> factory_;  
};
}  // namespace onnxruntime