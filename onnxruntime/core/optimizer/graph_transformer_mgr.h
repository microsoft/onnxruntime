// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/rewrite_rule.h"
using namespace ::onnxruntime::common;

namespace onnxruntime {

// Manages a list of graph transformers. It is initialized with a list of graph
// transformers. Each inference session can further register additional ones.
class GraphTransformerManager {
 public:
  explicit GraphTransformerManager(unsigned steps) : steps_(steps) {
  }

  // Register a transformer with a level and compatible providers list
  common::Status Register(std::unique_ptr<GraphTransformer> transformer,
                          const TransformerLevel& level,
                          const std::vector<std::string>& provider = {});

  // Apply all transformers registered for the given level on the given graph
  common::Status ApplyTransformers(Graph& graph, const TransformerLevel& level) const;

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphTransformerManager);

  const std::vector<std::string>& GetProvidersForTransformer(const std::string& name) const {
    const auto& entry = transformers_info_.find(name);
    ORT_ENFORCE(entry != transformers_info_.end());
    return entry->second.compatible_providers;
  }

  const unsigned steps_;

  struct TransformerInfo {
   public:
    TransformerInfo(TransformerLevel level, const std::vector<std::string>& providers, GraphTransformer* graphTransformer)
        : level{level}, compatible_providers{providers}, transformer{graphTransformer} {}

    TransformerInfo() = default;

    TransformerLevel level;
    std::vector<std::string> compatible_providers;
    GraphTransformer* transformer;
  };  

  // Older GCC versions don't support std::hash with enum types
  // Therefore, std::hash<T> appears to be undefined when T is an enum Type. This is fixed in version 6.1
  // TODO: remove this when we update to 6.1 or later
  struct EnumHashKey {
    template <typename T>
    size_t operator()(T t) const {
      return static_cast<size_t>(t);
    }
  };

  std::unordered_map<TransformerLevel, std::vector<std::unique_ptr<GraphTransformer>>, EnumHashKey> level_to_transformer_map_;
  std::unordered_map<std::string, TransformerInfo> transformers_info_;
};
}  // namespace onnxruntime