// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "core/graph/graph.h"
#include "core/optimizer/qdq_transformer/qdq_op_transformer.h"

namespace onnxruntime {

class QDQRegistry {
 public:
  using QDQTransformerCreator = std::function<std::unique_ptr<QDQOperatorTransformer>(Node&, Graph&)>;
  static bool Register(const std::string& op_type, QDQTransformerCreator creator) {
    if (qdqtransformer_creators_.count(op_type)) {
      return false;
    }

    qdqtransformer_creators_[op_type] = creator;
    return true;
  }

  static std::unique_ptr<QDQOperatorTransformer> CreateQDQTransformer(Node& node, Graph& graph) {
    auto it = qdqtransformer_creators_.find(node.OpType());
    if (it != qdqtransformer_creators_.end())
      return (it->second)(node, graph);

    return std::unique_ptr<QDQOperatorTransformer>();
  }

 private:
  static std::unordered_map<std::string, QDQTransformerCreator> qdqtransformer_creators_;
};

#define QDQ_CREATOR_BUILDER_NAME(op_type, Transformer) Register_##op_type##_qdq_##Transformer

#define DECLARE_QDQ_CREATOR(op_type, Transformer) \
  std ::pair<std::string, QDQRegistry::QDQTransformerCreator> QDQ_CREATOR_BUILDER_NAME(op_type, Transformer)();

#define DEFINE_QDQ_CREATOR(op_type, Transformer)                                                                \
  std::pair<std::string, QDQRegistry::QDQTransformerCreator> QDQ_CREATOR_BUILDER_NAME(op_type, Transformer)() { \
    return std::pair<std::string, QDQRegistry::QDQTransformerCreator>(                                          \
        #op_type,                                                                                               \
        [](Node& node, Graph& graph) { return std::make_unique<Transformer>(node, graph); });                   \
  }

#define REGISTER_QDQ_CREATOR(op_type, Transformer) \
  QDQ_CREATOR_BUILDER_NAME(op_type, Transformer)  \
  ()
}  // namespace onnxruntime
