// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <functional>
#include "orttraining/core/graph/generic_registry.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/optimizer/rule_based_graph_transformer.h"

namespace onnxruntime {
namespace training {

typedef GenericRegistry<GraphTransformer,
                        const InlinedHashSet<std::string_view>&>  // supported EP list
    GraphTransformerRegistryType;

typedef std::function<std::unique_ptr<GraphTransformer>(const InlinedHashSet<std::string_view>&)> GraphTransformerCreator;

struct GraphTransformerMeta {
  TransformerLevel level;
  bool before_gradient_builder;
};

class GraphTransformerRegistry {
 public:
  static GraphTransformerRegistry& GetInstance() {
    static GraphTransformerRegistry instance;
    return instance;
  }

  void RegisterExternalGraphTransformers();

  void Register(const std::string& name, const GraphTransformerCreator& creator, const GraphTransformerMeta& meta) {
    ORT_ENFORCE(!transformer_registry_.Contains(name), "Fail to register, the entry exists:", name);
    transformer_registry_.Register<GraphTransformer>(name, creator);
    name_to_meta_map_.insert({name, meta});
  }

  const std::unordered_map<std::string, GraphTransformerMeta>& GetAllRegisteredTransformers() {
    return name_to_meta_map_;
  }

  std::unique_ptr<GraphTransformer> CreateTransformer(const std::string& name, const InlinedHashSet<std::string_view>& ep_list) const {
    return transformer_registry_.MakeUnique(name, ep_list);
  }

 private:
  GraphTransformerRegistry() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphTransformerRegistry);

  GraphTransformerRegistryType transformer_registry_;
  std::unordered_map<std::string, GraphTransformerMeta> name_to_meta_map_;
};

class GraphTransformerRegisterOnce final {
 public:
  GraphTransformerRegisterOnce(const std::string& name, const GraphTransformerCreator& creator, TransformerLevel level, bool before_gradient_builder) {
    GraphTransformerRegistry::GetInstance().Register(name, creator, {level, before_gradient_builder});
  }
};

#define ONNX_REGISTER_EXTERNAL_GRAPH_TRANSFORMER(name, level, flag) \
  ONNX_REGISTER_EXTERNAL_GRAPH_TRANSFORMER_UNIQ(__COUNTER__, name, level, flag)
#define ONNX_REGISTER_EXTERNAL_GRAPH_TRANSFORMER_UNIQ(Counter, name, level, flag) \
  static ONNX_UNUSED onnxruntime::training::GraphTransformerRegisterOnce          \
      graph_transformer_register_once##name##Counter(                             \
          #name, [](const InlinedHashSet<std::string_view>& eps) {                \
            return std::make_unique<name>(eps);                                   \
          },                                                                      \
          TransformerLevel::level, flag);

#define ONNX_REGISTER_EXTERNAL_REWRITE_RULE(name, level, flag) \
  ONNX_REGISTER_EXTERNAL_REWRITE_RULE_UNIQ(__COUNTER__, name, level, flag)
#define ONNX_REGISTER_EXTERNAL_REWRITE_RULE_UNIQ(Counter, name, level, flag)                      \
  static ONNX_UNUSED onnxruntime::training::GraphTransformerRegisterOnce                          \
      graph_transformer_register_once##name##Counter(                                             \
          #name, [](const InlinedHashSet<std::string_view>& eps) {                                \
            auto rule_base_transformer = std::make_unique<RuleBasedGraphTransformer>(#name, eps); \
            ORT_THROW_IF_ERROR(rule_base_transformer->Register(std::make_unique<name>()));        \
            return rule_base_transformer;                                                         \
          },                                                                                      \
          TransformerLevel::level, flag);

void GenerateExternalTransformers(
    TransformerLevel level,
    bool before_gradient_builder,
    const InlinedHashSet<std::string_view>& ep_list,
    std::vector<std::unique_ptr<GraphTransformer>>& output);

}  // namespace training
}  // namespace onnxruntime
