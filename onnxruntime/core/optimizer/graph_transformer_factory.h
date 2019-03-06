// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/insert_cast_transformer.h"

using namespace ::onnxruntime::common;

namespace onnxruntime {
/*
* Owns all the graph transformers.
*/
class GraphTransformerFactory {
 public:
  explicit GraphTransformerFactory() {
    InitFactory();
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GraphTransformerFactory);

  std::vector<GraphTransformer*> GetTransformers(TransformerLevel level) const {
    std::vector<GraphTransformer*> transformers;
    const auto entry = level_to_string_.find(level);
    if (entry != level_to_string_.end()) {
      for (const auto& id : entry->second) {
        const auto transformer = GetTransformers(id);
        if (transformer != nullptr) {
          transformers.push_back(transformer);
        }
      }
    }
    return transformers;
  }

  GraphTransformer* GetTransformers(std::string id) const {
    const auto& entry = transformers_map_.find(id);
    if (entry != transformers_map_.end()) {
      return entry->second.get();
    }

    return nullptr;
  }

  std::vector<GraphTransformer*> GetTransformers(const std::vector<std::string>& ids) const {
    std::vector<GraphTransformer*> transformers;
    for (const auto& id : ids) {
      const auto transformer = GetTransformers(id);
      if (transformer != nullptr) {
        transformers.push_back(transformer);
      }
    }
    return transformers;
  }

  Status Register(std::unique_ptr<GraphTransformer> transformer) {
    auto name = transformer->Name();
    auto level = transformer->Level();

    if (transformers_map_.find(name) != transformers_map_.end()) {
      return Status(ONNXRUNTIME, FAIL, "This transformer is already registered");
    }

    transformers_map_[name] = std::move(transformer);
    level_to_string_[level].push_back(name);
    return Status::OK();
  }

 private:
  void InitFactory() {    
    Register(std::make_unique<ConvAddFusion>());
    Register(std::make_unique<ConvMulFusion>());
    Register(std::make_unique<ConvBNFusion>());    
    Register(std::make_unique<UnsqueezeElimination>());
    // TODO: ConvActivationFusion, GemmActivationFusion and MatMulAddFusion needs a fix. Enable this after bug fix
    //Register(std::make_unique<ConvActivationFusion>());
    //Register(std::make_unique<GemmActivationFusion>());    
    //Register(std::make_unique<MatMulAddFusion>());

    // Create node elimination transformer
    Register(CreateRuleBasedTransformer());
  }

  std::unique_ptr<GraphTransformer> CreateRuleBasedTransformer() {
    std::string transformer_name = "NodeEliminations";
    auto level = TransformerLevel::Optional_L1;
    auto graph_rewrite_rules = std::make_unique<TopDownRuleBasedTransformer>(
        transformer_name, "Top down transformer", level,
        std::vector<std::string>{onnxruntime::kCpuExecutionProvider});

    graph_rewrite_rules->Register("Identity", std::make_unique<EliminateIdentity>());
    graph_rewrite_rules->Register("Slice", std::make_unique<EliminateSlice>());

    return graph_rewrite_rules;
  }

  std::unordered_map<std::string, std::unique_ptr<GraphTransformer>> transformers_map_;
  std::unordered_map<TransformerLevel, std::vector<std::string>> level_to_string_;
};
}  // namespace onnxruntime
