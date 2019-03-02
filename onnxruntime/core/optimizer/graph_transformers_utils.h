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

namespace onnxruntime {

class GraphTransformerUtils {
 public:
  static std::vector<std::unique_ptr<GraphTransformer>> InitL2Transformers() {
    std::vector<std::unique_ptr<GraphTransformer>> transformers;

    transformers.push_back(std::move(std::make_unique<ConvActivationFusion>()));
    transformers.push_back(std::move(std::make_unique<ConvAddFusion>()));
    transformers.push_back(std::move(std::make_unique<ConvBNFusion>()));
    transformers.push_back(std::move(std::make_unique<ConvMulFusion>()));
    transformers.push_back(std::move(std::make_unique<GemmActivationFusion>()));
    transformers.push_back(std::move(std::make_unique<MatMulAddFusion>()));
    
    return transformers;
  }

  static std::vector<std::unique_ptr<GraphTransformer>> InitL1Transformers() {
    std::vector<std::unique_ptr<GraphTransformer>> transformers;

    auto graph_rewrite_rules = std::make_unique<TopDownRuleBasedTransformer>(
        "RuleTransformer", "Top down transformer", TransformerLevel::Optional_L1, 
        std::vector<std::string>{onnxruntime::kCpuExecutionProvider});
    graph_rewrite_rules->Register("Identity", std::make_unique<EliminateIdentity>());
    graph_rewrite_rules->Register("Slice", std::make_unique<EliminateSlice>());

    transformers.push_back(std::move(graph_rewrite_rules));
    transformers.push_back(std::move(std::make_unique<UnsqueezeElimination>()));

    return transformers;
  }

  static std::vector<std::unique_ptr<GraphTransformer>> InitDefaultProviderSpecificTransformers() {
    std::vector<std::unique_ptr<GraphTransformer>> transformers;
    // Add default provider specific transformations
    return transformers;
  }
};
}  // namespace onnxruntime
