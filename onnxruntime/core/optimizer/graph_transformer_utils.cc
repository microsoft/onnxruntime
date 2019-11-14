// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/dropout_elimination.h"
#include "core/optimizer/relu_clip_fusion.h"
#include "core/optimizer/shape_to_initializer.h"
#include "core/optimizer/nchwc_transformer.h"
#include "core/optimizer/free_dim_override_transformer.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/mlas/inc/mlas.h"
#include "core/session/inference_session.h"

namespace onnxruntime {

namespace optimizer_utils {

std::string GenerateRuleBasedTransformerName(TransformerLevel level) {
  return "Level" + std::to_string(static_cast<uint32_t>(level)) + "_RuleBasedTransformer";
}

std::vector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(TransformerLevel level,
                                                               const std::vector<std::string>& rules_to_enable) {
  std::vector<std::unique_ptr<RewriteRule>> rules;
  switch (level) {
    case TransformerLevel::Level1:
      rules.push_back(onnxruntime::make_unique<EliminateIdentity>());
      rules.push_back(onnxruntime::make_unique<EliminateSlice>());
      rules.push_back(onnxruntime::make_unique<UnsqueezeElimination>());
      rules.push_back(onnxruntime::make_unique<EliminateDropout>());
      rules.push_back(onnxruntime::make_unique<FuseReluClip>());
      rules.push_back(onnxruntime::make_unique<ShapeToInitializer>());
      rules.push_back(onnxruntime::make_unique<ConvAddFusion>());
      rules.push_back(onnxruntime::make_unique<ConvMulFusion>());
      rules.push_back(onnxruntime::make_unique<ConvBNFusion>());
      break;

    case TransformerLevel::Level2:
      // No level2 rules available today
      break;

    case TransformerLevel::Level3:
      break;

    default:
      ORT_ENFORCE(false, "Unsupported level" + std::to_string(static_cast<uint32_t>(level)));
  }

  if (!rules_to_enable.empty()) {
    std::vector<std::unique_ptr<RewriteRule>> filtered_list;
    for (const auto& rule_name : rules_to_enable) {
      std::for_each(rules.begin(), rules.end(), [&](std::unique_ptr<RewriteRule>& item) {
        if ((item != nullptr) && (item->Name() == rule_name)) {
          filtered_list.push_back(std::move(item));
        }
      });
    }
    return filtered_list;
  }

  return rules;
}

std::unique_ptr<RuleBasedGraphTransformer> GenerateRuleBasedGraphTransformer(TransformerLevel level,
                                                                             const std::vector<std::string>& rules_to_enable,
                                                                             const std::unordered_set<std::string>& compatible_execution_providers) {
  auto rewrite_rules_to_register = GenerateRewriteRules(level, rules_to_enable);
  if (rewrite_rules_to_register.empty()) {
    return nullptr;
  }

  std::unique_ptr<RuleBasedGraphTransformer> rule_transformer =
      onnxruntime::make_unique<RuleBasedGraphTransformer>(GenerateRuleBasedTransformerName(level),
                                                          compatible_execution_providers);
  for (auto& entry : rewrite_rules_to_register) {
    rule_transformer->Register(std::move(entry));
  }

  return rule_transformer;
}

std::vector<std::unique_ptr<GraphTransformer>> GenerateTransformers(TransformerLevel level,
                                                                    gsl::span<const FreeDimensionOverride> free_dimension_overrides,
                                                                    const std::vector<std::string>& transformers_and_rules_to_enable) {
  std::vector<std::unique_ptr<GraphTransformer>> transformers;
  std::unique_ptr<RuleBasedGraphTransformer> rule_transformer = nullptr;
  switch (level) {
    case TransformerLevel::Level1: {
      std::unordered_set<std::string> l1_execution_providers = {};

      transformers.emplace_back(onnxruntime::make_unique<ConstantFolding>(l1_execution_providers));
      transformers.emplace_back(onnxruntime::make_unique<MatMulAddFusion>(l1_execution_providers));
      transformers.emplace_back(onnxruntime::make_unique<ReshapeFusion>(l1_execution_providers));
      transformers.emplace_back(onnxruntime::make_unique<FreeDimensionOverrideTransformer>(free_dimension_overrides));

      rule_transformer = GenerateRuleBasedGraphTransformer(level, transformers_and_rules_to_enable, l1_execution_providers);
    } break;

    case TransformerLevel::Level2: {
      std::unordered_set<std::string> l2_execution_providers = {onnxruntime::kCpuExecutionProvider};

      // create rule based transformer consisting of all the level2 rewrite rules
      rule_transformer = GenerateRuleBasedGraphTransformer(level, transformers_and_rules_to_enable, l2_execution_providers);

      // create standalone transformers
#ifndef DISABLE_CONTRIB_OPS
      transformers.emplace_back(onnxruntime::make_unique<GemmActivationFusion>(l2_execution_providers));
      transformers.emplace_back(onnxruntime::make_unique<ConvActivationFusion>(l2_execution_providers));
      transformers.emplace_back(onnxruntime::make_unique<GeluFusion>(l2_execution_providers));
#endif
    } break;

    case TransformerLevel::Level3: {
#ifndef DISABLE_CONTRIB_OPS
      // Register the NCHWc layout transformer if supported by the platform.
      if (MlasNchwcGetBlockSize() > 1) {
        transformers.emplace_back(onnxruntime::make_unique<NchwcTransformer>());
      }
#endif

    } break;

    default:
      ORT_ENFORCE(false, "Unsupported level " + std::to_string(static_cast<uint32_t>(level)));
      break;
  }

  // if the custom list to enable transformers\rules is empty then return the default generated transformers and rules
  // otherwise generate a filtered list based on the provided custom list.
  if (transformers_and_rules_to_enable.empty()) {
    if (rule_transformer != nullptr) {
      transformers.emplace_back(std::move(rule_transformer));
    }
    return transformers;
  }
  std::vector<std::unique_ptr<GraphTransformer>> filtered_list;
  // If the rule-based transformer is not empty, it should be included in the custom transformer list below.
  if (rule_transformer != nullptr) {
    filtered_list.emplace_back(std::move(rule_transformer));
  }
  // pick custom transformers enabled for this session
  for (const auto& t_name : transformers_and_rules_to_enable) {
    std::for_each(transformers.begin(), transformers.end(),
                  [&](std::unique_ptr<GraphTransformer>& item) {
                    if ((item != nullptr) && (item->Name() == t_name)) {
                      filtered_list.push_back(std::move(item));
                    }
                  });
  }
  return filtered_list;
}

}  // namespace optimizer_utils
}  // namespace onnxruntime
