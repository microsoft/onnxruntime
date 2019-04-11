
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

namespace onnxruntime {

namespace transformer_utils {

std::vector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(TransformerLevel level,
                                                               const std::vector<std::string>* rules_to_enable) {
  std::vector<std::unique_ptr<RewriteRule>> rules;
  switch (level) {
    case TransformerLevel::Level1:
      rules.push_back(std::make_unique<EliminateIdentity>());
      rules.push_back(std::make_unique<EliminateSlice>());
      rules.push_back(std::make_unique<ConstantFolding>());
      break;

    case TransformerLevel::Level2:
      break;
    default:
      ORT_ENFORCE(false, "Unsupported level" + std::to_string(static_cast<uint32_t>(level)));
  }

  if (rules_to_enable != nullptr && !rules_to_enable->empty()) {
    std::vector<std::unique_ptr<RewriteRule>> filtered_list;
    for (const auto& rule_name : *rules_to_enable) {
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
                                                                             const std::vector<std::string>* rules_to_enable,
                                                                             const std::unordered_set<std::string>& compatible_execution_providers) {
  auto rewrite_rules_to_register = transformer_utils::GenerateRewriteRules(level, rules_to_enable);
  if (rewrite_rules_to_register.empty()) {
    return std::unique_ptr<RuleBasedGraphTransformer>{};
  }

  std::unique_ptr<RuleBasedGraphTransformer> rule_transformer =
      std::make_unique<RuleBasedGraphTransformer>(transformer_utils::GenerateRuleBasedTransformerName(level),
                                                  "Apply rewrite rules for Level" +
                                                      std::to_string(static_cast<uint32_t>(level)),
                                                  compatible_execution_providers);
  for (auto& entry : rewrite_rules_to_register) {
    rule_transformer->Register(std::move(entry));
  }

  return rule_transformer;
}

std::vector<std::unique_ptr<GraphTransformer>> GenerateTransformers(TransformerLevel level,
                                                                    std::vector<std::string>* transformers_and_rules_to_enable) {
  std::vector<std::unique_ptr<GraphTransformer>> transformers;

  // Generate rule-based transformer.
  bool non_empty_rule_transformer = false;

  switch (level) {
    case TransformerLevel::Level1: {
      std::unordered_set<std::string> l1_execution_providers = {};
      std::unique_ptr<RuleBasedGraphTransformer> rule_transformer =
          transformer_utils::GenerateRuleBasedGraphTransformer(level, transformers_and_rules_to_enable, l1_execution_providers);
      if (rule_transformer) {
        transformers.emplace_back(std::move(rule_transformer));
        non_empty_rule_transformer = true;
      }

      // At the moment, we have only a rule-based transformer for Level1.
    } break;

    case TransformerLevel::Level2: {
      std::unordered_set<std::string> l2_execution_providers = {onnxruntime::kCpuExecutionProvider};
      std::unique_ptr<RuleBasedGraphTransformer> rule_transformer =
          transformer_utils::GenerateRuleBasedGraphTransformer(level, transformers_and_rules_to_enable, l2_execution_providers);
      if (rule_transformer) {
        transformers.emplace_back(std::move(rule_transformer));
        non_empty_rule_transformer = true;
      }
      transformers.emplace_back(std::make_unique<GemmActivationFusion>(l2_execution_providers));
      transformers.emplace_back(std::make_unique<MatMulAddFusion>(l2_execution_providers));
      transformers.emplace_back(std::make_unique<ConvActivationFusion>(l2_execution_providers));
      transformers.emplace_back(std::make_unique<ConvAddFusion>());
      transformers.emplace_back(std::make_unique<ConvMulFusion>());
      transformers.emplace_back(std::make_unique<ConvBNFusion>());
    } break;

    default:
      ORT_ENFORCE(false, "Unsupported level " + std::to_string(static_cast<uint32_t>(level)));
      break;
  }

  // If the rule-based transformer is not empty, it should be included in the custom transformer list below.
  if (non_empty_rule_transformer) {
    transformers_and_rules_to_enable->push_back(transformer_utils::GenerateRuleBasedTransformerName(level));
  }
  if (transformers_and_rules_to_enable != nullptr && !transformers_and_rules_to_enable->empty()) {
    // pick custom transformers enabled for this session
    std::vector<std::unique_ptr<GraphTransformer>> filtered_list;
    for (const auto& t_name : *transformers_and_rules_to_enable) {
      std::for_each(transformers.begin(), transformers.end(),
                    [&](std::unique_ptr<GraphTransformer>& item) {
                      if ((item != nullptr) && (item->Name() == t_name)) {
                        filtered_list.push_back(std::move(item));
                      }
                    });
    }
    return filtered_list;
  }

  return transformers;
}

std::string GenerateRuleBasedTransformerName(TransformerLevel level) {
  return "Level" + std::to_string(static_cast<uint32_t>(level)) + "_RuleBasedTransformer";
}

}  // namespace transformer_utils
}  // namespace onnxruntime
