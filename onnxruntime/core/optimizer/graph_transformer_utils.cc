
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/unsqueeze_elimination.h"

namespace onnxruntime {

namespace transformer_utils {

std::vector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(const TransformerLevel& level, 
                                                               const std::vector<std::string>* rules_to_enable) {
  std::vector<std::unique_ptr<RewriteRule>> rules;
  switch (level) {
    case TransformerLevel::Level1:
      rules.push_back(std::make_unique<EliminateIdentity>());
      rules.push_back(std::make_unique<EliminateSlice>());
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
        if((item != nullptr) && (item->Name() == rule_name)) {
          filtered_list.push_back(std::move(item));
        }
      });
    }
    return filtered_list;
  }

  return rules;
}

std::vector<TransformerProviderSet> GenerateTransformers(const TransformerLevel& level, 
                                                         const std::vector<std::string>* transformers_to_enable) {
  std::vector<TransformerProviderSet> transformers;
  switch (level) {
    case TransformerLevel::Level1: {
      std::vector<std::string> l1_execution_providers = {};
      transformers.emplace_back(std::make_unique<UnsqueezeElimination>(), l1_execution_providers);
    } break;

    case TransformerLevel::Level2: {
      std::vector<std::string> l2_execution_providers = {onnxruntime::kCpuExecutionProvider};
      transformers.emplace_back(std::make_unique<ConvAddFusion>(), l2_execution_providers);
      transformers.emplace_back(std::make_unique<ConvMulFusion>(), l2_execution_providers);
    } break;

    default:
      ORT_ENFORCE(false, "Unsupported level " + std::to_string(static_cast<uint32_t>(level)));
      break;
  }

  if (transformers_to_enable != nullptr && !transformers_to_enable->empty()) {
    // pick custom transformers enabled for this session
    std::vector<TransformerProviderSet> filtered_list;
    for (const auto& t_name : *transformers_to_enable) {
      std::for_each(transformers.begin(), transformers.end(), 
          [&](TransformerProviderSet& item) {
             if((item.first != nullptr) && (item.first->Name() == t_name)){
               filtered_list.push_back(std::move(item));
             }
          });
    }
    return filtered_list;
  }

  return transformers;
}

}  // namespace transformerutils
}  // namespace onnxruntime
