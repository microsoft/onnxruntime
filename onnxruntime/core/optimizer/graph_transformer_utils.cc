
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/unsqueeze_elimination.h"

namespace onnxruntime {

namespace transformerutils {

std::vector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(const TransformerLevel& level, const std::vector<std::string>& custom_list) {
  std::vector<std::unique_ptr<RewriteRule>> rules;

  if (level == TransformerLevel::Level1) {
    rules.push_back(std::make_unique<EliminateIdentity>());
    rules.push_back(std::make_unique<EliminateSlice>());
  } else {
    return rules;
  }

  if (!custom_list.empty()) {
    std::vector<std::unique_ptr<RewriteRule>> filtered_list;
    for (const auto& rule_name : custom_list) {
      auto it = std::find_if(rules.begin(), rules.end(), [&](std::unique_ptr<RewriteRule>& item) {
        return item->Name() == rule_name;
      });
      if (it != rules.end()) {
        filtered_list.push_back(std::move(*it));
      }
    }
    return filtered_list;
  }

  return rules;
}

std::vector<std::pair<std::unique_ptr<GraphTransformer>, std::vector<std::string>>> GenerateTransformers(const TransformerLevel& level, const std::vector<std::string>& custom_list) {
  std::vector<std::pair<std::unique_ptr<GraphTransformer>, std::vector<std::string>>> transformers;

  if (level == TransformerLevel::Level1) {
    std::vector<std::string> execution_providers = {""};
    transformers.emplace_back(std::make_unique<UnsqueezeElimination>(), execution_providers);

  } else if (level == Level2) {
    std::vector<std::string> execution_providers = {onnxruntime::kCpuExecutionProvider};

    transformers.emplace_back(std::make_unique<ConvAddFusion>(), execution_providers);
    transformers.emplace_back(std::make_unique<ConvMulFusion>(), execution_providers);
  } else {
    return transformers;
  }

  if (!custom_list.empty()) {
    // pick custom transformers enabled for this session
    std::vector<std::pair<std::unique_ptr<GraphTransformer>, std::vector<std::string>>> filtered_list;
    for (const auto& t_name : custom_list) {
      auto it = std::find_if(transformers.begin(), transformers.end(),
                             [&](std::pair<std::unique_ptr<GraphTransformer>, std::vector<std::string>>& item) {
                               return item.first->Name() == t_name;
                             });

      if (it != transformers.end()) {
        filtered_list.push_back(std::move(*it));
      }
    }
    return filtered_list;
  }

  return transformers;
}

}  // namespace transformerutils
}  // namespace onnxruntime
