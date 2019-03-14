
#include "core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/unsqueeze_elimination.h"

namespace onnxruntime {

namespace transformerutils {

Status ValidateTransformerLevel(unsigned int level) {
  auto t_level = (unsigned int)std::pow(2, level);
  if (t_level < TransformerLevel::MaxTransformerLevel) {
    return Status::OK();
  }
  return Status(common::ONNXRUNTIME, common::FAIL, "Unsupported level " + level);
}

void SetTransformerContext(const uint32_t& level, uint32_t& levels_enabled, 
                           std::vector<TransformerLevel>* all_levels) {

  ORT_ENFORCE(ValidateTransformerLevel(level).IsOK(),
              "Unsupported transformer level specified", level);

  levels_enabled = TransformerLevel::Default;
  if (level == 1) {
    levels_enabled |= TransformerLevel::Level1;
  } else if (level == 2) {
    levels_enabled |= TransformerLevel::Level1 | TransformerLevel::Level2;
  }

  if (all_levels != nullptr) {
    all_levels->push_back(TransformerLevel::Level2);
    all_levels->push_back(TransformerLevel::Level1);
  }
}

std::vector<std::unique_ptr<RewriteRule>> GenerateRewriteRules(const TransformerLevel& level, 
                                                               const std::vector<std::string>* custom_list) {
  std::vector<std::unique_ptr<RewriteRule>> rules;
  switch (level) {
    case TransformerLevel::Level1:
      rules.push_back(std::make_unique<EliminateIdentity>());
      rules.push_back(std::make_unique<EliminateSlice>());
      break;

    case TransformerLevel::Level2:
      break;
    default:
      ORT_ENFORCE(false, "Unsupported level" + level);
  }

  if (custom_list != nullptr && !custom_list->empty()) {
    std::vector<std::unique_ptr<RewriteRule>> filtered_list;
    for (const auto& rule_name : *custom_list) {
      std::for_each(rules.begin(), rules.end(), [&](std::unique_ptr<RewriteRule>& item) {
        if (item->Name() == rule_name) {
          filtered_list.push_back(std::move(item));
        }
      });
    }
    return filtered_list;
  }

  return rules;
}

std::vector<TransformerProviderSet> GenerateTransformers(const TransformerLevel& level, 
                                                         const std::vector<std::string>* custom_list) {
  std::vector<TransformerProviderSet> transformers;
  switch (level) {
    case TransformerLevel::Level1: {
      std::vector<std::string> l1_execution_providers = {"", onnxruntime::kCpuExecutionProvider};
      transformers.emplace_back(std::make_unique<UnsqueezeElimination>(), l1_execution_providers);
    } break;

    case TransformerLevel::Level2: {
      std::vector<std::string> l2_execution_providers = {onnxruntime::kCpuExecutionProvider};
      transformers.emplace_back(std::make_unique<ConvAddFusion>(), l2_execution_providers);
      transformers.emplace_back(std::make_unique<ConvMulFusion>(), l2_execution_providers);
    } break;

    default:
      ORT_ENFORCE(false, "Unsupported level" + level);
      break;
  }

  if (custom_list != nullptr && !custom_list->empty()) {
    // pick custom transformers enabled for this session
    std::vector<TransformerProviderSet> filtered_list;
    for (const auto& t_name : *custom_list) {
      std::for_each(transformers.begin(), transformers.end(), 
          [&](TransformerProviderSet& item) {
             if(item.first->Name() == t_name){
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
