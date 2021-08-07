// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/framework/test_utils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"
#include "orttraining/core/optimizer/graph_transformer_utils.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/session/inference_session.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

TEST(GraphTransformerUtilsTestsForTraining, TestGenerateGraphTransformers) {
  // custom list of rules and transformers
  std::string l1_rule1 = "EliminateIdentity";
  std::string l1_transformer = "ConstantFolding";
  std::string l2_transformer = "ConvActivationFusion";
  std::unordered_set<std::string> disabled = {l1_rule1, l1_transformer, l2_transformer};
  CPUExecutionProvider cpu_ep(CPUExecutionProviderInfo{});

  auto all_transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level1, {}, cpu_ep);
  auto filtered_transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level1, {}, cpu_ep, disabled);

  // check ConstantFolding transformer was removed
  ASSERT_TRUE(filtered_transformers.size() == all_transformers.size() - 1);

  // check EliminateIdentity rule was removed from inside the rule based transformer
  auto l1_rule_transformer_name = optimizer_utils::GenerateRuleBasedTransformerName(TransformerLevel::Level1);
  RuleBasedGraphTransformer* rule_transformer = nullptr;
  for (const auto& transformer : filtered_transformers) {
    if (transformer->Name() == l1_rule_transformer_name) {
      rule_transformer = static_cast<RuleBasedGraphTransformer*>(transformer.get());

      // get the full set of rules and check EliminateIdentity was correctly removed
      auto l1_rewrite_rules = optimizer_utils::GenerateRewriteRules(TransformerLevel::Level1);
      ASSERT_TRUE(rule_transformer->RulesCount() == l1_rewrite_rules.size() - 1);
      break;
    }
  }

  ASSERT_TRUE(rule_transformer) << "RuleBased transformer should have been added by GenerateTransformers";

#ifndef DISABLE_CONTRIB_OPS
  // check that ConvActivationFusion was removed
  all_transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level2, {}, cpu_ep);
  filtered_transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level2, {}, cpu_ep, disabled);
  ASSERT_TRUE(filtered_transformers.size() == all_transformers.size() - 1);
#endif
}

}  // namespace test
}  // namespace onnxruntime
