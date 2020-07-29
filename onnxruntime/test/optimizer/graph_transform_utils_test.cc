// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/onnx_protobuf.h"
#include "test/framework/test_utils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"
#include "core/optimizer/graph_transformer_utils.h"
#include "core/session/inference_session.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

TEST(GraphTransformerUtilsTests, TestGenerateRewriterules) {
  // Generate all test
  auto rewrite_rules = optimizer_utils::GenerateRewriteRules(TransformerLevel::Level1);
  ASSERT_TRUE(rewrite_rules.size() != 0);

  // Rule name match test
  std::vector<std::string> custom_list = {"EliminateIdentity", "ConvAddFusion", "ConvMulFusion", "abc", "def"};
  rewrite_rules = optimizer_utils::GenerateRewriteRules(TransformerLevel::Level1, custom_list);
  // validate each rule returned is present in the custom list
  for (const auto& rule : rewrite_rules) {
    ASSERT_TRUE(std::find(custom_list.begin(), custom_list.end(), rule->Name()) != custom_list.end());
  }

  // Rule name no match test. Test to validate empty rules list is returned when
  // there is no match in custom list
  custom_list = {"abc"};
  rewrite_rules = optimizer_utils::GenerateRewriteRules(TransformerLevel::Level1, custom_list);
  ASSERT_TRUE(rewrite_rules.size() == 0);
}

TEST(GraphTransformerUtilsTests, TestGenerateGraphTransformers) {
  // custom list of rules and transformers
  std::string l1_rule1 = "EliminateIdentity";
  std::string l1_transformer = "ConstantFolding";
  std::string l2_transformer = "ConvActivationFusion";
  std::vector<std::string> custom_list = {l1_rule1, l1_transformer, l2_transformer};

  auto transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level1, {}, custom_list);
  ASSERT_TRUE(transformers.size() == 2);

  auto l1_rule_transformer_name = optimizer_utils::GenerateRuleBasedTransformerName(TransformerLevel::Level1);
  RuleBasedGraphTransformer* rule_transformer = nullptr;
  for (const auto& transformer : transformers) {
    if (transformer->Name() == l1_rule_transformer_name) {
      rule_transformer = static_cast<RuleBasedGraphTransformer*>(transformers[0].get());
    }
  }
  ASSERT_TRUE(rule_transformer && rule_transformer->RulesCount() == 1);

  transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level2, {}, custom_list);
#ifndef DISABLE_CONTRIB_OPS
  ASSERT_TRUE(transformers.size() == 1);
#else
  ASSERT_TRUE(transformers.size() == 0);
#endif
}

TEST(GraphTransformerUtilsTests, TestCustomOnlyTransformers) {
  // Transformers that are disabled by default. They can only be enabled by custom list.
  std::string l2_transformer = "GeluApproximation";

  std::vector<std::string> default_list = {};
  auto default_transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level2, {}, default_list);
  for (auto& transformer : default_transformers) {
    ASSERT_TRUE(transformer->Name() != l2_transformer);
  }

  std::vector<std::string> custom_list = {l2_transformer};
  auto custom_transformers = optimizer_utils::GenerateTransformers(TransformerLevel::Level2, {}, custom_list);
#ifndef DISABLE_CONTRIB_OPS
  ASSERT_TRUE(custom_transformers.size() == 1);
  ASSERT_TRUE(custom_transformers[0]->Name() == l2_transformer);
#else
  ASSERT_TRUE(custom_transformers.size() == 0);
#endif
}

}  // namespace test
}  // namespace onnxruntime
