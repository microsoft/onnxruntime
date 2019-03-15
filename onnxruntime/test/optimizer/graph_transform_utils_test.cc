// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/framework/test_utils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"
#include "core/optimizer/graph_transformer_utils.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

TEST(GraphTransformerUtilsTests, TestGenerateRewriterules) {
  // Generate all test
  auto rewrite_rules = transformer_utils::GenerateRewriteRules(TransformerLevel::Level1);
  ASSERT_TRUE(rewrite_rules.size() != 0);

  // Rule name match test
  std::vector<std::string> custom_list = {"EliminateIdentity", "ConvAddFusion", "ConvMulFusion", "abc", "def"};
  rewrite_rules = transformer_utils::GenerateRewriteRules(TransformerLevel::Level1, &custom_list);
  // validate each rule returned is present in the custom list
  for (const auto& rule : rewrite_rules) {
    ASSERT_TRUE(std::find(custom_list.begin(), custom_list.end(), rule->Name()) != custom_list.end());
  }

  // Rule name no match test. Test to validate empty rules list is returned when 
  // there is no match in custom list
  custom_list = {"abc"};
  rewrite_rules = transformer_utils::GenerateRewriteRules(TransformerLevel::Level1, &custom_list);
  ASSERT_TRUE(rewrite_rules.size() == 0);
}

TEST(GraphTransformerUtilsTests, TestGenerateGraphTransformers) {
  auto transformers = transformer_utils::GenerateTransformers(TransformerLevel::Level2);
  ASSERT_TRUE(transformers.size() != 0);

  // Transformer name match test
  std::vector<std::string> custom_list = {"EliminateIdentity", "ConvAddFusion", "ConvMulFusion", "abc", "def"};
  transformers = transformer_utils::GenerateTransformers(TransformerLevel::Level2, &custom_list);
  ASSERT_TRUE(transformers.size() == 2);
  // validate each rule returned is present in the custom list
  for (const auto& transformer : transformers) {
    ASSERT_TRUE(std::find(custom_list.begin(), custom_list.end(), transformer.first->Name()) != custom_list.end());
  }

  // Transformer name no match test. When there is no match empty list is expected.
  custom_list = {"EliminateIdentity"};
  transformers = transformer_utils::GenerateTransformers(TransformerLevel::Level2, &custom_list);
  ASSERT_TRUE(transformers.size() == 0);
}

}  // namespace test
}  // namespace onnxruntime
