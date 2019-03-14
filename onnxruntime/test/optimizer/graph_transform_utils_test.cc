// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/framework/data_types.h"
#include "core/framework/ml_value.h"
#include "core/util/math.h"
#include "core/platform/env.h"
#include "test/framework/test_utils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/graph_transformer_utils.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

TEST(GraphTransformerUtilsTests, TestValidateLevels) {  
  
  ASSERT_TRUE(transformerutils::ValidateTransformerLevel(0).IsOK());
  
  ASSERT_TRUE(transformerutils::ValidateTransformerLevel(1).IsOK());

  ASSERT_TRUE(transformerutils::ValidateTransformerLevel(2).IsOK());

  ASSERT_FALSE(transformerutils::ValidateTransformerLevel(3).IsOK());  
}

TEST(GraphTransformerUtilsTests, TestSetTransformerContext) {
  uint32_t levels_enabled;

  transformerutils::SetTransformerContext(0, levels_enabled);
  ASSERT_TRUE(levels_enabled == 1);

  transformerutils::SetTransformerContext(1, levels_enabled);
  ASSERT_TRUE(levels_enabled == 3);

  transformerutils::SetTransformerContext(2, levels_enabled);
  ASSERT_TRUE(levels_enabled == 7);  
}

TEST(GraphTransformerUtilsTests, TestGenerateRewriterules) {

  // Generate all test
  auto rewrite_rules = transformerutils::GenerateRewriteRules(TransformerLevel::Level1);
  ASSERT_TRUE(rewrite_rules.size() != 0);  

  // Rule name match test
  std::vector<std::string> custom_list = {"EliminateIdentity"};
  rewrite_rules = transformerutils::GenerateRewriteRules(TransformerLevel::Level1, &custom_list);
  ASSERT_TRUE(rewrite_rules.size() == custom_list.size());

  // 1 Rule name match test
  custom_list = {"EliminateIdentity", "ConvAddFusion", "ConvMulFusion", "abc", "def"};
  rewrite_rules = transformerutils::GenerateRewriteRules(TransformerLevel::Level1, &custom_list);
  ASSERT_TRUE(rewrite_rules.size() == 1);

  // Rule name no match test
  custom_list = {"abc"};
  rewrite_rules = transformerutils::GenerateRewriteRules(TransformerLevel::Level1, &custom_list);
  ASSERT_TRUE(rewrite_rules.size() == 0);
}

TEST(GraphTransformerUtilsTests, TestGenerateCustomRewriterules) {
  // Rule name match test
  std::vector<std::string> custom_list = {"EliminateIdentity"};
  auto rewrite_rules = transformerutils::GenerateRewriteRules(TransformerLevel::Level1, &custom_list);
  ASSERT_TRUE(rewrite_rules.size() == custom_list.size());

  // 1 Rule name match test
  custom_list = {"EliminateIdentity", "ConvAddFusion", "ConvMulFusion", "abc", "def"};
  rewrite_rules = transformerutils::GenerateRewriteRules(TransformerLevel::Level1, &custom_list);
  ASSERT_TRUE(rewrite_rules.size() == 1);

  // Rule name no match test
  custom_list = {"abc"};
  rewrite_rules = transformerutils::GenerateRewriteRules(TransformerLevel::Level1, &custom_list);
  ASSERT_TRUE(rewrite_rules.size() == 0);
}

TEST(GraphTransformerUtilsTests, TestGenerateGraphTransformers) {
  auto transformers = transformerutils::GenerateTransformers(TransformerLevel::Level2);
  ASSERT_TRUE(transformers.size() != 0);

  // Transformer name match test
  std::vector<std::string> custom_list = {"ConvAddFusion"};
  transformers = transformerutils::GenerateTransformers(TransformerLevel::Level2, &custom_list);
  ASSERT_TRUE(transformers.size() == custom_list.size());

  // Transformer name match test
  custom_list = {"EliminateIdentity", "ConvAddFusion", "ConvMulFusion", "abc", "def"};
  transformers = transformerutils::GenerateTransformers(TransformerLevel::Level2, &custom_list);
  ASSERT_TRUE(transformers.size() == 2);

  // Transformer name no match test
  custom_list = {"EliminateIdentity"};
  transformers = transformerutils::GenerateTransformers(TransformerLevel::Level2, &custom_list);
  ASSERT_TRUE(transformers.size() == 0);
}

}  // namespace test
}  // namespace onnxruntime
