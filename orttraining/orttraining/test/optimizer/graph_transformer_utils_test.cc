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
  std::vector<std::string> custom_list = {l1_rule1, l1_transformer, l2_transformer};

  auto transformers = training::transformer_utils::GenerateTransformers(TransformerLevel::Level1, {}, {}, custom_list);
  ASSERT_TRUE(transformers.size() == 1);

  auto l1_rule_transformer_name = optimizer_utils::GenerateRuleBasedTransformerName(TransformerLevel::Level1);
  RuleBasedGraphTransformer* rule_transformer = nullptr;
  for (const auto& transformer : transformers) {
    if (transformer->Name() == l1_rule_transformer_name) {
      rule_transformer = dynamic_cast<RuleBasedGraphTransformer*>(transformers[0].get());
    }
  }
  ASSERT_TRUE(rule_transformer && rule_transformer->RulesCount() == 1);

  transformers = training::transformer_utils::GenerateTransformers(TransformerLevel::Level2, {}, {}, custom_list);
#ifndef DISABLE_CONTRIB_OPS
  ASSERT_TRUE(transformers.size() == 1);
#else
  ASSERT_TRUE(transformers.size() == 0);
#endif
}

}  // namespace test
}  // namespace onnxruntime
