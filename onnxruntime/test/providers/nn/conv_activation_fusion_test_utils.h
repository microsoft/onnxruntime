// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <memory>
#include <string>

#include "gtest/gtest.h"

#include "core/graph/graph_utils.h"
#include "test/optimizer/fusion_test_util.h"
#include "test/unittest_util/framework_test_utils.h"

namespace onnxruntime {
namespace test {

struct ConvActivationFusionExecutionTestCase {
  const char* name;
  std::function<std::unique_ptr<IExecutionProvider>()> ep_factory;
  const char* fused_op_type;
  const char* fused_domain;
};

class ConvActivationFusionExecutionTests
    : public ::testing::TestWithParam<ConvActivationFusionExecutionTestCase> {};

inline void RunConvMulReluTanhFusionTest(
    const ConvActivationFusionExecutionTestCase& test_case,
    const std::function<void(SessionOptions&)>& add_session_options = {}) {
  auto ep = test_case.ep_factory();
  if (!ep) {
    GTEST_SKIP() << test_case.name << " EP unavailable in this build.";
  }
  const std::string ep_type = ep->Type();
  ep.reset();

  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({1, 3, 2, 2},
                                           {-2.0f, -1.0f, 1.0f, 2.0f,
                                            -1.5f, -0.5f, 0.5f, 1.5f,
                                            -3.0f, -0.25f, 0.25f, 3.0f});
    auto* weight = builder.MakeInitializer<float>({3, 3, 1, 1},
                                                  {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, -1.0f, 1.0f});
    auto* scale = builder.MakeInitializer<float>({3, 1, 1}, {1.0f, 2.0f, -1.0f});
    auto* folded_weight = builder.MakeIntermediate();
    auto* conv_out = builder.MakeIntermediate();
    auto* mul_out = builder.MakeIntermediate();
    auto* relu_out = builder.MakeIntermediate();
    auto* output = builder.MakeOutput();
    builder.AddNode("Relu", {weight}, {folded_weight});
    builder.AddNode("Conv", {input, folded_weight}, {conv_out});
    builder.AddNode("Mul", {conv_out, scale}, {mul_out});
    builder.AddNode("Relu", {mul_out}, {relu_out});
    builder.AddNode("Tanh", {relu_out}, {output});
  };

  auto check = [&test_case, &ep_type](InferenceSessionWrapper& session) {
    auto op_counts = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_counts["Relu"], 0) << "ConstantFolding and Relu fusion must both run";
    EXPECT_EQ(op_counts["Mul"], 0) << "ConvMulFusion must run";
    EXPECT_EQ(op_counts["Tanh"], 1) << "Tanh must remain after the fused Relu";
    int fused_conv_count = 0;
    for (const auto& node : session.GetGraph().Nodes()) {
      EXPECT_EQ(node.GetExecutionProviderType(), ep_type);
      if (node.OpType() == test_case.fused_op_type && node.Domain() == test_case.fused_domain) {
        ++fused_conv_count;
        const auto* activation = graph_utils::GetNodeAttribute(node, "activation");
        ASSERT_NE(activation, nullptr);
        EXPECT_EQ(activation->s(), "Relu");
      }
    }
    EXPECT_EQ(fused_conv_count, 1);
  };

  RunFusionTransformerTest(
      build_test_case, check, TransformerLevel::Level1, TransformerLevel::Level2, 17,
      /*per_sample_tolerance=*/1e-5, /*relative_per_sample_tolerance=*/1e-5,
      /*transformer=*/nullptr, test_case.ep_factory, add_session_options);
}

}  // namespace test
}  // namespace onnxruntime
