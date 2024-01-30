// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "gtest/gtest.h"
#include "graph_transform_test_builder.h"

#include "core/graph/graph.h"

namespace onnxruntime {
namespace test {

#ifndef DISABLE_CONTRIB_OPS

void TestConvPath(const std::vector<int64_t>& input_shape, const std::vector<int64_t>& weights_shape,
                  const std::vector<int64_t>& output_shape, int64_t group) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>(input_shape, -31, 31);
    auto* output_arg = builder.MakeOutput();
    auto* bias_arg = builder.MakeInitializer<float>({weights_shape[0]}, -20.f, 20.f);
    auto* add_arg = builder.MakeInput<float>(output_shape, -20.f, 20.f);
    auto* weight_arg = builder.MakeInitializer<float>(weights_shape, -2.f, 2.f);
    auto* conv_out_arg = builder.MakeIntermediate();

    auto& conv_node = builder.AddNode("Conv", {input_arg, weight_arg, bias_arg}, {conv_out_arg});
    conv_node.AddAttribute("group", group);
    builder.AddNode("Add", {conv_out_arg, add_arg}, {output_arg});
  };

  auto check_graph = [&](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.FusedConv"], 1);
  };
  InlinedHashSet<std::string> disabled_optimizers = {"NchwcTransformer"};
  TransformerTester(build_test_case,
                    check_graph,
                    TransformerLevel::Default,
                    TransformerLevel::Level3, 12, 0.0001, 0.000001,
                    0, {}, disabled_optimizers);
}

TEST(ConvAddActivationFusionTests, ConvExpandThenGemm) {
  // hit MlasConvAlgorithmExpandThenGemm
  TestConvPath({1, 16, 5, 5}, {16, 16, 3, 3}, {1, 16, 3, 3}, 1);
}

TEST(ConvAddActivationFusionTests, ConvDepthwise) {
  // MlasConvAlgorithmDepthwise or MlasConvAlgorithmExpandThenGemmSegmented
  TestConvPath({1, 16, 5, 5}, {16, 1, 3, 3}, {1, 16, 3, 3}, 16);
}
#ifdef __wasm__
TEST(ConvAddActivationFusionTests, DISABLED_ConvGemmDirect) {
#else
TEST(ConvAddActivationFusionTests, DISABLED_ConvGemmDirect) {
#endif
  // MlasConvAlgorithmGemmDirect
  TestConvPath({1, 16, 5, 5}, {16, 16, 1, 1}, {1, 16, 5, 5}, 1);
  // MlasConvAlgorithmGemmDirect
  TestConvPath({1, 16, 5, 5}, {16, 1, 5, 5}, {1, 16, 1, 1}, 16);
}

#endif  // DISABLE_CONTRIB_OPS

}  // namespace test
}  // namespace onnxruntime
