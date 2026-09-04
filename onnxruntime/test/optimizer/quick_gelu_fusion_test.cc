// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "gtest/gtest.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"

namespace onnxruntime {
namespace test {

namespace {

template <typename T>
void BuildQuickGeluGraph(ModelTestBuilder& builder) {
  auto* input = builder.MakeInput<T>(
      {2, 3},
      std::vector<T>{static_cast<T>(-2.0), static_cast<T>(-1.0), static_cast<T>(-0.5),
                     static_cast<T>(0.5), static_cast<T>(1.0), static_cast<T>(2.0)});
  auto* alpha = builder.MakeScalarInitializer<T>(static_cast<T>(1.702));
  auto* scaled = builder.MakeIntermediate<T>(std::vector<int64_t>{2, 3});
  auto* sigmoid = builder.MakeIntermediate<T>(std::vector<int64_t>{2, 3});
  auto* output = builder.MakeOutput<T>(std::vector<int64_t>{2, 3});

  builder.AddNode("Mul", {input, alpha}, {scaled});
  builder.AddNode("Sigmoid", {scaled}, {sigmoid});
  builder.AddNode("Mul", {input, sigmoid}, {output});
}

}  // namespace

TEST(QuickGeluFusionTest, CpuFloatIsFused) {
  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.QuickGelu"], 1);
    EXPECT_EQ(op_to_count["Sigmoid"], 0);
  };

  TransformerTester(BuildQuickGeluGraph<float>, check_graph,
                    TransformerLevel::Level1, TransformerLevel::Level2, 13, 1e-5);
}

TEST(QuickGeluFusionTest, CpuDoubleIsNotFused) {
  auto check_graph = [](InferenceSessionWrapper& session) {
    auto op_to_count = CountOpsInGraph(session.GetGraph());
    EXPECT_EQ(op_to_count["com.microsoft.QuickGelu"], 0);
    EXPECT_EQ(op_to_count["Sigmoid"], 1);
  };

  TransformerTester(BuildQuickGeluGraph<double>, check_graph,
                    TransformerLevel::Level1, TransformerLevel::Level2, 13, 1e-12);
}

}  // namespace test
}  // namespace onnxruntime
