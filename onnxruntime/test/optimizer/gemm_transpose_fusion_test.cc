// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/graph/model.h"
#include "core/optimizer/gemm_transpose_fusion.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

namespace {

void RunInputTransposeTest(const std::vector<int64_t>& perm,
                           const std::vector<int64_t>& input_shape,
                           bool expect_fusion,
                           int64_t expected_trans_a) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model("GemmTransposeFusionPermutation", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 13}}, {}, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  auto* input_a = builder.MakeInput<float>(input_shape, -1.0f, 1.0f);
  auto* transpose_output = builder.MakeIntermediate();
  Node& transpose = builder.AddNode("Transpose", {input_a}, {transpose_output});
  transpose.AddAttribute("perm", perm);

  auto* input_b = builder.MakeInput<float>({3, 4}, -1.0f, 1.0f);
  auto* output = builder.MakeOutput();
  Node& gemm = builder.AddNode("Gemm", {transpose_output, input_b}, {output});
  gemm.AddAttribute("transA", int64_t{0});
  gemm.AddAttribute("transB", int64_t{0});
  gemm.AddAttribute("alpha", 1.0f);
  gemm.AddAttribute("beta", 1.0f);

  ASSERT_STATUS_OK(graph.Resolve());

  RuleBasedGraphTransformer transformer("GemmTransposeFusionTest");
  ASSERT_STATUS_OK(transformer.Register(std::make_unique<GemmTransposeFusion>()));

  bool modified = false;
  ASSERT_STATUS_OK(transformer.Apply(graph, modified, logger));
  ASSERT_STATUS_OK(graph.Resolve());

  EXPECT_EQ(modified, expect_fusion);
  auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(op_counts["Transpose"], expect_fusion ? 0 : 1);
  EXPECT_EQ(op_counts["Gemm"], 1);

  const Node* resulting_gemm = nullptr;
  for (const auto& graph_node : graph.Nodes()) {
    if (graph_node.OpType() == "Gemm") {
      resulting_gemm = &graph_node;
      break;
    }
  }

  ASSERT_NE(resulting_gemm, nullptr);
  EXPECT_EQ(resulting_gemm->GetAttributes().at("transA").i(), expected_trans_a);
}

}  // namespace

TEST(GemmTransposeFusionTest, IdentityInputTransposeIsNotFolded) {
  RunInputTransposeTest({0, 1}, {2, 3}, false, 0);
}

TEST(GemmTransposeFusionTest, MatrixInputTransposeIsFolded) {
  RunInputTransposeTest({1, 0}, {3, 2}, true, 1);
}

}  // namespace test
}  // namespace onnxruntime
