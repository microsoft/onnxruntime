// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/graph/graph_utils.h"
#include "core/graph/model.h"
#include "core/optimizer/div_mul_fusion.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

TEST(DivMulFusionTest, PreservesProducerEdgeForReplacementInput) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model("DivMulFusionProducerEdge", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 13}}, {}, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  auto* input = builder.MakeInput<float>({2}, -1.0f, 1.0f);
  auto* produced = builder.MakeIntermediate();
  builder.AddNode("Neg", {input}, {produced});

  auto* one_a = builder.MakeScalarInitializer<float>(1.0f);
  auto* one_b = builder.MakeScalarInitializer<float>(1.0f);
  auto* div_output = builder.MakeIntermediate();
  builder.AddNode("Div", {one_a, one_b}, {div_output});

  auto* output = builder.MakeOutput();
  builder.AddNode("Mul", {produced, div_output}, {output});

  ASSERT_STATUS_OK(graph.Resolve());

  RuleBasedGraphTransformer transformer("DivMulFusionTest");
  ASSERT_STATUS_OK(transformer.Register(std::make_unique<DivMulFusion>()));

  bool modified = false;
  ASSERT_STATUS_OK(transformer.Apply(graph, modified, logger));
  EXPECT_TRUE(modified);

  const auto op_counts = CountOpsInGraph(graph);
  EXPECT_EQ(GetOpCount(op_counts, "Neg"), 1);
  EXPECT_EQ(GetOpCount(op_counts, "Div"), 1);
  EXPECT_EQ(GetOpCount(op_counts, "Mul"), 0);

  const Node* fused_div = nullptr;
  for (const auto& graph_node : graph.Nodes()) {
    if (graph_node.OpType() == "Div") {
      fused_div = &graph_node;
      break;
    }
  }

  ASSERT_NE(fused_div, nullptr);
  EXPECT_EQ(fused_div->InputDefs()[0]->Name(), produced->Name());
  ASSERT_NE(graph_utils::GetInputNode(*fused_div, 0), nullptr);
  EXPECT_EQ(graph_utils::GetInputNode(*fused_div, 0)->OpType(), "Neg");
  EXPECT_EQ(fused_div->GetInputEdgesCount(), 1);

  ASSERT_STATUS_OK(graph.Resolve());
}

}  // namespace test
}  // namespace onnxruntime
