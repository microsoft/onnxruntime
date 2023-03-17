// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/optimizer/qdq_transformer/ensure_unique_dq_for_node_unit.h"

#include "gtest/gtest.h"

#include "test/optimizer/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {

/* test ideas
 * - shared DQ among different nodes
 * - shared DQ among different inputs in a single node
 * - shared DQ among node and graph output
 * - shared DQ among node and subgraph (node implicit input)
 */

TEST(EnsureUniqueDQForNodeUnitTests, DQSharedAmongNodes) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    const size_t num_consumer_nodes = 3;

    const auto input_shape = std::vector<int64_t>{1, 2};
    const float scale = 0.5f;
    const uint8_t zero_point = 0;

    auto* dq_input = builder.MakeInput<uint8_t>(input_shape, uint8_t{0}, uint8_t{255});
    auto* dq_output = builder.MakeIntermediate();
    builder.AddDequantizeLinearNode(dq_input, scale, zero_point, dq_output);

    for (size_t i = 0; i < num_consumer_nodes; ++i) {
      auto* consumer_output = builder.MakeOutput();
      builder.AddNode("Softmax", {dq_output}, {consumer_output});
    }
  };

  auto check_transformed_graph = [](InferenceSessionWrapper& session) {
    const auto& graph = session.GetGraph();
    const auto op_counts = CountOpsInGraph(graph);
    ASSERT_EQ(OpCount(op_counts, "DequantizeLinear"), 3);
  };

  TransformerTester(build_test_case, check_transformed_graph, TransformerLevel::Default, TransformerLevel::Level1, 12, 0.0, 0.0, std::make_unique<EnsureUniqueDQForNodeUnit>());
}

}  // namespace onnxruntime::test
