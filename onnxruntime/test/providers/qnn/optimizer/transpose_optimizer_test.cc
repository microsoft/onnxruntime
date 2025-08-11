// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "core/graph/constants.h"
#include "core/optimizer/transpose_optimizer.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/graph_transform_test_builder.h"

namespace onnxruntime {
namespace test {

static void TestTransposeReshapeTranspose(const std::vector<int64_t>& input_shape,
                                          const std::vector<int64_t>& transpose1_perm,
                                          const std::vector<int64_t>& reshape_shape,
                                          const std::vector<int64_t>& transpose2_perm,
                                          const bool expected_optimized = true) {
  auto build_test_case = [&](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>(input_shape, 0.0, 1.0);
    auto* reshape_shape_value = builder.MakeInitializer<int64_t>({int64_t(reshape_shape.size())}, reshape_shape);

    auto* transpose1_out = builder.MakeIntermediate();
    auto* reshape_out = builder.MakeIntermediate();
    auto* transpose2_out = builder.MakeOutput();

    auto& transpose1 = builder.AddNode("Transpose", {input_arg}, {transpose1_out});
    transpose1.AddAttribute("perm", transpose1_perm);
    transpose1.SetExecutionProviderType(kQnnExecutionProvider);

    auto& reshape = builder.AddNode("Reshape", {transpose1_out, reshape_shape_value}, {reshape_out});
    reshape.SetExecutionProviderType(kQnnExecutionProvider);

    auto& transpose2 = builder.AddNode("Transpose", {reshape_out}, {transpose2_out});
    transpose2.AddAttribute("perm", transpose2_perm);
    transpose2.SetExecutionProviderType(kQnnExecutionProvider);
  };

  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model("TransformerTester", false, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder helper(graph);
  build_test_case(helper);
  helper.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  std::unique_ptr<TransposeOptimizer> optimizer = std::make_unique<TransposeOptimizer>(CPUAllocator::DefaultInstance(),
                                                                                       kQnnExecutionProvider);
  bool modified = false;
  ASSERT_STATUS_OK(optimizer->Apply(graph, modified, logger));
  ASSERT_EQ(modified, expected_optimized);

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], expected_optimized ? 0 : 2);
  ASSERT_EQ(op_to_count["Reshape"], 1);
}

TEST(QnnTransposeOptimizerTests, TransposeReshapeTranspose) {
  TestTransposeReshapeTranspose({1, 3, 32}, {0, 2, 1}, {1, 1, 32, 3}, {0, 3, 1, 2});
  TestTransposeReshapeTranspose({1, 32, 3}, {0, 2, 1}, {1, 3, 1, 32}, {0, 2, 3, 1});
  TestTransposeReshapeTranspose({1, 3, 32, 32}, {0, 2, 3, 1}, {1, 32 * 32, 3}, {0, 2, 1});
  TestTransposeReshapeTranspose({1, 3, 32, 32}, {0, 2, 3, 1}, {1, 32 * 32, 1, 3}, {0, 3, 1, 2});
  TestTransposeReshapeTranspose({1, 32, 32, 3}, {0, 3, 1, 2}, {1, 3, 32 * 32}, {0, 2, 1});
  TestTransposeReshapeTranspose({1, 32, 32, 3}, {0, 3, 1, 2}, {1, 3, 32 * 32, 1}, {0, 2, 3, 1});

  TestTransposeReshapeTranspose({1, 3, 32}, {0, 2, 1}, {1, 8, 2, 6}, {0, 3, 1, 2}, false);
  TestTransposeReshapeTranspose({1, 3, 32, 32}, {0, 2, 3, 1}, {1, 32, 16, 6}, {0, 3, 1, 2}, false);
  TestTransposeReshapeTranspose({1, 32, 32, 3}, {0, 3, 1, 2}, {1, 6, 16, 32}, {0, 2, 3, 1}, false);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)
