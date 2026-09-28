// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "core/graph/model.h"
#include "core/optimizer/fuse_initializers_transformer.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {

TEST(TransformerTest, FuseInitializerUsesCastOutputValueName) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model("FuseInitializerUsesCastOutputValueName", false, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  NodeArg* initializer = builder.MakeInitializer<MLFloat16>({1}, {MLFloat16(1.0f)});
  NodeArg* cast_output = builder.MakeIntermediate<float>(std::vector<int64_t>{1});
  Node& cast_node = graph.AddNode("cast_node_name", "Cast", "", {initializer}, {cast_output});
  cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

  NodeArg* output = builder.MakeOutput<float>(std::vector<int64_t>{1});
  graph.AddNode("consumer_node_name", "Neg", "", {cast_output}, {output});

  graph.SetOutputs({output});
  ASSERT_STATUS_OK(graph.Resolve());
  ASSERT_NE(cast_node.Name(), cast_output->Name());

  FuseInitializersTransformer transformer("TransformerTest.FusedInitializers",
                                          DataTypeImpl::GetTensorType<MLFloat16>(),
                                          DataTypeImpl::GetTensorType<float>());
  bool modified = false;
  ASSERT_STATUS_OK(transformer.Apply(graph, modified, logger));
  ASSERT_STATUS_OK(graph.Resolve());

  EXPECT_TRUE(modified);
  EXPECT_EQ(0, CountOpsInGraph(graph)["Cast"]);
  EXPECT_EQ(1, CountOpsInGraph(graph)["Neg"]);
}

TEST(TransformerTest, FuseInitializerReportsIntentionalRemoval) {
  auto& logger = DefaultLoggingManager().DefaultLogger();
  Model model("FuseInitializerReportsIntentionalRemoval", false, logger);
  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  NodeArg* initializer = builder.MakeInitializer<MLFloat16>({1}, {MLFloat16(1.0f)});
  NodeArg* cast_output = builder.MakeIntermediate<float>(std::vector<int64_t>{1});
  Node& cast_node = graph.AddNode("cast_node", "Cast", "", {initializer}, {cast_output});
  cast_node.AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));

  NodeArg* output = builder.MakeOutput<float>(std::vector<int64_t>{1});
  graph.AddNode("consumer_node", "Neg", "", {cast_output}, {output});

  graph.SetOutputs({output});
  ASSERT_STATUS_OK(graph.Resolve());

  const NodeIndex cast_node_index = cast_node.Index();
  std::vector<NodeIndex> removed_node_indices;
  graph.SetNodeRemovalCallback(
      [&removed_node_indices](const Graph&, gsl::span<const NodeIndex> node_indices) {
        removed_node_indices.insert(
            removed_node_indices.end(), node_indices.begin(), node_indices.end());
      });

  FuseInitializersTransformer transformer(
      "TransformerTest.FusedInitializers",
      DataTypeImpl::GetTensorType<MLFloat16>(),
      DataTypeImpl::GetTensorType<float>());
  bool modified = false;
  ASSERT_STATUS_OK(transformer.Apply(graph, modified, logger));
  graph.SetNodeRemovalCallback({});

  EXPECT_TRUE(modified);
  ASSERT_EQ(removed_node_indices.size(), 1U);
  EXPECT_EQ(removed_node_indices[0], cast_node_index);
  EXPECT_EQ(graph.GetNode(cast_node_index), nullptr);
}

}  // namespace test
}  // namespace onnxruntime
