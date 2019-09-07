// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/graph/graph.h"
#include "core/graph/model.h"
#include "core/graph/training/gradient_builder_base.h"
#include "core/graph/training/training_optimizer.h"
#include "core/training/optimizer_graph_builder.h"
#include "test/framework/test_utils.h"

#define ASSERT_STATUS_OK(expr)                           \
  do {                                                   \
    const Status status = (expr);                        \
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage(); \
  } while (0)

using onnxruntime::test::CountOpsInGraph;

namespace onnxruntime {
namespace training {
namespace test {
namespace {
const std::vector<std::string> k_weight_names{"weight_1", "weight_2"};
constexpr const char* const k_optimizer_op_name = "SGDOptimizer";
constexpr const char* const k_all_reduce_op_name = "HorovodAllReduce";

// sets up a base graph with weight and gradient NodeArgs for each weight name
void SetUpBaseGraph(Graph& graph) {
  ONNX_NAMESPACE::TypeProto weight_type{};
  weight_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  weight_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  ONNX_NAMESPACE::TensorProto weight_initializer_base{};
  weight_initializer_base.add_dims(1);
  weight_initializer_base.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  weight_initializer_base.add_float_data(1.0f);

  ONNX_NAMESPACE::TensorProto weight_gradient_initializer_base{weight_initializer_base};
  weight_gradient_initializer_base.set_float_data(0, 2.0f);

  std::vector<const NodeArg*> weights_and_gradients{};

  for (const auto& weight_name : k_weight_names) {
    ONNX_NAMESPACE::TensorProto weight_initializer{weight_initializer_base};
    weight_initializer.set_name(weight_name);
    graph.AddInitializedTensor(weight_initializer);
    weights_and_gradients.emplace_back(&graph.GetOrCreateNodeArg(weight_name, &weight_type));

    const std::string weight_gradient_name = GradientBuilderBase::GradientName(weight_name);
    ONNX_NAMESPACE::TensorProto weight_gradient_initializer{weight_gradient_initializer_base};
    weight_gradient_initializer.set_name(weight_gradient_name);
    graph.AddInitializedTensor(weight_gradient_initializer);
    weights_and_gradients.emplace_back(&graph.GetOrCreateNodeArg(weight_gradient_name, &weight_type));
  }

  // make the weights and gradients persist past Graph::Resolve()
  graph.SetInputs(weights_and_gradients);
  graph.SetOutputs(weights_and_gradients);

  ASSERT_STATUS_OK(graph.Resolve());
}

std::unordered_map<std::string, OptimizerNodeConfig> GetOptInfoMap() {
  std::unordered_map<std::string, OptimizerNodeConfig> result{};
  std::transform(
      k_weight_names.begin(), k_weight_names.end(), std::inserter(result, result.end()),
      [](const std::string& weight_name) {
        return std::make_pair(
            weight_name, OptimizerNodeConfig{k_optimizer_op_name, nullptr, 1.0f, {}});
      });
  return result;
}

class OptimizerGraphBuilderTest : public testing::Test {
 protected:
  OptimizerGraphBuilderTest() : model_{"test_model"}, graph_{model_.MainGraph()} {
  }

  virtual void SetUp() override {
    SetUpBaseGraph(graph_);
  }

  Model model_;
  Graph& graph_;
};

OptimizerBuilderRegistry& GetOptimizerBuilderRegistry() {
  return OptimizerBuilderRegistry::GetInstance();
}

int GetOpCount(const std::map<std::string, int>& op_counts, const std::string& op_type) {
  auto op_count_it = op_counts.find(op_type);
  return op_count_it != op_counts.end() ? op_count_it->second : 0;
}
}  // namespace

TEST_F(OptimizerGraphBuilderTest, ConditionalOptimizer) {
  OptimizerGraphConfig opt_graph_config{};
  opt_graph_config.use_mixed_precision = true;

  OptimizerGraphBuilder optimizer_graph_builder{
      GetOptimizerBuilderRegistry(), opt_graph_config, GetOptInfoMap()};

  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph_));

  // verify that optimizers are in the If subgraph
  auto op_counts = CountOpsInGraph(graph_, false);
  ASSERT_EQ(GetOpCount(op_counts, "If"), 1);
  ASSERT_EQ(GetOpCount(op_counts, k_optimizer_op_name), 0);

  auto if_node_it = std::find_if(
      graph_.Nodes().begin(), graph_.Nodes().end(),
      [](Node& node) {
        return node.OpType() == "If";
      });
  ASSERT_NE(if_node_it, graph_.Nodes().end());

  Graph* then_subgraph = if_node_it->GetMutableGraphAttribute("then_branch");
  ASSERT_NE(then_subgraph, nullptr);

  auto then_subgraph_op_counts = CountOpsInGraph(*then_subgraph);
  ASSERT_EQ(GetOpCount(then_subgraph_op_counts, k_optimizer_op_name), k_weight_names.size());
}

TEST_F(OptimizerGraphBuilderTest, UnconditionalOptimizer) {
  OptimizerGraphConfig opt_graph_config{};
  opt_graph_config.use_mixed_precision = false;

  OptimizerGraphBuilder optimizer_graph_builder{
      GetOptimizerBuilderRegistry(), opt_graph_config, GetOptInfoMap()};

  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph_));

  // verify that optimizers are in the main graph
  auto op_counts = CountOpsInGraph(graph_, false);
  ASSERT_EQ(GetOpCount(op_counts, "If"), 0);
  ASSERT_EQ(GetOpCount(op_counts, k_optimizer_op_name), k_weight_names.size());
}

#ifdef USE_HOROVOD
TEST_F(OptimizerGraphBuilderTest, AllReduceNodeAdded) {
  OptimizerGraphConfig opt_graph_config{};
  opt_graph_config.world_size = 2;

  OptimizerGraphBuilder optimizer_graph_builder{
      GetOptimizerBuilderRegistry(), opt_graph_config, GetOptInfoMap()};

  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph_));

  // verify that AllReduce nodes are added
  auto op_counts = CountOpsInGraph(graph_);
  ASSERT_EQ(GetOpCount(op_counts, k_all_reduce_op_name), k_weight_names.size());
}
#endif  // USE_HOROVOD

TEST_F(OptimizerGraphBuilderTest, AllReduceNodeNotAdded) {
  OptimizerGraphConfig opt_graph_config{};
  opt_graph_config.world_size = 1;

  OptimizerGraphBuilder optimizer_graph_builder{
      GetOptimizerBuilderRegistry(), opt_graph_config, GetOptInfoMap()};

  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph_));

  // verify no AllReduce nodes are added
  auto op_counts = CountOpsInGraph(graph_);
  ASSERT_EQ(GetOpCount(op_counts, k_all_reduce_op_name), 0);
}
}  // namespace test
}  // namespace training
}  // namespace onnxruntime
