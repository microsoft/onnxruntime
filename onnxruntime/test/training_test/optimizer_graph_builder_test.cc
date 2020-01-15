// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <unordered_set>
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
#include "test/util/include/gtest_utils.h"
#include "test/test_environment.h"

using onnxruntime::test::CountOpsInGraph;

namespace onnxruntime {
namespace training {
namespace test {
namespace {
const std::vector<const char*> k_weight_names{"weight_1", "weight_2"};
constexpr const char* const k_loss_scaling_factor_name = "loss_scaling_factor";
constexpr const char* const k_optimizer_op_name = "SGDOptimizer";
constexpr const char* const k_all_reduce_op_name = "HorovodAllReduce";
constexpr const char* const k_is_all_finite_op_name = "IsAllFinite";
constexpr const char* const k_unscale_op_name = "MixedPrecisionScale";
constexpr const char* const k_gradient_accumulator_op_name = "GradientAccumulator";
constexpr const char* const k_zero_gradient_op_name = "ZeroGradient";

Status SetUpBaseGraph(Graph& graph);

class OptimizerGraphBuilderTest : public testing::Test {
 protected:
  OptimizerGraphBuilderTest() : model_{"test_model", false, onnxruntime::test::DefaultLoggingManager().DefaultLogger()},
   graph_{model_.MainGraph()} {
  }

  virtual void SetUp() override {
    ASSERT_STATUS_OK(SetUpBaseGraph(graph_));
  }

  Model model_;
  Graph& graph_;
};

// sets up a base graph with weight and gradient NodeArgs for each weight name and a loss scaling factor NodeArg
Status SetUpBaseGraph(Graph& graph) {
  ONNX_NAMESPACE::TypeProto float_tensor_type{};
  float_tensor_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  float_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  ONNX_NAMESPACE::TensorProto weight_initializer_base{};
  weight_initializer_base.add_dims(1);
  weight_initializer_base.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  weight_initializer_base.add_float_data(1.0f);

  ONNX_NAMESPACE::TensorProto weight_gradient_initializer_base{weight_initializer_base};
  weight_gradient_initializer_base.set_float_data(0, 2.0f);

  std::vector<const NodeArg*> all_nodeargs{};

  for (const auto& weight_name : k_weight_names) {
    auto& weight_arg = graph.GetOrCreateNodeArg(weight_name, &float_tensor_type);
    ONNX_NAMESPACE::TensorProto weight_initializer{weight_initializer_base};
    weight_initializer.set_name(weight_name);
    graph.AddInitializedTensor(weight_initializer);

    const std::string weight_gradient_name = GradientBuilderBase::GradientName(weight_name);
    auto& weight_grad_arg = graph.GetOrCreateNodeArg(weight_gradient_name, &float_tensor_type);
    ONNX_NAMESPACE::TensorProto weight_gradient_initializer{weight_gradient_initializer_base};
    weight_gradient_initializer.set_name(weight_gradient_name);
    graph.AddInitializedTensor(weight_gradient_initializer);

    all_nodeargs.emplace_back(&weight_arg);
    all_nodeargs.emplace_back(&weight_grad_arg);
  }

  auto& loss_scaling_factor_arg = graph.GetOrCreateNodeArg(k_loss_scaling_factor_name, &float_tensor_type);
  ONNX_NAMESPACE::TensorProto loss_scaling_factor_initializer{weight_initializer_base};
  loss_scaling_factor_initializer.set_name(k_loss_scaling_factor_name);
  loss_scaling_factor_initializer.set_float_data(0, 3.0f);
  graph.AddInitializedTensor(loss_scaling_factor_initializer);
  all_nodeargs.emplace_back(&loss_scaling_factor_arg);

  // make the values persist past Graph::Resolve()
  graph.SetInputs(all_nodeargs);
  graph.SetOutputs(all_nodeargs);

  return graph.Resolve();
}

std::unordered_map<std::string, OptimizerNodeConfig> GetOptInfoMap() {
  std::unordered_map<std::string, OptimizerNodeConfig> result{};
  std::transform(
      k_weight_names.begin(), k_weight_names.end(), std::inserter(result, result.end()),
      [](const std::string& weight_name) {
        return std::make_pair(
            weight_name, OptimizerNodeConfig{k_optimizer_op_name, nullptr, "Learning_Rate", {}});
      });
  return result;
}

OptimizerBuilderRegistry& GetOptimizerBuilderRegistry() {
  return OptimizerBuilderRegistry::GetInstance();
}

int GetOpCount(const std::map<std::string, int>& op_counts, const std::string& op_type) {
  auto op_count_it = op_counts.find(op_type);
  return op_count_it != op_counts.end() ? op_count_it->second : 0;
}

void TestOptimizersAndMixedPrecision(bool use_mixed_precision, bool use_loss_scaling_factor, Graph& graph) {
  OptimizerGraphConfig opt_graph_config{};
  opt_graph_config.use_mixed_precision = use_mixed_precision;
  opt_graph_config.loss_scale_input_name = use_loss_scaling_factor ? k_loss_scaling_factor_name : "";

  OptimizerGraphBuilder optimizer_graph_builder{
      GetOptimizerBuilderRegistry(), opt_graph_config, GetOptInfoMap()};

  std::unordered_map<std::string, std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, opt_initializer_names, opt_graph_outputs));

  auto op_counts = CountOpsInGraph(graph, false);

  if (use_mixed_precision) {
    // verify that optimizers are in the If then_branch subgraph
    // verify that nothing is in the If else_branch subgraph
    // verify that finite gradient checks are in the main graph
    // verify that gradient unscaling is in the main graph if using a loss scaling factor
    ASSERT_EQ(GetOpCount(op_counts, k_optimizer_op_name), k_weight_names.size());
    //TODO: enable this when AllIsFinite is introduced
    //ASSERT_EQ(GetOpCount(op_counts, k_is_finite_op_name), k_weight_names.size());

    // the scale for mixed precision is moved to optimizer, so this check is not needed.
    //ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), use_loss_scaling_factor ? k_weight_names.size() : 0);

    // TODO: Re-enable following code when condtional weight update is handeled by If Node
    /*
    ASSERT_EQ(GetOpCount(op_counts, "If"), 1);
    auto if_node_it = std::find_if(
        graph.Nodes().begin(), graph.Nodes().end(),
        [](Node& node) {
          return node.OpType() == "If";
        });
    ASSERT_NE(if_node_it, graph.Nodes().end());

    Graph* then_subgraph = if_node_it->GetMutableGraphAttribute("then_branch");
    ASSERT_NE(then_subgraph, nullptr);

    auto then_subgraph_op_counts = CountOpsInGraph(*then_subgraph);
    ASSERT_EQ(GetOpCount(then_subgraph_op_counts, k_optimizer_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(then_subgraph_op_counts, k_is_finite_op_name), 0);
    ASSERT_EQ(GetOpCount(then_subgraph_op_counts, k_unscale_op_name), 0);

    Graph* else_subgraph = if_node_it->GetMutableGraphAttribute("else_branch");
    ASSERT_NE(else_subgraph, nullptr);

    auto else_subgraph_op_counts = CountOpsInGraph(*else_subgraph);
    ASSERT_TRUE(else_subgraph_op_counts.empty());
    */
  } else {  // !use_mixed_precision
    // verify that optimizers are in the main graph
    // verify that gradient unscaling and finite gradient checks are not added
    ASSERT_EQ(GetOpCount(op_counts, "If"), 0);
    ASSERT_EQ(GetOpCount(op_counts, k_optimizer_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_is_all_finite_op_name), 0);
    ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), 0);
  }
}
}  // namespace

TEST_F(OptimizerGraphBuilderTest, OptimizersWithMixedPrecisionWithLossScaling) {
  TestOptimizersAndMixedPrecision(true, true, graph_);
}

TEST_F(OptimizerGraphBuilderTest, OptimizersWithMixedPrecisionWithoutLossScaling) {
  TestOptimizersAndMixedPrecision(true, false, graph_);
}

TEST_F(OptimizerGraphBuilderTest, OptimizersWithoutMixedPrecision) {
  TestOptimizersAndMixedPrecision(false, false, graph_);
}

TEST_F(OptimizerGraphBuilderTest, OptimizersWithGradientAccumulation) {
  OptimizerGraphConfig opt_graph_config{};
  opt_graph_config.gradient_accumulation_steps = 10;

  OptimizerGraphBuilder optimizer_graph_builder{
      GetOptimizerBuilderRegistry(), opt_graph_config, GetOptInfoMap()};

  std::unordered_map<std::string, std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph_, opt_initializer_names, opt_graph_outputs));

  // verify that gradient_accumulator and zero_gradient nodes are added
  auto op_counts = CountOpsInGraph(graph_, false);
  ASSERT_EQ(GetOpCount(op_counts, k_gradient_accumulator_op_name), k_weight_names.size());
  ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), k_weight_names.size());
  ASSERT_GT(opt_graph_outputs.count(kGradientAccumulationOutputKey), 0);
}

TEST_F(OptimizerGraphBuilderTest, OptimizersWithoutGradientAccumulation) {
  OptimizerGraphConfig opt_graph_config{};

  OptimizerGraphBuilder optimizer_graph_builder{
      GetOptimizerBuilderRegistry(), opt_graph_config, GetOptInfoMap()};

  std::unordered_map<std::string, std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph_, opt_initializer_names, opt_graph_outputs));

  // verify that gradient_accumulator and zero_gradient nodes are not added
  auto op_counts = CountOpsInGraph(graph_, false);
  ASSERT_EQ(GetOpCount(op_counts, k_gradient_accumulator_op_name), 0);
  ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), 0);
  ASSERT_EQ(opt_graph_outputs.count(kGradientAccumulationOutputKey), 0);
}

#ifdef USE_HOROVOD
TEST_F(OptimizerGraphBuilderTest, AllReduceNodeAdded) {
  OptimizerGraphConfig opt_graph_config{};
  opt_graph_config.world_size = 2;

  OptimizerGraphBuilder optimizer_graph_builder{
      GetOptimizerBuilderRegistry(), opt_graph_config, GetOptInfoMap()};

  std::unordered_map<std::string, std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph_, opt_initializer_names, opt_graph_outputs));

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

  std::unordered_map<std::string, std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph_, opt_initializer_names, opt_graph_outputs));

  // verify no AllReduce nodes are added
  auto op_counts = CountOpsInGraph(graph_);
  ASSERT_EQ(GetOpCount(op_counts, k_all_reduce_op_name), 0);
}
}  // namespace test
}  // namespace training
}  // namespace onnxruntime
