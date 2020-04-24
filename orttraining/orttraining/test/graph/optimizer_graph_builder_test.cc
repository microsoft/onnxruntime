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
#include "core/util/math.h"
#include "orttraining/core/graph/gradient_builder_base.h"
#include "orttraining/core/graph/optimizer_builder.h"
#include "orttraining/core/graph/optimizer_graph_builder.h"
#include "orttraining/core/graph/allreduce_optimizer_graph_builder.h"
#include "orttraining/core/graph/adasum_optimizer_graph_builder.h"
#include "orttraining/core/graph/zero_optimizer_graph_builder.h"
#include "test/framework/test_utils.h"
#include "test/util/include/asserts.h"
#include "test/test_environment.h"

using onnxruntime::test::CountOpsInGraph;
using namespace onnxruntime::training;

namespace onnxruntime {
namespace test {
namespace {

const std::vector<const char*> k_weight_names{"W0", "W1", "W2", "W3", "W4"};
const std::vector<int64_t> k_weight_dims{160, 160, 80, 400, 80};
constexpr const char* const k_loss_scaling_factor_name = "loss_scaling_factor";
constexpr const char* const k_optimizer_op_name = "AdamOptimizer";
constexpr const char* const k_horovod_all_reduce_op_name = "HorovodAllReduce";
constexpr const char* const k_all_reduce_op_name = "NcclAllReduce";
constexpr const char* const k_all_gather_op_name = "NcclAllGather";
constexpr const char* const k_reduce_scatter_op_name = "NcclReduceScatter";
constexpr const char* const k_is_all_finite_op_name = "IsAllFinite";
constexpr const char* const k_gradient_norm_op_name = "ReduceAllL2";
constexpr const char* const k_unscale_op_name = "MixedPrecisionScale";
constexpr const char* const k_inplace_accumulator_op_name = "InPlaceAccumulator";
constexpr const char* const k_zero_gradient_op_name = "ZeroGradient";

static void AddFP32Initializer(Graph& graph, const std::string& name, int64_t dims) {
  ONNX_NAMESPACE::TypeProto tensor_type{};
  tensor_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dims);
  graph.GetOrCreateNodeArg(name, &tensor_type);

  ONNX_NAMESPACE::TensorProto initializer{};
  initializer.set_name(name);
  initializer.add_dims(dims);
  initializer.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  for (int64_t i = 0; i < dims; i++) {
    initializer.add_float_data(float(i));
  }
  graph.AddInitializedTensor(initializer);
}

static void AddFP16Initializer(Graph& graph, const std::string& name, int64_t dims) {
  ONNX_NAMESPACE::TypeProto tensor_type{};
  tensor_type.mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(dims);
  graph.GetOrCreateNodeArg(name, &tensor_type);

  ONNX_NAMESPACE::TensorProto initializer{};
  initializer.set_name(name);
  initializer.add_dims(dims);
  initializer.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
  for (int64_t i = 0; i < dims; i++) {
    initializer.add_int32_data(math::floatToHalf(float(i)));
  }
  graph.AddInitializedTensor(initializer);
}

// sets up a base graph with weight and gradient NodeArgs for each weight name
static std::shared_ptr<Model> CreateTestModel(OptimizerGraphConfig config) {
  auto model = std::make_shared<Model>("test_model", false, DefaultLoggingManager().DefaultLogger());
  Graph& graph = model->MainGraph();

  std::unordered_set<std::string> weight_and_gradient_names{};
  for (size_t i = 0; i < k_weight_names.size(); i++) {
    std::string weight_name = k_weight_names[i];
    std::string fp16_weight_name = weight_name + "_fp16";
    std::string gradient_name = GradientBuilderBase::GradientName(weight_name);
    int64_t dims = k_weight_dims[i];

    if (config.use_mixed_precision) {
      AddFP32Initializer(graph, weight_name, dims);
      AddFP16Initializer(graph, fp16_weight_name, dims);
      AddFP16Initializer(graph, gradient_name, dims);
      weight_and_gradient_names.emplace(weight_name);
      weight_and_gradient_names.emplace(fp16_weight_name);
      weight_and_gradient_names.emplace(gradient_name);
    } else {
      AddFP32Initializer(graph, weight_name, dims);
      AddFP32Initializer(graph, gradient_name, dims);
      weight_and_gradient_names.emplace(weight_name);
      weight_and_gradient_names.emplace(gradient_name);
    }
  }

  Graph::ResolveOptions resolve_options{};
  resolve_options.initializer_names_to_preserve = &weight_and_gradient_names;
  EXPECT_STATUS_OK(graph.Resolve(resolve_options));
  return model;
}

static std::unordered_map<std::string, OptimizerNodeConfig> GetOptInfoMap(Graph& graph, OptimizerGraphConfig config) {
  std::unordered_map<std::string, OptimizerNodeConfig> result{};
  for (const std::string& weight_name : k_weight_names) {
    if (config.use_mixed_precision) {
      const NodeArg* fp16_weight_arg = graph.GetNodeArg(weight_name + "_fp16");
      result[weight_name] = OptimizerNodeConfig{k_optimizer_op_name, fp16_weight_arg, "Learning_Rate", {}};
    } else {
      result[weight_name] = OptimizerNodeConfig{k_optimizer_op_name, nullptr, "Learning_Rate", {}};
    }
  }
  return result;
}

static OptimizerBuilderRegistry& GetOptimizerBuilderRegistry() {
  return OptimizerBuilderRegistry::GetInstance();
}

static int GetOpCount(const std::map<std::string, int>& op_counts, const std::string& op_type) {
  auto op_count_it = op_counts.find(op_type);
  return op_count_it != op_counts.end() ? op_count_it->second : 0;
}

}  // namespace

static void TestDefaultOptimizerGraphBuilder(OptimizerGraphConfig config) {
  auto model = CreateTestModel(config);
  Graph& graph = model->MainGraph();
  OptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap(graph, config));

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, opt_initializer_names, opt_graph_outputs));

  auto op_counts = CountOpsInGraph(graph, false);

  // verify gradient accumulation operations exist
  if (config.gradient_accumulation_steps > 1) {
    ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_inplace_accumulator_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), k_weight_names.size());
    ASSERT_GT(opt_graph_outputs.count(OptimizerOutputKey::GradientAccumulation), 0);
  }

  // verify mixed precision operations exist
  if (config.use_mixed_precision) {
    ASSERT_GT(GetOpCount(op_counts, k_gradient_norm_op_name), 0);
    ASSERT_GT(GetOpCount(op_counts, k_is_all_finite_op_name), 0);
  }

  // verify optimizers exist
  ASSERT_EQ(GetOpCount(op_counts, k_optimizer_op_name), k_weight_names.size());

  // verify distributed operations don't exist
  ASSERT_EQ(GetOpCount(op_counts, k_all_reduce_op_name), 0);
  ASSERT_EQ(GetOpCount(op_counts, k_reduce_scatter_op_name), 0);
  ASSERT_EQ(GetOpCount(op_counts, k_all_gather_op_name), 0);
  ASSERT_EQ(GetOpCount(op_counts, k_horovod_all_reduce_op_name), 0);
}

TEST(OptimizerGraphBuilderTest, Default_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestDefaultOptimizerGraphBuilder(config);
}

TEST(OptimizerGraphBuilderTest, Default_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestDefaultOptimizerGraphBuilder(config);
}

TEST(OptimizerGraphBuilderTest, Default_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestDefaultOptimizerGraphBuilder(config);
}

TEST(OptimizerGraphBuilderTest, Default_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestDefaultOptimizerGraphBuilder(config);
}

#if defined(USE_HOROVOD) || defined(USE_NCCL)
static void TestAllreduceOptimizerGraphBuilder(OptimizerGraphConfig config) {
  auto model = CreateTestModel(config);
  Graph& graph = model->MainGraph();
  AllreduceOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap(graph, config));

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, opt_initializer_names, opt_graph_outputs));

  auto op_counts = CountOpsInGraph(graph, false);

  // verify gradient accumulation operations exist
  if (config.gradient_accumulation_steps > 1) {
    ASSERT_GT(GetOpCount(op_counts, k_unscale_op_name), 0);
    ASSERT_EQ(GetOpCount(op_counts, k_inplace_accumulator_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), k_weight_names.size());
    ASSERT_GT(opt_graph_outputs.count(OptimizerOutputKey::GradientAccumulation), 0);
  }

  // verify mixed precision operations exist
  if (config.use_mixed_precision) {
    ASSERT_GT(GetOpCount(op_counts, k_gradient_norm_op_name), 0);
    ASSERT_GT(GetOpCount(op_counts, k_is_all_finite_op_name), 0);
  }

  // verify allreduce operations exist
  ASSERT_GT(GetOpCount(op_counts, k_unscale_op_name), 0);
  if (config.use_nccl) {
    ASSERT_GT(GetOpCount(op_counts, k_all_reduce_op_name), 0);
  } else {
    ASSERT_GT(GetOpCount(op_counts, k_horovod_all_reduce_op_name), 0);
  }

  // verify optimizers exist
  ASSERT_EQ(GetOpCount(op_counts, k_optimizer_op_name), k_weight_names.size());
}

static void TestAdasumOptimizerGraphBuilder(OptimizerGraphConfig config) {
  auto model = CreateTestModel(config);
  Graph& graph = model->MainGraph();
  AdasumOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap(graph, config));

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, opt_initializer_names, opt_graph_outputs));

  auto op_counts = CountOpsInGraph(graph, false);

  // verify gradient accumulation operations exist
  if (config.gradient_accumulation_steps > 1) {
    ASSERT_GT(GetOpCount(op_counts, k_unscale_op_name), 0);
    ASSERT_EQ(GetOpCount(op_counts, k_inplace_accumulator_op_name), k_weight_names.size() * 2);
    ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), k_weight_names.size());
    ASSERT_GT(opt_graph_outputs.count(OptimizerOutputKey::GradientAccumulation), 0);
  }

  // verify mixed precision operations exist
  if (config.use_mixed_precision) {
    ASSERT_GT(GetOpCount(op_counts, k_gradient_norm_op_name), 0);
    ASSERT_GT(GetOpCount(op_counts, k_is_all_finite_op_name), 0);
  }

  // verify allreduce operations exist
  ASSERT_GT(GetOpCount(op_counts, k_unscale_op_name), 0);
  ASSERT_GT(GetOpCount(op_counts, k_horovod_all_reduce_op_name), 0);

  // verify in place adder operations exist
  ASSERT_GT(GetOpCount(op_counts, k_inplace_accumulator_op_name), 0);

  // verify optimizers exist
  ASSERT_EQ(GetOpCount(op_counts, k_optimizer_op_name), k_weight_names.size());
}
#endif

#ifdef USE_HOROVOD
TEST(OptimizerGraphBuilderTest, Allreduce_Horovod_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config);
}

TEST(OptimizerGraphBuilderTest, Allreduce_Horovod_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config);
}

TEST(OptimizerGraphBuilderTest, Allreduce_Horovod_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config);
}

TEST(OptimizerGraphBuilderTest, Allreduce_Horovod_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config);
}
TEST(OptimizerGraphBuilderTest, Adasum_Horovod_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchical;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestAdasumOptimizerGraphBuilder(config);
}
TEST(OptimizerGraphBuilderTest, Adasum_Horovod_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchical;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestAdasumOptimizerGraphBuilder(config);
}

TEST(OptimizerGraphBuilderTest, Adasum_Horovod_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchical;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAdasumOptimizerGraphBuilder(config);
}

TEST(OptimizerGraphBuilderTest, Adasum_Horovod_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchical;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAdasumOptimizerGraphBuilder(config);
}

#endif  // USE_HOROVOD

#ifdef USE_NCCL
TEST(OptimizerGraphBuilderTest, Allreduce_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config);
}

TEST(OptimizerGraphBuilderTest, Allreduce_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config);
}

TEST(OptimizerGraphBuilderTest, Allreduce_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config);
}

TEST(OptimizerGraphBuilderTest, Allreduce_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config);
}

static void TestZeROOptimizerGraphBuilder(OptimizerGraphConfig config) {
  auto model = CreateTestModel(config);
  Graph& graph = model->MainGraph();
  ZeROOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap(graph, config));

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(graph, opt_initializer_names, opt_graph_outputs));

  auto op_counts = CountOpsInGraph(graph, false);

  // verify gradient accumulation operations exist
  if (config.gradient_accumulation_steps > 1) {
    ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_inplace_accumulator_op_name), k_weight_names.size());
    ASSERT_EQ(GetOpCount(op_counts, k_zero_gradient_op_name), k_weight_names.size());
    ASSERT_GT(opt_graph_outputs.count(OptimizerOutputKey::GradientAccumulation), 0);
  }

  // verify mixed precision operations exist
  if (config.use_mixed_precision) {
    ASSERT_GT(GetOpCount(op_counts, k_gradient_norm_op_name), 0);
    ASSERT_GT(GetOpCount(op_counts, k_is_all_finite_op_name), 0);
  }

  // verify ZeRO operations exist
  ASSERT_EQ(GetOpCount(op_counts, k_unscale_op_name), k_weight_names.size());
  ASSERT_GT(GetOpCount(op_counts, k_reduce_scatter_op_name), 0);
  ASSERT_GT(GetOpCount(op_counts, k_all_gather_op_name), 0);

  // verify optimizers exist
  ASSERT_GT(GetOpCount(op_counts, k_optimizer_op_name), 0);
  ASSERT_LT(GetOpCount(op_counts, k_optimizer_op_name), k_weight_names.size());
}

TEST(OptimizerGraphBuilderTest, ZeRO_NoGradientAccumulation_NoMixedPrecision) {
  for (int rank = 0; rank < 4; rank++) {
    OptimizerGraphConfig config;
    config.data_parallel_group_rank = rank;
    config.data_parallel_group_size = 4;
    config.use_nccl = true;
    config.partition_optimizer = true;
    config.gradient_accumulation_steps = 1;
    config.use_mixed_precision = false;
    TestZeROOptimizerGraphBuilder(config);
  }
}

TEST(OptimizerGraphBuilderTest, ZeRO_WithGradientAccumulation_NoMixedPrecision) {
  for (int rank = 0; rank < 4; rank++) {
    OptimizerGraphConfig config;
    config.data_parallel_group_rank = rank;
    config.data_parallel_group_size = 4;
    config.use_nccl = true;
    config.partition_optimizer = true;
    config.gradient_accumulation_steps = 10;
    config.use_mixed_precision = false;
    TestZeROOptimizerGraphBuilder(config);
  }
}

TEST(OptimizerGraphBuilderTest, ZeRO_NoGradientAccumulation_WithMixedPrecision) {
  for (int rank = 0; rank < 4; rank++) {
    OptimizerGraphConfig config;
    config.data_parallel_group_rank = rank;
    config.data_parallel_group_size = 4;
    config.use_nccl = true;
    config.partition_optimizer = true;
    config.gradient_accumulation_steps = 1;
    config.use_mixed_precision = true;
    config.loss_scale_input_name = k_loss_scaling_factor_name;
    TestZeROOptimizerGraphBuilder(config);
  }
}

TEST(OptimizerGraphBuilderTest, ZeRO_WithGradientAccumulation_WithMixedPrecision) {
  for (int rank = 0; rank < 4; rank++) {
    OptimizerGraphConfig config;
    config.data_parallel_group_rank = rank;
    config.data_parallel_group_size = 4;
    config.use_nccl = true;
    config.partition_optimizer = true;
    config.gradient_accumulation_steps = 10;
    config.use_mixed_precision = true;
    config.loss_scale_input_name = k_loss_scaling_factor_name;
    TestZeROOptimizerGraphBuilder(config);
  }
}

#endif  // USE_NCCL

}  // namespace test
}  // namespace onnxruntime
