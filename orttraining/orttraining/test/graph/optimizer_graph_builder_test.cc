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

namespace onnxruntime {
namespace training {
namespace test {
namespace {

const std::vector<const char*> k_weight_names{"weight_1", "weight_2"};
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

// sets up a base graph with weight and gradient NodeArgs for each weight name
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

  std::unordered_set<std::string> weight_and_gradient_names{};

  for (const auto& weight_name : k_weight_names) {
    graph.GetOrCreateNodeArg(weight_name, &float_tensor_type);
    ONNX_NAMESPACE::TensorProto weight_initializer{weight_initializer_base};
    weight_initializer.set_name(weight_name);
    graph.AddInitializedTensor(weight_initializer);

    const std::string weight_gradient_name = GradientBuilderBase::GradientName(weight_name);
    graph.GetOrCreateNodeArg(weight_gradient_name, &float_tensor_type);
    ONNX_NAMESPACE::TensorProto weight_gradient_initializer{weight_gradient_initializer_base};
    weight_gradient_initializer.set_name(weight_gradient_name);
    graph.AddInitializedTensor(weight_gradient_initializer);

    weight_and_gradient_names.emplace(weight_name);
    weight_and_gradient_names.emplace(weight_gradient_name);
  }

  Graph::ResolveOptions resolve_options{};
  resolve_options.initializer_names_to_preserve = &weight_and_gradient_names;
  return graph.Resolve(resolve_options);
}

Status SetUpBaseGraph(Graph& graph, const std::vector<const char*> weight_names) {
  // std::unordered_set<std::string> weight_and_gradient_names{};

  for (const auto& weight_name : weight_names) {
    NodeArg* weight_arg = graph.GetNodeArg(weight_name);
    // auto weight_def = weight_arg->TypeAsProto();

    const std::string weight_gradient_name = GradientBuilderBase::GradientName(weight_name);
    NodeArg* gradient_arg = graph.GetNodeArg(weight_gradient_name);
    gradient_arg->SetShape(*weight_arg->Shape());


    // weight_and_gradient_names.emplace(weight_name);
    // weight_and_gradient_names.emplace(weight_gradient_name);
  }

  // Graph::ResolveOptions resolve_options{};
  // resolve_options.initializer_names_to_preserve = &weight_and_gradient_names;
  // return graph.Resolve(resolve_options);
  return Status::OK();
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

std::unordered_map<std::string, OptimizerNodeConfig> GetOptInfoMap(const std::vector<const char*> weight_names) {
  std::unordered_map<std::string, OptimizerNodeConfig> result{};
  std::transform(
      weight_names.begin(), weight_names.end(), std::inserter(result, result.end()),
      [](const std::string& weight_name) {
        return std::make_pair(
            weight_name, OptimizerNodeConfig{k_optimizer_op_name, nullptr, "Learning_Rate", {}});
      });
  return result;
}

OptimizerBuilderRegistry& GetOptimizerBuilderRegistry() {
  return OptimizerBuilderRegistry::GetInstance();
}

static int GetOpCount(const std::map<std::string, int>& op_counts, const std::string& op_type) {
  static std::string ms_domain_prefix{std::string(kMSDomain) + '.'};

  auto op_count_it = op_counts.find(ms_domain_prefix + op_type);
  return op_count_it != op_counts.end() ? op_count_it->second : 0;
}

}  // namespace

static void TestDefaultOptimizerGraphBuilder(OptimizerGraphConfig config, Graph& graph) {
  std::unordered_map<std::string, std::string> updated_weight_names_map;
  OptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap(), updated_weight_names_map);

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

TEST_F(OptimizerGraphBuilderTest, Default_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestDefaultOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Default_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestDefaultOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Default_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestDefaultOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Default_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestDefaultOptimizerGraphBuilder(config, graph_);
}

#if defined(USE_NCCL) || defined(USE_HOROVOD)
static void TestAllreduceOptimizerGraphBuilder(OptimizerGraphConfig config, Graph& graph) {
  std::unordered_map<std::string, std::string> updated_weight_names_map;
  AllreduceOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap(),updated_weight_names_map);

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

#endif

#ifdef USE_HOROVOD
static void TestAdasumOptimizerGraphBuilder(OptimizerGraphConfig config, Graph& graph) {
  std::unordered_map<std::string, std::string> updated_weight_names_map;
  AdasumOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap(), updated_weight_names_map);

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

TEST_F(OptimizerGraphBuilderTest, Allreduce_Horovod_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_Horovod_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_Horovod_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_Horovod_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}
TEST_F(OptimizerGraphBuilderTest, Adasum_Horovod_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchical;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestAdasumOptimizerGraphBuilder(config, graph_);
}
TEST_F(OptimizerGraphBuilderTest, Adasum_Horovod_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchical;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestAdasumOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Adasum_Horovod_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchical;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAdasumOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Adasum_Horovod_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = false;
  config.adasum_reduction_type = AdasumReductionType::GpuHierarchical;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAdasumOptimizerGraphBuilder(config, graph_);
}

#endif  // USE_HOROVOD

#ifdef USE_NCCL
TEST_F(OptimizerGraphBuilderTest, Allreduce_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, Allreduce_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestAllreduceOptimizerGraphBuilder(config, graph_);
}

static void TestZeROOptimizerGraphBuilder(OptimizerGraphConfig config, Graph& graph) {
  std::unordered_map<std::string, std::string> updated_weight_names_map;
  ZeROOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), config, GetOptInfoMap(), updated_weight_names_map);

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
  ASSERT_EQ(GetOpCount(op_counts, k_optimizer_op_name), k_weight_names.size());
}

TEST_F(OptimizerGraphBuilderTest, ZeRO_NoGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.deepspeed_zero = ZeROConfig{0};
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = false;
  TestZeROOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, ZeRO_WithGradientAccumulation_NoMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.deepspeed_zero = ZeROConfig{0};
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = false;
  TestZeROOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, ZeRO_NoGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.deepspeed_zero = ZeROConfig{0};
  config.gradient_accumulation_steps = 1;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestZeROOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, ZeRO_WithGradientAccumulation_WithMixedPrecision) {
  OptimizerGraphConfig config;
  config.data_parallel_group_size = 4;
  config.use_nccl = true;
  config.deepspeed_zero = ZeROConfig{0};
  config.gradient_accumulation_steps = 10;
  config.use_mixed_precision = true;
  config.loss_scale_input_name = k_loss_scaling_factor_name;
  TestZeROOptimizerGraphBuilder(config, graph_);
}

TEST_F(OptimizerGraphBuilderTest, ZeRO_WithGradientAccumulation_WithMixedPrecision_BERT) {
  // const auto model_path = ORT_TSTR("testdata/bert_toy_optimized_bw.onnx");
     
  OptimizerGraphConfig optim_config;
  optim_config.data_parallel_group_size = 4;
  optim_config.use_nccl = true;
  optim_config.deepspeed_zero = ZeROConfig{0};
  optim_config.gradient_accumulation_steps = 1;
  optim_config.use_mixed_precision = true;
  optim_config.loss_scale_input_name = k_loss_scaling_factor_name;
 
  const auto model_path = ORT_TSTR("testdata/bert_toy_optimized_fp16_bw.onnx");
  // const std::vector<const char*> weight_names={"bert.embeddings.LayerNorm.bias", "bert.embeddings.LayerNorm.weight", "bert.embeddings.position_embeddings.weight", "bert.embeddings.token_type_embeddings.weight", "bert.embeddings.word_embeddings.weight", "bert.encoder.layer.0.attention.output.LayerNorm.bias", "bert.encoder.layer.0.attention.output.LayerNorm.weight", "bert.encoder.layer.0.attention.output.dense.bias", "bert.encoder.layer.0.attention.self.key.bias", "bert.encoder.layer.0.attention.self.query.bias", "bert.encoder.layer.0.attention.self.value.bias", "bert.encoder.layer.0.intermediate.dense.bias", "bert.encoder.layer.0.output.LayerNorm.bias", "bert.encoder.layer.0.output.LayerNorm.weight", "bert.encoder.layer.0.output.dense.bias", "bert.encoder.layer.1.attention.output.LayerNorm.bias", "bert.encoder.layer.1.attention.output.LayerNorm.weight", "bert.encoder.layer.1.attention.output.dense.bias", "bert.encoder.layer.1.attention.self.key.bias", "bert.encoder.layer.1.attention.self.query.bias", "bert.encoder.layer.1.attention.self.value.bias", "bert.encoder.layer.1.intermediate.dense.bias", "bert.encoder.layer.1.output.LayerNorm.bias", "bert.encoder.layer.1.output.LayerNorm.weight", "bert.encoder.layer.1.output.dense.bias", "bert.encoder.layer.2.attention.output.LayerNorm.bias", "bert.encoder.layer.2.attention.output.LayerNorm.weight", "bert.encoder.layer.2.attention.output.dense.bias", "bert.encoder.layer.2.attention.self.key.bias", "bert.encoder.layer.2.attention.self.query.bias", "bert.encoder.layer.2.attention.self.value.bias", "bert.encoder.layer.2.intermediate.dense.bias", "bert.encoder.layer.2.output.LayerNorm.bias", "bert.encoder.layer.2.output.LayerNorm.weight", "bert.encoder.layer.2.output.dense.bias", "bert.encoder.layer.3.attention.output.LayerNorm.bias", "bert.encoder.layer.3.attention.output.LayerNorm.weight", "bert.encoder.layer.3.attention.output.dense.bias", "bert.encoder.layer.3.attention.self.key.bias", "bert.encoder.layer.3.attention.self.query.bias", "bert.encoder.layer.3.attention.self.value.bias", "bert.encoder.layer.3.intermediate.dense.bias", "bert.encoder.layer.3.output.LayerNorm.bias", "bert.encoder.layer.3.output.LayerNorm.weight", "bert.encoder.layer.3.output.dense.bias", "bert.encoder.layer.4.attention.output.LayerNorm.bias", "bert.encoder.layer.4.attention.output.LayerNorm.weight", "bert.encoder.layer.4.attention.output.dense.bias", "bert.encoder.layer.4.attention.self.key.bias", "bert.encoder.layer.4.attention.self.query.bias", "bert.encoder.layer.4.attention.self.value.bias", "bert.encoder.layer.4.intermediate.dense.bias", "bert.encoder.layer.4.output.LayerNorm.bias", "bert.encoder.layer.4.output.LayerNorm.weight", "bert.encoder.layer.4.output.dense.bias", "bert.pooler.dense.bias", "bert.pooler.dense.weight", "cls.predictions.bias", "cls.predictions.transform.LayerNorm.bias", "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.dense.bias", "cls.seq_relationship.bias", "cls.seq_relationship.weight", "bert.encoder.layer.0.attention.self.query.weight_transposed", "bert.encoder.layer.0.attention.self.key.weight_transposed", "bert.encoder.layer.0.attention.self.value.weight_transposed", "bert.encoder.layer.0.attention.output.dense.weight_transposed", "bert.encoder.layer.0.intermediate.dense.weight_transposed", "bert.encoder.layer.0.output.dense.weight_transposed", "bert.encoder.layer.1.attention.self.query.weight_transposed", "bert.encoder.layer.1.attention.self.key.weight_transposed", "bert.encoder.layer.1.attention.self.value.weight_transposed", "bert.encoder.layer.1.attention.output.dense.weight_transposed", "bert.encoder.layer.1.intermediate.dense.weight_transposed", "bert.encoder.layer.1.output.dense.weight_transposed", "bert.encoder.layer.2.attention.self.query.weight_transposed", "bert.encoder.layer.2.attention.self.key.weight_transposed", "bert.encoder.layer.2.attention.self.value.weight_transposed", "bert.encoder.layer.2.attention.output.dense.weight_transposed", "bert.encoder.layer.2.intermediate.dense.weight_transposed", "bert.encoder.layer.2.output.dense.weight_transposed", "bert.encoder.layer.3.attention.self.query.weight_transposed", "bert.encoder.layer.3.attention.self.key.weight_transposed", "bert.encoder.layer.3.attention.self.value.weight_transposed", "bert.encoder.layer.3.attention.output.dense.weight_transposed", "bert.encoder.layer.3.intermediate.dense.weight_transposed", "bert.encoder.layer.3.output.dense.weight_transposed", "bert.encoder.layer.4.attention.self.query.weight_transposed", "bert.encoder.layer.4.attention.self.key.weight_transposed", "bert.encoder.layer.4.attention.self.value.weight_transposed", "bert.encoder.layer.4.attention.output.dense.weight_transposed", "bert.encoder.layer.4.intermediate.dense.weight_transposed", "bert.encoder.layer.4.output.dense.weight_transposed", "cls.predictions.transform.dense.weight_transposed"};
  const std::vector<const char*> weight_names={"bert.embeddings.LayerNorm.bias_fp16", "bert.embeddings.LayerNorm.weight_fp16", "bert.embeddings.position_embeddings.weight_fp16", "bert.embeddings.token_type_embeddings.weight_fp16", "bert.embeddings.word_embeddings.weight_fp16", "bert.encoder.layer.0.attention.output.LayerNorm.bias_fp16", "bert.encoder.layer.0.attention.output.LayerNorm.weight_fp16", "bert.encoder.layer.0.attention.output.dense.bias_fp16", "bert.encoder.layer.0.attention.self.key.bias_fp16", "bert.encoder.layer.0.attention.self.query.bias_fp16", "bert.encoder.layer.0.attention.self.value.bias_fp16", "bert.encoder.layer.0.intermediate.dense.bias_fp16", "bert.encoder.layer.0.output.LayerNorm.bias_fp16", "bert.encoder.layer.0.output.LayerNorm.weight_fp16", "bert.encoder.layer.0.output.dense.bias_fp16", "bert.encoder.layer.1.attention.output.LayerNorm.bias_fp16", "bert.encoder.layer.1.attention.output.LayerNorm.weight_fp16", "bert.encoder.layer.1.attention.output.dense.bias_fp16", "bert.encoder.layer.1.attention.self.key.bias_fp16", "bert.encoder.layer.1.attention.self.query.bias_fp16", "bert.encoder.layer.1.attention.self.value.bias_fp16", "bert.encoder.layer.1.intermediate.dense.bias_fp16", "bert.encoder.layer.1.output.LayerNorm.bias_fp16", "bert.encoder.layer.1.output.LayerNorm.weight_fp16", "bert.encoder.layer.1.output.dense.bias_fp16", "bert.encoder.layer.2.attention.output.LayerNorm.bias_fp16", "bert.encoder.layer.2.attention.output.LayerNorm.weight_fp16", "bert.encoder.layer.2.attention.output.dense.bias_fp16", "bert.encoder.layer.2.attention.self.key.bias_fp16", "bert.encoder.layer.2.attention.self.query.bias_fp16", "bert.encoder.layer.2.attention.self.value.bias_fp16", "bert.encoder.layer.2.intermediate.dense.bias_fp16", "bert.encoder.layer.2.output.LayerNorm.bias_fp16", "bert.encoder.layer.2.output.LayerNorm.weight_fp16", "bert.encoder.layer.2.output.dense.bias_fp16", "bert.encoder.layer.3.attention.output.LayerNorm.bias_fp16", "bert.encoder.layer.3.attention.output.LayerNorm.weight_fp16", "bert.encoder.layer.3.attention.output.dense.bias_fp16", "bert.encoder.layer.3.attention.self.key.bias_fp16", "bert.encoder.layer.3.attention.self.query.bias_fp16", "bert.encoder.layer.3.attention.self.value.bias_fp16", "bert.encoder.layer.3.intermediate.dense.bias_fp16", "bert.encoder.layer.3.output.LayerNorm.bias_fp16", "bert.encoder.layer.3.output.LayerNorm.weight_fp16", "bert.encoder.layer.3.output.dense.bias_fp16", "bert.encoder.layer.4.attention.output.LayerNorm.bias_fp16", "bert.encoder.layer.4.attention.output.LayerNorm.weight_fp16", "bert.encoder.layer.4.attention.output.dense.bias_fp16", "bert.encoder.layer.4.attention.self.key.bias_fp16", "bert.encoder.layer.4.attention.self.query.bias_fp16", "bert.encoder.layer.4.attention.self.value.bias_fp16", "bert.encoder.layer.4.intermediate.dense.bias_fp16", "bert.encoder.layer.4.output.LayerNorm.bias_fp16", "bert.encoder.layer.4.output.LayerNorm.weight_fp16", "bert.encoder.layer.4.output.dense.bias_fp16", "bert.pooler.dense.bias_fp16", "bert.pooler.dense.weight_fp16", "cls.predictions.bias_fp16", "cls.predictions.transform.LayerNorm.bias_fp16", "cls.predictions.transform.LayerNorm.weight_fp16", "cls.predictions.transform.dense.bias_fp16", "cls.seq_relationship.bias_fp16", "cls.seq_relationship.weight_fp16", "bert.encoder.layer.0.attention.self.query.weight_transposed_fp16", "bert.encoder.layer.0.attention.self.key.weight_transposed_fp16", "bert.encoder.layer.0.attention.self.value.weight_transposed_fp16", "bert.encoder.layer.0.attention.output.dense.weight_transposed_fp16", "bert.encoder.layer.0.intermediate.dense.weight_transposed_fp16", "bert.encoder.layer.0.output.dense.weight_transposed_fp16", "bert.encoder.layer.1.attention.self.query.weight_transposed_fp16", "bert.encoder.layer.1.attention.self.key.weight_transposed_fp16", "bert.encoder.layer.1.attention.self.value.weight_transposed_fp16", "bert.encoder.layer.1.attention.output.dense.weight_transposed_fp16", "bert.encoder.layer.1.intermediate.dense.weight_transposed_fp16", "bert.encoder.layer.1.output.dense.weight_transposed_fp16", "bert.encoder.layer.2.attention.self.query.weight_transposed_fp16", "bert.encoder.layer.2.attention.self.key.weight_transposed_fp16", "bert.encoder.layer.2.attention.self.value.weight_transposed_fp16", "bert.encoder.layer.2.attention.output.dense.weight_transposed_fp16", "bert.encoder.layer.2.intermediate.dense.weight_transposed_fp16", "bert.encoder.layer.2.output.dense.weight_transposed_fp16", "bert.encoder.layer.3.attention.self.query.weight_transposed_fp16", "bert.encoder.layer.3.attention.self.key.weight_transposed_fp16", "bert.encoder.layer.3.attention.self.value.weight_transposed_fp16", "bert.encoder.layer.3.attention.output.dense.weight_transposed_fp16", "bert.encoder.layer.3.intermediate.dense.weight_transposed_fp16", "bert.encoder.layer.3.output.dense.weight_transposed_fp16", "bert.encoder.layer.4.attention.self.query.weight_transposed_fp16", "bert.encoder.layer.4.attention.self.key.weight_transposed_fp16", "bert.encoder.layer.4.attention.self.value.weight_transposed_fp16", "bert.encoder.layer.4.attention.output.dense.weight_transposed_fp16", "bert.encoder.layer.4.intermediate.dense.weight_transposed_fp16", "bert.encoder.layer.4.output.dense.weight_transposed_fp16", "cls.predictions.transform.dense.weight_transposed_fp16"};
  std::shared_ptr<Model> loaded_model;
  EXPECT_TRUE(Model::Load(model_path, loaded_model, nullptr, onnxruntime::test::DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& bert_graph = loaded_model->MainGraph();

  SetUpBaseGraph(bert_graph, weight_names);

  std::unordered_map<std::string, std::string> updated_weight_names_map;
  ZeROOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), optim_config, GetOptInfoMap(weight_names), updated_weight_names_map);

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(bert_graph, opt_initializer_names, opt_graph_outputs));
  // Model::Save(*loaded_model, "testdata/bert_toy_optimized_bw_zero.onnx");
    Model::Save(*loaded_model, "testdata/bert_toy_optimized_fp16_bw_zero.onnx");
}

TEST_F(OptimizerGraphBuilderTest, ZeRO_WithGradientAccumulation_WithMixedPrecision_BERT_fp32) {
  const auto model_path = ORT_TSTR("testdata/bert_toy_optimized_bw.onnx");
     
  OptimizerGraphConfig optim_config;
  optim_config.data_parallel_group_size = 4;
  optim_config.use_nccl = true;
  optim_config.deepspeed_zero = ZeROConfig{0};
  optim_config.gradient_accumulation_steps = 1;
  optim_config.use_mixed_precision = true;
  optim_config.loss_scale_input_name = k_loss_scaling_factor_name;
 
  const std::vector<const char*> weight_names={"bert.embeddings.LayerNorm.bias", "bert.embeddings.LayerNorm.weight", "bert.embeddings.position_embeddings.weight", "bert.embeddings.token_type_embeddings.weight", "bert.embeddings.word_embeddings.weight", "bert.encoder.layer.0.attention.output.LayerNorm.bias", "bert.encoder.layer.0.attention.output.LayerNorm.weight", "bert.encoder.layer.0.attention.output.dense.bias", "bert.encoder.layer.0.attention.self.key.bias", "bert.encoder.layer.0.attention.self.query.bias", "bert.encoder.layer.0.attention.self.value.bias", "bert.encoder.layer.0.intermediate.dense.bias", "bert.encoder.layer.0.output.LayerNorm.bias", "bert.encoder.layer.0.output.LayerNorm.weight", "bert.encoder.layer.0.output.dense.bias", "bert.encoder.layer.1.attention.output.LayerNorm.bias", "bert.encoder.layer.1.attention.output.LayerNorm.weight", "bert.encoder.layer.1.attention.output.dense.bias", "bert.encoder.layer.1.attention.self.key.bias", "bert.encoder.layer.1.attention.self.query.bias", "bert.encoder.layer.1.attention.self.value.bias", "bert.encoder.layer.1.intermediate.dense.bias", "bert.encoder.layer.1.output.LayerNorm.bias", "bert.encoder.layer.1.output.LayerNorm.weight", "bert.encoder.layer.1.output.dense.bias", "bert.encoder.layer.2.attention.output.LayerNorm.bias", "bert.encoder.layer.2.attention.output.LayerNorm.weight", "bert.encoder.layer.2.attention.output.dense.bias", "bert.encoder.layer.2.attention.self.key.bias", "bert.encoder.layer.2.attention.self.query.bias", "bert.encoder.layer.2.attention.self.value.bias", "bert.encoder.layer.2.intermediate.dense.bias", "bert.encoder.layer.2.output.LayerNorm.bias", "bert.encoder.layer.2.output.LayerNorm.weight", "bert.encoder.layer.2.output.dense.bias", "bert.encoder.layer.3.attention.output.LayerNorm.bias", "bert.encoder.layer.3.attention.output.LayerNorm.weight", "bert.encoder.layer.3.attention.output.dense.bias", "bert.encoder.layer.3.attention.self.key.bias", "bert.encoder.layer.3.attention.self.query.bias", "bert.encoder.layer.3.attention.self.value.bias", "bert.encoder.layer.3.intermediate.dense.bias", "bert.encoder.layer.3.output.LayerNorm.bias", "bert.encoder.layer.3.output.LayerNorm.weight", "bert.encoder.layer.3.output.dense.bias", "bert.encoder.layer.4.attention.output.LayerNorm.bias", "bert.encoder.layer.4.attention.output.LayerNorm.weight", "bert.encoder.layer.4.attention.output.dense.bias", "bert.encoder.layer.4.attention.self.key.bias", "bert.encoder.layer.4.attention.self.query.bias", "bert.encoder.layer.4.attention.self.value.bias", "bert.encoder.layer.4.intermediate.dense.bias", "bert.encoder.layer.4.output.LayerNorm.bias", "bert.encoder.layer.4.output.LayerNorm.weight", "bert.encoder.layer.4.output.dense.bias", "bert.pooler.dense.bias", "bert.pooler.dense.weight", "cls.predictions.bias", "cls.predictions.transform.LayerNorm.bias", "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.dense.bias", "cls.seq_relationship.bias", "cls.seq_relationship.weight", "bert.encoder.layer.0.attention.self.query.weight_transposed", "bert.encoder.layer.0.attention.self.key.weight_transposed", "bert.encoder.layer.0.attention.self.value.weight_transposed", "bert.encoder.layer.0.attention.output.dense.weight_transposed", "bert.encoder.layer.0.intermediate.dense.weight_transposed", "bert.encoder.layer.0.output.dense.weight_transposed", "bert.encoder.layer.1.attention.self.query.weight_transposed", "bert.encoder.layer.1.attention.self.key.weight_transposed", "bert.encoder.layer.1.attention.self.value.weight_transposed", "bert.encoder.layer.1.attention.output.dense.weight_transposed", "bert.encoder.layer.1.intermediate.dense.weight_transposed", "bert.encoder.layer.1.output.dense.weight_transposed", "bert.encoder.layer.2.attention.self.query.weight_transposed", "bert.encoder.layer.2.attention.self.key.weight_transposed", "bert.encoder.layer.2.attention.self.value.weight_transposed", "bert.encoder.layer.2.attention.output.dense.weight_transposed", "bert.encoder.layer.2.intermediate.dense.weight_transposed", "bert.encoder.layer.2.output.dense.weight_transposed", "bert.encoder.layer.3.attention.self.query.weight_transposed", "bert.encoder.layer.3.attention.self.key.weight_transposed", "bert.encoder.layer.3.attention.self.value.weight_transposed", "bert.encoder.layer.3.attention.output.dense.weight_transposed", "bert.encoder.layer.3.intermediate.dense.weight_transposed", "bert.encoder.layer.3.output.dense.weight_transposed", "bert.encoder.layer.4.attention.self.query.weight_transposed", "bert.encoder.layer.4.attention.self.key.weight_transposed", "bert.encoder.layer.4.attention.self.value.weight_transposed", "bert.encoder.layer.4.attention.output.dense.weight_transposed", "bert.encoder.layer.4.intermediate.dense.weight_transposed", "bert.encoder.layer.4.output.dense.weight_transposed", "cls.predictions.transform.dense.weight_transposed"};
  std::shared_ptr<Model> loaded_model;
  EXPECT_TRUE(Model::Load(model_path, loaded_model, nullptr, onnxruntime::test::DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& bert_graph = loaded_model->MainGraph();

  SetUpBaseGraph(bert_graph, weight_names);

  std::unordered_map<std::string, std::string> updated_weight_names_map;
  ZeROOptimizerGraphBuilder optimizer_graph_builder(
      GetOptimizerBuilderRegistry(), optim_config, GetOptInfoMap(weight_names), updated_weight_names_map);

  OptimizerOutputKeyMap<std::string> opt_graph_outputs;
  std::unordered_set<std::string> opt_initializer_names;
  ASSERT_STATUS_OK(optimizer_graph_builder.Build(bert_graph, opt_initializer_names, opt_graph_outputs));
  Model::Save(*loaded_model, "testdata/bert_toy_optimized_bw_zero.onnx");
}

// TEST_F(OptimizerGraphBuilderTest, ZeRO_WithGradientAccumulation_WithMixedPrecision_BERT_alt) {
//   // const auto model_path = ORT_TSTR("testdata/bert_toy_optimized_bw.onnx");
     
//   OptimizerGraphConfig optim_config;
//   optim_config.data_parallel_group_size = 4;
//   optim_config.use_nccl = true;
//   optim_config.deepspeed_zero = ZeROConfig{0};
//   optim_config.gradient_accumulation_steps = 10;
//   optim_config.use_mixed_precision = true;
//   optim_config.loss_scale_input_name = k_loss_scaling_factor_name;
//   TrainingSession::TrainingConfiguration config{};
//   config.model_with_training_graph_path = ORT_TSTR("testdata/bert_toy_optimized_fp16_bw_zero.onnx");
//   config.loss_function_config = TrainingSession::TrainingConfiguration::LossFunctionConfiguration{};
  // config.loss_function_config.value().loss_function_info =
  //     LossFunctionInfo(OpDef("BertLoss", kOnnxDomain),
  //                      "total_loss",
  //                      {/*prediction_masked_lm*/ "prediction_scores",
  //                       /*prediction_next_sentence*/ "seq_relationship_score",
  //                       /*masked_lm_positions*/ "masked_lm_positions",
  //                       /*masked_lm_ids*/ "masked_lm_ids",
  //                       /*next_sentence_labels*/ "next_sentence_labels",
  //                       /*mlm_loss*/ "mlm_loss",
  //                       /*nsp_loss*/ "nsp_loss"});
//   config.weight_names_to_not_train = {
//       "position_01",            // Slice's dat input
//       "op_min_ends_expand_10",  //op_min_ends_expand_10
//   };
//   config.immutable_weights = {
//       {"Div", {{1, 8.0f}, {1, 1.4142135381698608f}}},
//       {"Add", {{1, 1.0f}, {1, 9.999999960041972e-13f}}},
//       {"Mul", {{1, 0.5f}, {1, -10000.0f}}},
//       {"Sub", {{0, 1.0f}}}};
  
//   TrainingSession::TrainingConfiguration::MixedPrecisionConfiguration mixed_precision_config{};
//   mixed_precision_config.use_mixed_precision_initializers = true;
//   config.mixed_precision_config = mixed_precision_config;

//   TrainingSession::TrainingConfiguration::OptimizerConfiguration opt{};
//   opt.name = parameters.training_optimizer_name;
//   opt.learning_rate_input_name = parameters.lr_params_feed_name;
//   opt.use_mixed_precision_moments = false;
//   opt.do_all_reduce_in_mixed_precision_type = true;
//   opt.use_nccl = true;
//   opt.deepspeed_zero = onnxruntime::training::ZeROConfig(0);
  
  

//   config.optimizer_config = opt;

  
//   const auto model_path = ORT_TSTR("testdata/bert_toy_optimized_fp16_bw.onnx");
//   // const std::vector<const char*> weight_names={"bert.embeddings.LayerNorm.bias", "bert.embeddings.LayerNorm.weight", "bert.embeddings.position_embeddings.weight", "bert.embeddings.token_type_embeddings.weight", "bert.embeddings.word_embeddings.weight", "bert.encoder.layer.0.attention.output.LayerNorm.bias", "bert.encoder.layer.0.attention.output.LayerNorm.weight", "bert.encoder.layer.0.attention.output.dense.bias", "bert.encoder.layer.0.attention.self.key.bias", "bert.encoder.layer.0.attention.self.query.bias", "bert.encoder.layer.0.attention.self.value.bias", "bert.encoder.layer.0.intermediate.dense.bias", "bert.encoder.layer.0.output.LayerNorm.bias", "bert.encoder.layer.0.output.LayerNorm.weight", "bert.encoder.layer.0.output.dense.bias", "bert.encoder.layer.1.attention.output.LayerNorm.bias", "bert.encoder.layer.1.attention.output.LayerNorm.weight", "bert.encoder.layer.1.attention.output.dense.bias", "bert.encoder.layer.1.attention.self.key.bias", "bert.encoder.layer.1.attention.self.query.bias", "bert.encoder.layer.1.attention.self.value.bias", "bert.encoder.layer.1.intermediate.dense.bias", "bert.encoder.layer.1.output.LayerNorm.bias", "bert.encoder.layer.1.output.LayerNorm.weight", "bert.encoder.layer.1.output.dense.bias", "bert.encoder.layer.2.attention.output.LayerNorm.bias", "bert.encoder.layer.2.attention.output.LayerNorm.weight", "bert.encoder.layer.2.attention.output.dense.bias", "bert.encoder.layer.2.attention.self.key.bias", "bert.encoder.layer.2.attention.self.query.bias", "bert.encoder.layer.2.attention.self.value.bias", "bert.encoder.layer.2.intermediate.dense.bias", "bert.encoder.layer.2.output.LayerNorm.bias", "bert.encoder.layer.2.output.LayerNorm.weight", "bert.encoder.layer.2.output.dense.bias", "bert.encoder.layer.3.attention.output.LayerNorm.bias", "bert.encoder.layer.3.attention.output.LayerNorm.weight", "bert.encoder.layer.3.attention.output.dense.bias", "bert.encoder.layer.3.attention.self.key.bias", "bert.encoder.layer.3.attention.self.query.bias", "bert.encoder.layer.3.attention.self.value.bias", "bert.encoder.layer.3.intermediate.dense.bias", "bert.encoder.layer.3.output.LayerNorm.bias", "bert.encoder.layer.3.output.LayerNorm.weight", "bert.encoder.layer.3.output.dense.bias", "bert.encoder.layer.4.attention.output.LayerNorm.bias", "bert.encoder.layer.4.attention.output.LayerNorm.weight", "bert.encoder.layer.4.attention.output.dense.bias", "bert.encoder.layer.4.attention.self.key.bias", "bert.encoder.layer.4.attention.self.query.bias", "bert.encoder.layer.4.attention.self.value.bias", "bert.encoder.layer.4.intermediate.dense.bias", "bert.encoder.layer.4.output.LayerNorm.bias", "bert.encoder.layer.4.output.LayerNorm.weight", "bert.encoder.layer.4.output.dense.bias", "bert.pooler.dense.bias", "bert.pooler.dense.weight", "cls.predictions.bias", "cls.predictions.transform.LayerNorm.bias", "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.dense.bias", "cls.seq_relationship.bias", "cls.seq_relationship.weight", "bert.encoder.layer.0.attention.self.query.weight_transposed", "bert.encoder.layer.0.attention.self.key.weight_transposed", "bert.encoder.layer.0.attention.self.value.weight_transposed", "bert.encoder.layer.0.attention.output.dense.weight_transposed", "bert.encoder.layer.0.intermediate.dense.weight_transposed", "bert.encoder.layer.0.output.dense.weight_transposed", "bert.encoder.layer.1.attention.self.query.weight_transposed", "bert.encoder.layer.1.attention.self.key.weight_transposed", "bert.encoder.layer.1.attention.self.value.weight_transposed", "bert.encoder.layer.1.attention.output.dense.weight_transposed", "bert.encoder.layer.1.intermediate.dense.weight_transposed", "bert.encoder.layer.1.output.dense.weight_transposed", "bert.encoder.layer.2.attention.self.query.weight_transposed", "bert.encoder.layer.2.attention.self.key.weight_transposed", "bert.encoder.layer.2.attention.self.value.weight_transposed", "bert.encoder.layer.2.attention.output.dense.weight_transposed", "bert.encoder.layer.2.intermediate.dense.weight_transposed", "bert.encoder.layer.2.output.dense.weight_transposed", "bert.encoder.layer.3.attention.self.query.weight_transposed", "bert.encoder.layer.3.attention.self.key.weight_transposed", "bert.encoder.layer.3.attention.self.value.weight_transposed", "bert.encoder.layer.3.attention.output.dense.weight_transposed", "bert.encoder.layer.3.intermediate.dense.weight_transposed", "bert.encoder.layer.3.output.dense.weight_transposed", "bert.encoder.layer.4.attention.self.query.weight_transposed", "bert.encoder.layer.4.attention.self.key.weight_transposed", "bert.encoder.layer.4.attention.self.value.weight_transposed", "bert.encoder.layer.4.attention.output.dense.weight_transposed", "bert.encoder.layer.4.intermediate.dense.weight_transposed", "bert.encoder.layer.4.output.dense.weight_transposed", "cls.predictions.transform.dense.weight_transposed"};
//   const std::vector<const char*> weight_names={"bert.embeddings.LayerNorm.bias_fp16", "bert.embeddings.LayerNorm.weight_fp16", "bert.embeddings.position_embeddings.weight_fp16", "bert.embeddings.token_type_embeddings.weight_fp16", "bert.embeddings.word_embeddings.weight_fp16", "bert.encoder.layer.0.attention.output.LayerNorm.bias_fp16", "bert.encoder.layer.0.attention.output.LayerNorm.weight_fp16", "bert.encoder.layer.0.attention.output.dense.bias_fp16", "bert.encoder.layer.0.attention.self.key.bias_fp16", "bert.encoder.layer.0.attention.self.query.bias_fp16", "bert.encoder.layer.0.attention.self.value.bias_fp16", "bert.encoder.layer.0.intermediate.dense.bias_fp16", "bert.encoder.layer.0.output.LayerNorm.bias_fp16", "bert.encoder.layer.0.output.LayerNorm.weight_fp16", "bert.encoder.layer.0.output.dense.bias_fp16", "bert.encoder.layer.1.attention.output.LayerNorm.bias_fp16", "bert.encoder.layer.1.attention.output.LayerNorm.weight_fp16", "bert.encoder.layer.1.attention.output.dense.bias_fp16", "bert.encoder.layer.1.attention.self.key.bias_fp16", "bert.encoder.layer.1.attention.self.query.bias_fp16", "bert.encoder.layer.1.attention.self.value.bias_fp16", "bert.encoder.layer.1.intermediate.dense.bias_fp16", "bert.encoder.layer.1.output.LayerNorm.bias_fp16", "bert.encoder.layer.1.output.LayerNorm.weight_fp16", "bert.encoder.layer.1.output.dense.bias_fp16", "bert.encoder.layer.2.attention.output.LayerNorm.bias_fp16", "bert.encoder.layer.2.attention.output.LayerNorm.weight_fp16", "bert.encoder.layer.2.attention.output.dense.bias_fp16", "bert.encoder.layer.2.attention.self.key.bias_fp16", "bert.encoder.layer.2.attention.self.query.bias_fp16", "bert.encoder.layer.2.attention.self.value.bias_fp16", "bert.encoder.layer.2.intermediate.dense.bias_fp16", "bert.encoder.layer.2.output.LayerNorm.bias_fp16", "bert.encoder.layer.2.output.LayerNorm.weight_fp16", "bert.encoder.layer.2.output.dense.bias_fp16", "bert.encoder.layer.3.attention.output.LayerNorm.bias_fp16", "bert.encoder.layer.3.attention.output.LayerNorm.weight_fp16", "bert.encoder.layer.3.attention.output.dense.bias_fp16", "bert.encoder.layer.3.attention.self.key.bias_fp16", "bert.encoder.layer.3.attention.self.query.bias_fp16", "bert.encoder.layer.3.attention.self.value.bias_fp16", "bert.encoder.layer.3.intermediate.dense.bias_fp16", "bert.encoder.layer.3.output.LayerNorm.bias_fp16", "bert.encoder.layer.3.output.LayerNorm.weight_fp16", "bert.encoder.layer.3.output.dense.bias_fp16", "bert.encoder.layer.4.attention.output.LayerNorm.bias_fp16", "bert.encoder.layer.4.attention.output.LayerNorm.weight_fp16", "bert.encoder.layer.4.attention.output.dense.bias_fp16", "bert.encoder.layer.4.attention.self.key.bias_fp16", "bert.encoder.layer.4.attention.self.query.bias_fp16", "bert.encoder.layer.4.attention.self.value.bias_fp16", "bert.encoder.layer.4.intermediate.dense.bias_fp16", "bert.encoder.layer.4.output.LayerNorm.bias_fp16", "bert.encoder.layer.4.output.LayerNorm.weight_fp16", "bert.encoder.layer.4.output.dense.bias_fp16", "bert.pooler.dense.bias_fp16", "bert.pooler.dense.weight_fp16", "cls.predictions.bias_fp16", "cls.predictions.transform.LayerNorm.bias_fp16", "cls.predictions.transform.LayerNorm.weight_fp16", "cls.predictions.transform.dense.bias_fp16", "cls.seq_relationship.bias_fp16", "cls.seq_relationship.weight_fp16", "bert.encoder.layer.0.attention.self.query.weight_transposed_fp16", "bert.encoder.layer.0.attention.self.key.weight_transposed_fp16", "bert.encoder.layer.0.attention.self.value.weight_transposed_fp16", "bert.encoder.layer.0.attention.output.dense.weight_transposed_fp16", "bert.encoder.layer.0.intermediate.dense.weight_transposed_fp16", "bert.encoder.layer.0.output.dense.weight_transposed_fp16", "bert.encoder.layer.1.attention.self.query.weight_transposed_fp16", "bert.encoder.layer.1.attention.self.key.weight_transposed_fp16", "bert.encoder.layer.1.attention.self.value.weight_transposed_fp16", "bert.encoder.layer.1.attention.output.dense.weight_transposed_fp16", "bert.encoder.layer.1.intermediate.dense.weight_transposed_fp16", "bert.encoder.layer.1.output.dense.weight_transposed_fp16", "bert.encoder.layer.2.attention.self.query.weight_transposed_fp16", "bert.encoder.layer.2.attention.self.key.weight_transposed_fp16", "bert.encoder.layer.2.attention.self.value.weight_transposed_fp16", "bert.encoder.layer.2.attention.output.dense.weight_transposed_fp16", "bert.encoder.layer.2.intermediate.dense.weight_transposed_fp16", "bert.encoder.layer.2.output.dense.weight_transposed_fp16", "bert.encoder.layer.3.attention.self.query.weight_transposed_fp16", "bert.encoder.layer.3.attention.self.key.weight_transposed_fp16", "bert.encoder.layer.3.attention.self.value.weight_transposed_fp16", "bert.encoder.layer.3.attention.output.dense.weight_transposed_fp16", "bert.encoder.layer.3.intermediate.dense.weight_transposed_fp16", "bert.encoder.layer.3.output.dense.weight_transposed_fp16", "bert.encoder.layer.4.attention.self.query.weight_transposed_fp16", "bert.encoder.layer.4.attention.self.key.weight_transposed_fp16", "bert.encoder.layer.4.attention.self.value.weight_transposed_fp16", "bert.encoder.layer.4.attention.output.dense.weight_transposed_fp16", "bert.encoder.layer.4.intermediate.dense.weight_transposed_fp16", "bert.encoder.layer.4.output.dense.weight_transposed_fp16", "cls.predictions.transform.dense.weight_transposed_fp16"};
//   std::shared_ptr<Model> loaded_model;
//   EXPECT_TRUE(Model::Load(model_path, loaded_model, nullptr, onnxruntime::test::DefaultLoggingManager().DefaultLogger()).IsOK());
//   Graph& bert_graph = loaded_model->MainGraph();

//   SetUpBaseGraph(bert_graph, weight_names);

//   ZeROOptimizerGraphBuilder optimizer_graph_builder(
//       GetOptimizerBuilderRegistry(), optim_config, GetOptInfoMap(weight_names));

//   OptimizerOutputKeyMap<std::string> opt_graph_outputs;
//   std::unordered_set<std::string> opt_initializer_names;
//   ASSERT_STATUS_OK(optimizer_graph_builder.Build(bert_graph, opt_initializer_names, opt_graph_outputs));
//   // Model::Save(*loaded_model, "testdata/bert_toy_optimized_bw_zero.onnx");
//     Model::Save(*loaded_model, "testdata/bert_toy_optimized_fp16_bw_zero.onnx");
  

//   // std::unique_ptr<Environment> env;
//   // ORT_RETURN_IF_ERROR(Environment::Create(nullptr, env));

//   // SessionOptions so{};
//   // TrainingSession training_session{so, *env};

//   // std::cout << "Loading source model file = " << ToMBString(model_path) << "\n";

//   // ORT_RETURN_IF_ERROR(training_session.Load(model_path));

//   // TrainingSession::TrainingConfigurationResult config_result{};
//   // ORT_RETURN_IF_ERROR(training_session.ConfigureForTraining(config, config_result));

// }

#endif  // USE_NCCL

}  // namespace test
}  // namespace training
}  // namespace onnxruntime
