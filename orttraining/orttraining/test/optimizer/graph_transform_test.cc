// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "core/graph/model.h"

#include "test/framework/test_utils.h"
#include "test/test_environment.h"

#include "gtest/gtest.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/utils.h"
#include "core/optimizer/bias_gelu_fusion.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/dropout_elimination.h"
#include "orttraining/core/optimizer/gist_encode_decode.h"
#include "orttraining/core/optimizer/megatron_transformer.h"
#include "orttraining/core/optimizer/concat_replacement.h"
#include "orttraining/core/optimizer/batchnorm_replacement.h"
#include "orttraining/core/optimizer/localized_recompute.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/asserts.h"
#include "orttraining/test/optimizer/horizontal_parallel_test_utils.h"
#include "orttraining/core/session/training_session.h"

#include <random>

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

TEST_F(GraphTransformationTests, BatchNormReplacement) {
  Model model("BatchNormReplacement", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 14}, {"com.microsoft", 1}},
              {}, *logger_);
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // 1x3x3x3
  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  TypeProto scale_tensor_type;
  scale_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  scale_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  auto& input_X = graph.GetOrCreateNodeArg("X", &input_tensor_type);
  auto& input_scale = graph.GetOrCreateNodeArg("scale", &scale_tensor_type);
  auto& input_B = graph.GetOrCreateNodeArg("B", &scale_tensor_type);
  auto& input_mean = graph.GetOrCreateNodeArg("input_mean", &scale_tensor_type);
  auto& input_var = graph.GetOrCreateNodeArg("input_var", &scale_tensor_type);

  auto& output_Y = graph.GetOrCreateNodeArg("Y", &input_tensor_type);
  graph.AddNode("BN", "BatchNormalization", "", {&input_X, &input_scale, &input_B, &input_mean, &input_var}, {&output_Y});

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("BatchNormReplacement");
  rule_transformer_L1->Register(std::make_unique<BatchNormReplacement>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  
  ASSERT_TRUE(graph.NumberOfNodes() == 1);
  // Make sure that BN was updated to add outputs
  ASSERT_TRUE(graph.Nodes().begin()->MutableOutputDefs().size() == 5);
  ASSERT_TRUE(graph.Nodes().begin()->OpType().compare("BatchNormInternal") == 0);
}


TEST_F(GraphTransformationTests, BatchNormReplacementWithOptionalOutputPresentOpset14) {
  Model model("BatchNormReplacement", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 14}, {"com.microsoft", 1}},
              {}, *logger_);
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // 1x3x3x3
  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  TypeProto scale_tensor_type;
  scale_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  scale_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  auto& input_X = graph.GetOrCreateNodeArg("X", &input_tensor_type);
  auto& input_scale = graph.GetOrCreateNodeArg("scale", &scale_tensor_type);
  auto& input_B = graph.GetOrCreateNodeArg("B", &scale_tensor_type);
  auto& input_mean = graph.GetOrCreateNodeArg("input_mean", &scale_tensor_type);
  auto& input_var = graph.GetOrCreateNodeArg("input_var", &scale_tensor_type);

  auto& output_Y = graph.GetOrCreateNodeArg("Y", &input_tensor_type);
  auto& output_running_mean = graph.GetOrCreateNodeArg("running_mean", &scale_tensor_type);
  auto& output_running_var = graph.GetOrCreateNodeArg("running_var", &scale_tensor_type);
  auto& bn_node = graph.AddNode("BN", "BatchNormalization", "", {&input_X, &input_scale, &input_B, &input_mean, &input_var},
                                                {&output_Y, &output_running_mean, &output_running_var});
  bn_node.AddAttribute("training_mode", static_cast<int64_t>(1));

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("BatchNormReplacement");
  rule_transformer_L1->Register(std::make_unique<BatchNormReplacement>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  
  ASSERT_TRUE(graph.NumberOfNodes() == 1);
  // Make sure that BN was updated to add outputs
  ASSERT_TRUE(graph.Nodes().begin()->MutableOutputDefs().size() == 5);
  ASSERT_TRUE(graph.Nodes().begin()->OpType().compare("BatchNormInternal") == 0);
}


TEST_F(GraphTransformationTests, BatchNormReplacementWithOptionalOutputPresentOpset9) {
  Model model("BatchNormReplacement", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 9}, {"com.microsoft", 1}},
              {}, *logger_);
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  // 1x3x3x3
  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  TypeProto scale_tensor_type;
  scale_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  scale_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(3);

  auto& input_X = graph.GetOrCreateNodeArg("X", &input_tensor_type);
  auto& input_scale = graph.GetOrCreateNodeArg("scale", &scale_tensor_type);
  auto& input_B = graph.GetOrCreateNodeArg("B", &scale_tensor_type);
  auto& input_mean = graph.GetOrCreateNodeArg("input_mean", &scale_tensor_type);
  auto& input_var = graph.GetOrCreateNodeArg("input_var", &scale_tensor_type);

  auto& output_Y = graph.GetOrCreateNodeArg("Y", &input_tensor_type);
  auto& output_running_mean = graph.GetOrCreateNodeArg("running_mean", &scale_tensor_type);
  auto& output_running_var = graph.GetOrCreateNodeArg("running_var", &scale_tensor_type);
  auto& saved_mean = graph.GetOrCreateNodeArg("saved_mean", &scale_tensor_type);
  auto& saved_var = graph.GetOrCreateNodeArg("saved_var", &scale_tensor_type);
  graph.AddNode("BN", "BatchNormalization", "", {&input_X, &input_scale, &input_B, &input_mean, &input_var},
                                                {&output_Y, &output_running_mean, &output_running_var, &saved_mean, &saved_var});

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("BatchNormReplacement");
  rule_transformer_L1->Register(std::make_unique<BatchNormReplacement>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  
  ASSERT_TRUE(graph.NumberOfNodes() == 1);
  // Make sure that BN was updated to add outputs
  ASSERT_TRUE(graph.Nodes().begin()->MutableOutputDefs().size() == 5);
  ASSERT_TRUE(graph.Nodes().begin()->OpType().compare("BatchNormInternal") == 0);
}

TEST_F(GraphTransformationTests, DropoutWithZeroRatioElimination) {
  auto model_uri = MODEL_FOLDER "dropout_ratio.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 10);
  ASSERT_TRUE(op_to_count["Dropout"] == 5);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(std::make_unique<EliminateDropout>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);

  ASSERT_TRUE(op_to_count["Identity"] == 10);
  ASSERT_TRUE(op_to_count["Dropout"] == 2);
}

Node* GetNodeByName(Graph& graph, std::string node_name) {
  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  for (auto node_index : node_topology_list) {
    auto* p_node = graph.GetNode(node_index);
    if (p_node->Name().compare(node_name) == 0) {
      return p_node;
    }
  }

  return nullptr;
}

// MegatronF/G and ConcatTraining is defined only for training, and in msdomain.
#ifndef DISABLE_CONTRIB_OPS
TEST_F(GraphTransformationTests, ConcatReplacement) {
  auto model_uri = MODEL_FOLDER "concat_trainable.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("ConcatReplacement");
  rule_transformer_L1->Register(std::make_unique<ConcatReplacement>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["com.microsoft.ConcatTraining"], 1);
}

TEST_F(GraphTransformationTests, MegatronMLPPartitionRank0) {
  auto model_uri = MODEL_FOLDER "model_parallel/mlp_megatron_basic_test.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unordered_map<std::string, std::string> updated_weight_names;
  std::unordered_set<std::string> weights_to_train;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
  training::TrainingSession::OptimizerState init_optim_state;
  IExecutionProvider* e = TestCPUExecutionProvider();
  graph_transformation_mgr.Register(std::make_unique<MegatronTransformer>(0, 2, updated_weight_names, weights_to_train, weight_partition_info, init_optim_state, *e), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  auto model_uri2 = "mlp_megatron_basic_test_partition_rank0.onnx";
  Model::Save(*p_model, model_uri2);

  {
    std::vector<float> expected_value = {
        0.0f, 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f,
        0.16f, 0.17f, 0.18f, 0.19f, 0.20f, 0.21f, 0.22f, 0.23f,
        0.32f, 0.33f, 0.34f, 0.35f, 0.36f, 0.37f, 0.38f, 0.39f,
        0.48f, 0.49f, 0.50f, 0.51f, 0.52f, 0.53f, 0.54f, 0.55f};
    auto a1_weight_arg = GetNodeByName(graph, "matmul")->MutableInputDefs()[1];
    ORT_ENFORCE(a1_weight_arg != nullptr);
    std::vector<int64_t> expected_shape = {4, 8};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, a1_weight_arg, actual_val, actual_shape);
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));

    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {0.0f, 0.01f, 0.02f, 0.03f, 0.04f, 0.05f, 0.06f, 0.07f};
    auto input_arg = GetNodeByName(graph, "add")->MutableInputDefs()[1];
    ORT_ENFORCE(input_arg != nullptr);
    std::vector<int64_t> expected_shape = {8};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, input_arg, actual_val, actual_shape);
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));

    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {
        0.0f, 0.01f, 0.02f, 0.03f,
        0.04f, 0.05f, 0.06f, 0.07f,
        0.08f, 0.09f, 0.10f, 0.11f,
        0.12f, 0.13f, 0.14f, 0.15f,
        0.16f, 0.17f, 0.18f, 0.19f,
        0.20f, 0.21f, 0.22f, 0.23f,
        0.24f, 0.25f, 0.26f, 0.27f,
        0.28f, 0.29f, 0.30f, 0.31f};
    auto input_arg = GetNodeByName(graph, "matmul2")->MutableInputDefs()[1];
    ORT_ENFORCE(input_arg != nullptr);
    std::vector<int64_t> expected_shape = {8, 4};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, input_arg, actual_val, actual_shape);
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));

    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
}

TEST_F(GraphTransformationTests, MegatronMLPPartitionRank1) {
  auto model_uri = MODEL_FOLDER "model_parallel/mlp_megatron_basic_test.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unordered_map<std::string, std::string> updated_weight_names;
  std::unordered_set<std::string> weights_to_train;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
  training::TrainingSession::OptimizerState init_optim_state;
  IExecutionProvider* e = TestCPUExecutionProvider();
  graph_transformation_mgr.Register(std::make_unique<MegatronTransformer>(1, 2, updated_weight_names, weights_to_train, weight_partition_info, init_optim_state, *e), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  auto model_uri2 = "mlp_megatron_basic_test_partition_rank1.onnx";
  Model::Save(*p_model, model_uri2);

  {
    std::vector<float> expected_value = {
        0.08f, 0.09f, 0.10f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f,
        0.24f, 0.25f, 0.26f, 0.27f, 0.28f, 0.29f, 0.30f, 0.31f,
        0.40f, 0.41f, 0.42f, 0.43f, 0.44f, 0.45f, 0.46f, 0.47f,
        0.56f, 0.57f, 0.58f, 0.59f, 0.60f, 0.61f, 0.62f, 0.63f};
    auto a1_weight_arg = GetNodeByName(graph, "matmul")->MutableInputDefs()[1];
    ORT_ENFORCE(a1_weight_arg != nullptr);
    std::vector<int64_t> expected_shape = {4, 8};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, a1_weight_arg, actual_val, actual_shape);
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));

    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {0.08f, 0.09f, 0.1f, 0.11f, 0.12f, 0.13f, 0.14f, 0.15f};
    auto a1_bias_arg = GetNodeByName(graph, "add")->MutableInputDefs()[1];
    ORT_ENFORCE(a1_bias_arg != nullptr);
    std::vector<int64_t> expected_shape = {8};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, a1_bias_arg, actual_val, actual_shape);
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));

    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {
        0.32f, 0.33f, 0.34f, 0.35f,
        0.36f, 0.37f, 0.38f, 0.39f,
        0.40f, 0.41f, 0.42f, 0.43f,
        0.44f, 0.45f, 0.46f, 0.47f,
        0.48f, 0.49f, 0.50f, 0.51f,
        0.52f, 0.53f, 0.54f, 0.55f,
        0.56f, 0.57f, 0.58f, 0.59f,
        0.60f, 0.61f, 0.62f, 0.63f};
    auto input_arg = GetNodeByName(graph, "matmul2")->MutableInputDefs()[1];
    ORT_ENFORCE(input_arg != nullptr);
    std::vector<int64_t> expected_shape = {8, 4};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, input_arg, actual_val, actual_shape);
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));

    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
}

TEST_F(GraphTransformationTests, MegatronSelfAttentionPartitionRank0) {
  auto model_uri = MODEL_FOLDER "model_parallel/self_attention_megatron_basic_test.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unordered_map<std::string, std::string> updated_weight_names;
  std::unordered_set<std::string> weights_to_train;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
  training::TrainingSession::OptimizerState init_optim_state;
  IExecutionProvider* e = TestCPUExecutionProvider();
  graph_transformation_mgr.Register(std::make_unique<MegatronTransformer>(0, 2, updated_weight_names, weights_to_train, weight_partition_info, init_optim_state, *e), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  auto model_uri2 = "self_attention_megatron_basic_test_partition_rank0.onnx";
  Model::Save(*p_model, model_uri2);

  {
    std::vector<float> expected_value = {
        0.00f, 0.01f, 0.04f, 0.05f, 0.08f, 0.09f,
        0.12f, 0.13f, 0.16f, 0.17f, 0.20f, 0.21f,
        0.24f, 0.25f, 0.28f, 0.29f, 0.32f, 0.33f,
        0.36f, 0.37f, 0.40f, 0.41f, 0.44f, 0.45f};
    auto qkv_weight_arg = GetNodeByName(graph, "matmul1")->MutableInputDefs()[1];
    ORT_ENFORCE(qkv_weight_arg != nullptr);
    std::vector<int64_t> expected_shape = {4, 6};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, qkv_weight_arg, actual_val, actual_shape);
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {0.00f, 0.01f, 0.04f, 0.05f, 0.08f, 0.09f};
    auto qkv_bias_arg = GetNodeByName(graph, "add1")->MutableInputDefs()[1];
    ORT_ENFORCE(qkv_bias_arg != nullptr);
    std::vector<int64_t> expected_shape = {6};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, qkv_bias_arg, actual_val, actual_shape);
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {
        0.00f, 0.01f, 0.02f, 0.03f,
        0.04f, 0.05f, 0.06f, 0.07f};
    auto dense_weight_arg = GetNodeByName(graph, "matmul4")->MutableInputDefs()[1];
    ORT_ENFORCE(dense_weight_arg != nullptr);
    std::vector<int64_t> expected_shape = {2, 4};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, dense_weight_arg, actual_val, actual_shape);
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    Node* node = graph.GetNode(GetNodeByName(graph, "matmul1")->InputNodesBegin()->Index());
    ORT_ENFORCE(node->OpType().compare("MegatronF") == 0);
    node = graph.GetNode(GetNodeByName(graph, "add2")->InputNodesBegin()->Index());
    ORT_ENFORCE(node->OpType().compare("MegatronG") == 0);
  }
}

TEST_F(GraphTransformationTests, MegatronSelfAttentionPartitionRank1) {
  auto model_uri = MODEL_FOLDER "model_parallel/self_attention_megatron_basic_test.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unordered_map<std::string, std::string> updated_weight_names;
  std::unordered_set<std::string> weights_to_train;
  std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
  training::TrainingSession::OptimizerState init_optim_state;
  IExecutionProvider* e = TestCPUExecutionProvider();
  graph_transformation_mgr.Register(std::make_unique<MegatronTransformer>(1, 2, updated_weight_names, weights_to_train, weight_partition_info, init_optim_state, *e), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  auto model_uri2 = "self_attention_megatron_basic_test_partition_rank1.onnx";
  Model::Save(*p_model, model_uri2);

  {
    std::vector<float> expected_value = {
        0.02f, 0.03f, 0.06f, 0.07f, 0.10f, 0.11f,
        0.14f, 0.15f, 0.18f, 0.19f, 0.22f, 0.23f,
        0.26f, 0.27f, 0.30f, 0.31f, 0.34f, 0.35f,
        0.38f, 0.39f, 0.42f, 0.43f, 0.46f, 0.47f};
    auto qkv_weight_arg = GetNodeByName(graph, "matmul1")->MutableInputDefs()[1];
    ORT_ENFORCE(qkv_weight_arg != nullptr);
    std::vector<int64_t> expected_shape = {4, 6};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, qkv_weight_arg, actual_val, actual_shape);
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {0.02f, 0.03f, 0.06f, 0.07f, 0.10f, 0.11f};
    auto qkv_bias_arg = GetNodeByName(graph, "add1")->MutableInputDefs()[1];
    ORT_ENFORCE(qkv_bias_arg != nullptr);
    std::vector<int64_t> expected_shape = {6};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, qkv_bias_arg, actual_val, actual_shape);
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    std::vector<float> expected_value = {
        0.08f, 0.09f, 0.10f, 0.11f,
        0.12f, 0.13f, 0.14f, 0.15f};
    auto dense_weight_arg = GetNodeByName(graph, "matmul4")->MutableInputDefs()[1];
    ORT_ENFORCE(dense_weight_arg != nullptr);
    std::vector<int64_t> expected_shape = {2, 4};
    std::vector<float> actual_val;
    std::vector<int64_t> actual_shape;
    horizontal_parallel_test_utils::GetDataAndShapeFromTensorProto(graph, dense_weight_arg, actual_val, actual_shape);
    ASSERT_TRUE(std::equal(expected_shape.begin(), expected_shape.end(), actual_shape.begin()));
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_value, actual_val, false);
  }
  {
    Node* node = graph.GetNode(GetNodeByName(graph, "matmul1")->InputNodesBegin()->Index());
    ORT_ENFORCE(node->OpType().compare("MegatronF") == 0);
    node = graph.GetNode(GetNodeByName(graph, "add2")->InputNodesBegin()->Index());
    ORT_ENFORCE(node->OpType().compare("MegatronG") == 0);
  }
}

TEST_F(GraphTransformationTests, BiasGeluRecomputeTest) {
  auto model_uri = MODEL_FOLDER "fusion/bias_gelu_fusion_recompute.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::make_unique<GeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(std::make_unique<BiasGeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(std::make_unique<GeluRecompute>(), TransformerLevel::Level2);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.BiasGelu"] == 2);
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "com.microsoft.BiasGelu") {
      ASSERT_TRUE(node.InputDefs().size() == 2);
    }
  }
}

// We only tested on CUDA run.
#if defined(USE_CUDA)
static void RunPartitionCorrectnessTest(std::string model_path,
                                        const logging::Logger& logger,
                                        const int total_rank,
                                        std::vector<std::string> input_names,
                                        std::vector<std::vector<int64_t>> input_dims) {
  const PathString model_uri = ToPathString(model_path) + ORT_TSTR(".onnx");
  // const int total_rank = 4;
  std::vector<Graph*> graphs;
  std::vector<std::shared_ptr<Model>> p_models(total_rank);
  for (auto i = 0; i < total_rank; i++) {
    Status ret = Model::Load(model_uri, p_models[i], nullptr, logger);
    ORT_ENFORCE(ret.IsOK());
    Graph& graph = p_models[i]->MainGraph();
    onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
    std::unordered_map<std::string, std::string> updated_weight_names;
    std::unordered_set<std::string> weights_to_train;
    std::unordered_map<std::string, training::TrainingSession::PartitionInfo> weight_partition_info;
    training::TrainingSession::OptimizerState init_optim_state;
    IExecutionProvider* e = TestCPUExecutionProvider();
    graph_transformation_mgr.Register(std::make_unique<MegatronTransformer>(i, total_rank, updated_weight_names, weights_to_train, weight_partition_info, init_optim_state, *e), TransformerLevel::Level1);
    ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, logger);
    ORT_ENFORCE(ret.IsOK());
    graphs.push_back(&graph);
    auto model_uri2 = ToPathString(model_path) + ORT_TSTR("_partition_rank_") + ToPathString(std::to_string(i)) + ORT_TSTR(".onnx");
    Model::Save(*p_models[i], model_uri2);
  }

  onnxruntime::Model combine_model("combine_graph", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}, {kMSDomain, 1}}, {}, logger);
  auto& combine_graph = combine_model.MainGraph();
  auto ret = horizontal_parallel_test_utils::MergeGraphsOnAllWorkers(graphs, combine_graph);
  ORT_ENFORCE(ret.IsOK());
  auto model_uri2 = ToPathString(model_path) + ORT_TSTR("_partition_combine.onnx");
  Model::Save(combine_model, model_uri2);

  float scale = 1.f;
  float mean = 0.f;
  float seed = 123.f;

  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution{mean, scale};

  ORT_ENFORCE(input_names.size() == input_dims.size());
  NameMLValMap feeds;
  for (size_t i = 0; i < input_dims.size(); i++) {
    std::vector<int64_t> dims_X = input_dims[i];
    std::vector<float> values_X(TensorShape(dims_X).Size());
    std::for_each(values_X.begin(), values_X.end(),
                  [&generator, &distribution](float& value) { value = distribution(generator); });

    OrtValue ml_value;
    CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_X, values_X, &ml_value);
    feeds.insert(std::make_pair(input_names[i], ml_value));
  }

  std::vector<OrtValue> expected_ort_values;
  {
    SessionOptions so;
    so.session_logid = "RawGraphRun";

    InferenceSession session_object{so, GetEnvironment()};
    std::unique_ptr<IExecutionProvider> execution_provider = DefaultCudaExecutionProvider();
    EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

    Status st;
    ASSERT_TRUE((st = session_object.Load(model_uri)).IsOK()) << st;
    ASSERT_TRUE((st = session_object.Initialize()).IsOK()) << st;

    // prepare outputs
    std::vector<std::string> output_names;
    output_names.push_back("output");

    // Now run
    RunOptions run_options;
    run_options.training_mode = true;
    st = session_object.Run(run_options, feeds, output_names, &expected_ort_values);

    EXPECT_TRUE(st.IsOK());
  }

  std::vector<OrtValue> actual_ort_values;
  {
    SessionOptions so;
    so.session_logid = "SplitThenCombineRun";

    InferenceSession session_object{so, GetEnvironment()};
    std::unique_ptr<IExecutionProvider> execution_provider = DefaultCudaExecutionProvider();
    EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

    Status st;
    ASSERT_TRUE((st = session_object.Load(model_uri2)).IsOK()) << st;
    ASSERT_TRUE((st = session_object.Initialize()).IsOK()) << st;

    // prepare outputs
    std::vector<std::string> output_names;
    for (auto i = 0; i < total_rank; i++) {
      output_names.push_back("output_rank_" + std::to_string(i));
    }

    // Now run
    RunOptions run_options;
    run_options.training_mode = true;
    st = session_object.Run(run_options, feeds, output_names, &actual_ort_values);

    EXPECT_TRUE(st.IsOK());
  }

  auto& expected_val = expected_ort_values[0].Get<Tensor>();
  for (auto i = 0; i < total_rank; i++) {
    auto& actual_val = actual_ort_values[i].Get<Tensor>();
    horizontal_parallel_test_utils::VerifyOutputs(expected_val, actual_val, true);
    horizontal_parallel_test_utils::VerifyOutputs(expected_val, actual_val, false);
  }
}

TEST_F(GraphTransformationTests, MegatronMLPPartitionCorrectnessTest) {
  RunPartitionCorrectnessTest("testdata/transform/model_parallel/mlp_megatron_basic_test", *logger_, 4, {"input"}, {{8, 16, 4}});
}

TEST_F(GraphTransformationTests, MegatronBARTMLPPartitionCorrectnessTest) {
  RunPartitionCorrectnessTest("testdata/transform/model_parallel/bart_mlp_megatron_basic_test", *logger_, 4, {"input"}, {{8, 16, 4}});
}

TEST_F(GraphTransformationTests, MegatronSelfAttentionPartitionCorrectnessTest) {
  RunPartitionCorrectnessTest("testdata/transform/model_parallel/self_attention_megatron_basic_test", *logger_, 2, {"input", "mask"}, {{8, 16, 4}, {8, 1, 16, 16}});
}

TEST_F(GraphTransformationTests, MegatronBARTSelfAttentionPartitionCorrectnessTest) {
  RunPartitionCorrectnessTest("testdata/transform/model_parallel/bart_self_attention_megatron_basic_test", *logger_, 2, {"input"}, {{6, 8, 4}});
}
// end of USE_CUDA
#endif

// end of DISABLE_CONTRIB_OPS
#endif

}  // namespace test
}  // namespace onnxruntime
