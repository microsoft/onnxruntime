// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "core/graph/model.h"

#include "test/framework/test_utils.h"
#include "test/test_environment.h"

#include "gtest/gtest.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/utils.h"
#include "orttraining/core/optimizer/bias_dropout_fusion.h"
#include "orttraining/core/optimizer/gist_encode_decode.h"
#include "orttraining/core/optimizer/nonzero_shape_setter.h"
#include "orttraining/core/optimizer/megatron_transformer.h"
#include "orttraining/core/optimizer/concat_replacement.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/asserts.h"
#include "orttraining/test/optimizer/horizontal_parallel_test_utils.h"

#include <random>

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

TEST_F(GraphTransformationTests, GistEncodeDecode) {
  auto model_uri = MODEL_FOLDER "../test_training_model.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleGistTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<GistEncodeDecode>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["GistBinarizeEncoder"] == op_to_count["GistBinarizeEncoder"]);
}

static void TestBiasDropoutFusion(const PathString& file_path, const logging::Logger& logger, const int add_count = 0) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<BiasDropoutFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, logger);
  ASSERT_STATUS_OK(ret);

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  ASSERT_EQ(op_to_count["Add"], add_count);
  ASSERT_EQ(op_to_count["Dropout"], 0);
  ASSERT_EQ(op_to_count["TrainableDropout"], 0);
  ASSERT_EQ(op_to_count["BiasDropout"], 1);
}

TEST_F(GraphTransformationTests, BiasDropoutFusionTest) {
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_fusion1.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_fusion2.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion1.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion2.onnx", *logger_);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_dropout_residual_fusion_mismatch.onnx", *logger_, 1);
  TestBiasDropoutFusion(MODEL_FOLDER "fusion/bias_trainabledropout_residual_fusion.onnx", *logger_);
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

TEST_F(GraphTransformationTests, NonZeroShapeSetter) {
  auto model_uri = MODEL_FOLDER "nonzero_shape_setter.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("NonZeroShapeSetter1");
  rule_transformer_L1->Register(onnxruntime::make_unique<NonZeroShapeSetter>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  auto nonzero_shape = GetNodeByName(graph, "nonzero")->OutputDefs()[0]->Shape();
  ASSERT_TRUE(nonzero_shape->dim_size() == 2);
  ASSERT_TRUE(nonzero_shape->dim(0).dim_value() == 2);
  ASSERT_TRUE(nonzero_shape->dim(1).dim_param() == "nonzero_nonzero_count");
}

// MegatronF/G and ConcatTraining is defined only for training, and in msdomain.
#ifndef DISABLE_CONTRIB_OPS
TEST_F(GraphTransformationTests, ConcatReplacement) {
  auto model_uri = MODEL_FOLDER "concat_trainable.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("ConcatReplacement");
  rule_transformer_L1->Register(onnxruntime::make_unique<ConcatReplacement>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["ConcatTraining"], 1);
}

TEST_F(GraphTransformationTests, MegatronMLPPartitionRank0) {
  auto model_uri = MODEL_FOLDER "model_parallel/mlp_megatron_basic_test.onnx";
  std::shared_ptr<Model> p_model;
  auto ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(ret.IsOK());
  Graph& graph = p_model->MainGraph();
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MegatronTransformer>(0, 2), TransformerLevel::Level1);
  ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

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
  auto ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MegatronTransformer>(1, 2), TransformerLevel::Level1);
  ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

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
  auto ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(ret.IsOK());
  Graph& graph = p_model->MainGraph();
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MegatronTransformer>(0, 2), TransformerLevel::Level1);
  ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

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
  auto ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MegatronTransformer>(1, 2), TransformerLevel::Level1);
  ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

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

// We only tested on CUDA run.
#if defined(USE_CUDA)
TEST_F(GraphTransformationTests, MegatronMLPPartitionCorrectnessTest) {
  auto model_uri = MODEL_FOLDER "model_parallel/mlp_megatron_basic_test.onnx";
  const int total_rank = 4;
  std::vector<Graph*> graphs;
  std::vector<std::shared_ptr<Model>> p_models(total_rank);
  for (auto i = 0; i < total_rank; i++) {
    auto ret = Model::Load(model_uri, p_models[i], nullptr, *logger_);
    ASSERT_TRUE(ret.IsOK());
    Graph& graph = p_models[i]->MainGraph();
    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    graph_transformation_mgr.Register(onnxruntime::make_unique<MegatronTransformer>(i, total_rank), TransformerLevel::Level1);
    ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
    ASSERT_TRUE(ret.IsOK());
    graphs.push_back(&graph);
  }

  onnxruntime::Model combine_model("combine_graph", false, *logger_);
  auto& combine_graph = combine_model.MainGraph();
  auto ret = horizontal_parallel_test_utils::MergeGraphsOnAllWorkers(graphs, combine_graph);
  ORT_ENFORCE(ret.IsOK());
  auto model_uri2 = "mlp_megatron_basic_test_partition_combine.onnx";
  Model::Save(combine_model, model_uri2);

  float scale = 1.f;
  float mean = 0.f;
  float seed = 123.f;

  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution{mean, scale};

  std::vector<int64_t> dims_X = {8, 16, 4};
  std::vector<float> values_X(TensorShape(dims_X).Size());
  std::for_each(values_X.begin(), values_X.end(),
                [&generator, &distribution](float& value) { value = distribution(generator); });

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

    OrtValue ml_value;
    CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_X, values_X, &ml_value);
    NameMLValMap feeds;
    feeds.insert(std::make_pair("input", ml_value));

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

    OrtValue ml_value;
    CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_X, values_X, &ml_value);
    NameMLValMap feeds;
    feeds.insert(std::make_pair("input", ml_value));

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

TEST_F(GraphTransformationTests, MegatronSelfAttentionPartitionCorrectnessTest) {
  auto model_uri = MODEL_FOLDER "model_parallel/self_attention_megatron_basic_test.onnx";
  const int total_rank = 2;  // The test graph is too small to partition to 4, so use 2 instead here.
  std::vector<Graph*> graphs;
  std::vector<std::shared_ptr<Model>> p_models(total_rank);
  for (auto i = 0; i < total_rank; i++) {
    auto ret = Model::Load(model_uri, p_models[i], nullptr, *logger_);
    ASSERT_TRUE(ret.IsOK());
    Graph& graph = p_models[i]->MainGraph();
    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    graph_transformation_mgr.Register(onnxruntime::make_unique<MegatronTransformer>(i, total_rank), TransformerLevel::Level1);
    ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
    ASSERT_TRUE(ret.IsOK());
    graphs.push_back(&graph);
  }

  // Dropout seed checking.
  const AttributeProto* attr = graph_utils::GetNodeAttribute(*GetNodeByName(*graphs[0], "dropout1"), "seed");
  ORT_ENFORCE(attr != nullptr && attr->has_i());
  int64_t dropout1_rank0_seed = attr->i();
  attr = graph_utils::GetNodeAttribute(*GetNodeByName(*graphs[0], "dropout2"), "seed");
  ORT_ENFORCE(attr != nullptr && attr->has_i());
  int64_t dropout2_rank0_seed = attr->i();
  for (auto i = 1; i < total_rank; i++) {
    attr = graph_utils::GetNodeAttribute(*GetNodeByName(*graphs[i], "dropout1"), "seed");
    ORT_ENFORCE(attr != nullptr && attr->has_i() && attr->i() == dropout1_rank0_seed + i);
    attr = graph_utils::GetNodeAttribute(*GetNodeByName(*graphs[i], "dropout2"), "seed");
    ORT_ENFORCE(attr != nullptr && attr->has_i() && attr->i() == dropout2_rank0_seed);
  }

  onnxruntime::Model combine_model("combine_graph", false, *logger_);
  auto& combine_graph = combine_model.MainGraph();
  auto ret = horizontal_parallel_test_utils::MergeGraphsOnAllWorkers(graphs, combine_graph);
  ORT_ENFORCE(ret.IsOK());
  auto model_uri2 = "self_attention_megatron_basic_test_partition_combine.onnx";
  Model::Save(combine_model, model_uri2);

  float scale = 1.f;
  float mean = 0.f;
  float seed = 123.f;

  std::default_random_engine generator{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution{mean, scale};

  std::vector<int64_t> dims_X = {8, 16, 4};
  std::vector<float> values_X(TensorShape(dims_X).Size());
  std::for_each(values_X.begin(), values_X.end(),
                [&generator, &distribution](float& value) { value = distribution(generator); });

  std::vector<int64_t> dims_Mask = {8, 1, 16, 16};
  std::vector<float> values_Mask(TensorShape(dims_Mask).Size());
  std::for_each(values_Mask.begin(), values_Mask.end(),
                [&generator, &distribution](float& value) { value = distribution(generator); });

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

    NameMLValMap feeds;

    OrtValue ml_value;
    CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_X, values_X, &ml_value);
    feeds.insert(std::make_pair("input", ml_value));

    OrtValue mask_value;
    CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_Mask, values_Mask, &mask_value);
    feeds.insert(std::make_pair("mask", mask_value));

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

    NameMLValMap feeds;
    OrtValue ml_value;
    CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_X, values_X, &ml_value);
    feeds.insert(std::make_pair("input", ml_value));

    OrtValue mask_value;
    CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_Mask, values_Mask, &mask_value);
    feeds.insert(std::make_pair("mask", mask_value));

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

// end of USE_CUDA
#endif

// end of DISABLE_CONTRIB_OPS
#endif

}  // namespace test
}  // namespace onnxruntime
