// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/framework/data_types.h"
#include "core/framework/ml_value.h"
#include "core/util/math.h"
#include "core/platform/env.h"
#include "test/framework/test_utils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/graph_transformer_utils.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

static const std::string MODEL_FOLDER = "testdata/transform/";

// Returns a map with the number of occurrences of each operator in the graph.
// Helper function to check that the graph transformations have been successfully applied.
std::map<std::string, int> CountOpsInGraph(const Graph& graph) {
  std::map<std::string, int> op_to_count;
  for (auto& node : graph.Nodes()) {
    op_to_count[node.OpType()] =
        op_to_count.count(node.OpType()) == 0 ? 1 : ++op_to_count[node.OpType()];
  }
  return op_to_count;
}

// Takes as input an onnx file uri and a list of transformers and rules. Then it creates,
// initializes, and returns an inference session.
std::unique_ptr<InferenceSession> CreateSession(const std::string& model_uri,
                                                std::vector<std::string>& rules_and_transformers) {
  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  std::unique_ptr<InferenceSession> session_object = 
	  std::make_unique<InferenceSession>(so, &DefaultLoggingManager());
  if (!session_object->Load(model_uri).IsOK()) {
    return std::unique_ptr<InferenceSession>{};
  }
  session_object->AddCustomTransformerList(rules_and_transformers);
  if (!session_object->Initialize().IsOK()) {
    return std::unique_ptr<InferenceSession>{};
  }
  return session_object;
}

// Takes a list of transformers and rules; creates a transformer manager and registers the transformers/rules
// to it; applies the transformers/rules to the given graph.
void RegisterAndApplyTransformers(Graph& graph, std::vector<std::string>& rules_and_transformers) {
  auto levels = {TransformerLevel::Level1, TransformerLevel::Level2};
  GraphTransformerManager graph_transformation_mgr{5};
  // Register transformers.
  for (auto level : levels) {
    auto transformers_to_register =
        transformer_utils::GenerateTransformers(level, &rules_and_transformers);
    for (auto& entry : transformers_to_register) {
      graph_transformation_mgr.Register(std::move(entry.first), level, std::move(entry.second));
    }
  }
  // Apply transformers.
  for (auto level : levels) {
    ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, level).IsOK());
  }
}

TEST(GraphTransformationTests, IdentityElimination) {
  string model_uri = MODEL_FOLDER + "abs-id-max.onnx";
  std::vector<std::string> rule_list = {"EliminateIdentity"};

  // Check that session gets initialized properly.
  ASSERT_TRUE(CreateSession(model_uri, rule_list));

  // Check that transformations actually happen.
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 1);

  RegisterAndApplyTransformers(graph, rule_list);

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 0);
}

TEST(GraphTransformationTests, SliceElimination) {
  string model_uri = MODEL_FOLDER + "slice-elim.onnx";
  std::vector<std::string> rule_list = {"EliminateSlice"};

  // Check that session gets initialized properly.
  ASSERT_TRUE(CreateSession(model_uri, rule_list));

  // Check that transformations actually happen.
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Slice"] == 5);

  RegisterAndApplyTransformers(graph, rule_list);

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Slice"] == 3);
}

TEST(GraphTransformationTests, ConstantFolding1) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";
  std::vector<std::string> rule_list = {"ConstantFolding"};

  // Check that session gets initialized properly.
  ASSERT_TRUE(CreateSession(model_uri, rule_list));

  // Check that transformations actually happen.
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 2);

  RegisterAndApplyTransformers(graph, rule_list);

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST(GraphTransformationTests, FuseConvBNMulAddUnsqueeze) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";
  std::vector<std::string> rules_and_transformers =
      {"UnsqueezeElimination", "ConvBNFusion", "ConvMulFusion", "ConvAddFusion"};

  // Check that session gets initialized properly.
  ASSERT_TRUE(CreateSession(model_uri, rules_and_transformers));

  // Check that transformations actually happen.
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Conv"] == 1);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 2);
  ASSERT_TRUE(op_to_count["BatchNormalization"] == 1);
  ASSERT_TRUE(op_to_count["Mul"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);

  RegisterAndApplyTransformers(graph, rules_and_transformers);

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Conv"] == 1);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  ASSERT_TRUE(op_to_count["BatchNormalization"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
}

TEST(GraphTransformationTests, FuseConvActivation) {
  std::vector<std::string> rules_and_transformers = {"ConvActivationFusion"};
  std::string activations[] = {"relu", "sigmoid", "softsign", "tanh", "leakyrelu"};

  for (std::string act : activations) {
    std::string model_uri = MODEL_FOLDER + "fusion/conv_" + act + ".onnx";

    // Check that session gets initialized properly.
    ASSERT_TRUE(CreateSession(model_uri, rules_and_transformers));

	// Check that transformations actually happen.
    std::shared_ptr<Model> model;
    ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
	// TODO: Add before checks.

    RegisterAndApplyTransformers(graph, rules_and_transformers);

    op_to_count = CountOpsInGraph(graph);
    // TODO: Add after checks.
  }
}

TEST(GraphTransformationTests, FuseConvBNNoBias) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-no-bias.onnx";
  std::vector<std::string> rules_and_transformers = {"ConvBNFusion"};

  // Check that session gets initialized properly.
  ASSERT_TRUE(CreateSession(model_uri, rules_and_transformers));

  // Check that transformations actually happen.
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  // TODO: Add before checks.

  RegisterAndApplyTransformers(graph, rules_and_transformers);

  op_to_count = CountOpsInGraph(graph);
  // TODO: Add after checks.
}

TEST(GraphTransformationTests, FuseConvMulNoBias) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-mul-no-bias.onnx";
  std::vector<std::string> rules_and_transformers = {"UnsqueezeElimination", "ConvMulFusion"};

  // Check that session gets initialized properly.
  ASSERT_TRUE(CreateSession(model_uri, rules_and_transformers));

  // Check that transformations actually happen.
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  // TODO: Add before checks.

  RegisterAndApplyTransformers(graph, rules_and_transformers);

  op_to_count = CountOpsInGraph(graph);
  // TODO: Add after checks.
}

TEST(GraphTransformationTests, FuseConvAddNoBias) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-add-no-bias.onnx";
  std::vector<std::string> rules_and_transformers = {"UnsqueezeElimination", "ConvAddFusion"};

  // Check that session gets initialized properly.
  ASSERT_TRUE(CreateSession(model_uri, rules_and_transformers));

  // Check that transformations actually happen.
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  // TODO: Add before checks.

  RegisterAndApplyTransformers(graph, rules_and_transformers);

  op_to_count = CountOpsInGraph(graph);
  // TODO: Add after checks.
}

TEST(GraphTransformationTests, FuseConvBNMulAddUnsqueezeNoBias) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-mul-add-unsqueeze-no-bias.onnx";
  std::vector<std::string> rules_and_transformers =
      {"UnsqueezeElimination", "ConvBNFusion", "ConvMulFusion", "ConvAddFusion"};

  // Check that session gets initialized properly.
  ASSERT_TRUE(CreateSession(model_uri, rules_and_transformers));

  // Check that transformations actually happen.
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  // TODO: Add before checks.

  RegisterAndApplyTransformers(graph, rules_and_transformers);

  op_to_count = CountOpsInGraph(graph);
  // TODO: Add after checks.
}

TEST(GraphTransformationTests, FuseConvAddMul3D) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-add-mul-3d.onnx";
  std::vector<std::string> rules_and_transformers = {"ConvMulFusion", "ConvAddFusion"};

  // Check that session gets initialized properly.
  ASSERT_TRUE(CreateSession(model_uri, rules_and_transformers));

  // Check that transformations actually happen.
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  // TODO: Add before checks.

  RegisterAndApplyTransformers(graph, rules_and_transformers);

  op_to_count = CountOpsInGraph(graph);
  // TODO: Add after checks.
}

TEST(GraphTransformationTests, MatMulAddFusion_two_input) {
  string model_uri = MODEL_FOLDER + "matmul_add_fusion/2Input/model.onnx";
  std::vector<std::string> rules_and_transformers = {"MatMulAddFusion"};

  // Check that session gets initialized properly.
  ASSERT_TRUE(CreateSession(model_uri, rules_and_transformers));

  // Check that transformations actually happen.
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  // TODO: Add before checks.

  RegisterAndApplyTransformers(graph, rules_and_transformers);

  op_to_count = CountOpsInGraph(graph);
  // TODO: Add after checks.
}

TEST(GraphTransformationTests, MatMulAddFusion_three_input) {
  string model_uri = MODEL_FOLDER + "matmul_add_fusion/3Input/model.onnx";
  std::vector<std::string> rules_and_transformers = {"MatMulAddFusion"};

  // Check that session gets initialized properly.
  ASSERT_TRUE(CreateSession(model_uri, rules_and_transformers));

  // Check that transformations actually happen.
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  // TODO: Add before checks.

  RegisterAndApplyTransformers(graph, rules_and_transformers);

  op_to_count = CountOpsInGraph(graph);
  // TODO: Add after checks.
}

TEST(GraphTransformationTests, Gemm_Relu_three_input) {
  string model_uri = MODEL_FOLDER + "matmul_add_fusion/3Input/gemm_relu.onnx";
  std::vector<std::string> rules_and_transformers = {"GemmActivationFusion"};

  // Check that session gets initialized properly.
  ASSERT_TRUE(CreateSession(model_uri, rules_and_transformers));

  // Check that transformations actually happen.
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  // TODO: Add before checks.

  RegisterAndApplyTransformers(graph, rules_and_transformers);

  op_to_count = CountOpsInGraph(graph);
  // TODO: Add after checks.
}

TEST(GraphTransformationTests, FuseConvBnAddMulFloat16) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-add-mul-float16.onnx";
  std::vector<std::string> rules_and_transformers = {"ConvBNFusion", "ConvMulFusion", "ConvAddFusion"};

  // Check that session gets initialized properly.
  std::unique_ptr<InferenceSession> session_object = CreateSession(model_uri, rules_and_transformers);
  
  ASSERT_TRUE(session_object);

  NameMLValMap feeds;
  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  MLValue ml_value_x;

  auto x_f = MLFloat16(math::floatToHalf(1.0));
  std::vector<int64_t> dims_x = {1, 1, 3, 3};
  std::vector<MLFloat16> values_x;
  for (int i = 0; i < 9; ++i) {
    values_x.push_back(x_f);
  }
  CreateMLValue<MLFloat16>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_x, values_x, &ml_value_x);
  feeds.insert(std::make_pair("X", ml_value_x));

  std::vector<std::string> output_names;
  output_names.push_back("PROD");
  std::vector<MLValue> fetches;

  ASSERT_TRUE(session_object->Run(run_options, feeds, output_names, &fetches).IsOK());

  auto prod_f = MLFloat16(math::floatToHalf(6.0));
  std::vector<int64_t> expected_dims_prod = {1, 1, 2, 2};
  std::vector<MLFloat16> expected_values_prod;
  for (int i = 0; i < 4; ++i) {
    expected_values_prod.push_back(prod_f);
  }

  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims_prod);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<MLFloat16> found(rtensor.template Data<MLFloat16>(), rtensor.template Data<MLFloat16>() + expected_dims_prod.size());
  ASSERT_EQ(expected_values_prod, found);
}

}  // namespace test
}  // namespace onnxruntime
