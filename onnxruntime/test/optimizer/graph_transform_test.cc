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
TEST(GraphTransformationTests, IdentityElimination) {
  string model_uri = MODEL_FOLDER + "abs-id-max.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 1);

  std::unique_ptr<TopDownRuleBasedTransformer> rule_transformer =
      std::make_unique<TopDownRuleBasedTransformer>("RuleTransformer1", "First rule transformer");
  rule_transformer->Register(std::make_unique<EliminateIdentity>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer), TransformerLevel::Level1, {});
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 0);
}

TEST(GraphTransformationTests, SliceElimination) {
  string model_uri = MODEL_FOLDER + "slice-elim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Slice"] == 5);

  std::unique_ptr<TopDownRuleBasedTransformer> rule_transformer =
      std::make_unique<TopDownRuleBasedTransformer>("RuleTransformer1", "First rule transformer");
  rule_transformer->Register(std::make_unique<EliminateSlice>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer), TransformerLevel::Level1, {});
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Slice"] == 3);
}

TEST(GraphTransformationTests, ConstantFolding1) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 2);

  std::unique_ptr<TopDownRuleBasedTransformer> rule_transformer =
      std::make_unique<TopDownRuleBasedTransformer>("RuleTransformer1", "First rule transformer");

  rule_transformer->Register(std::make_unique<ConstantFolding>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};

  graph_transformation_mgr.Register(std::move(rule_transformer), TransformerLevel::Level1, {});
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST(GraphTransformationTests, FuseConvBNMulAddUnsqueeze) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  std::unique_ptr<UnsqueezeElimination> Unsqueeze_transformer = std::make_unique<UnsqueezeElimination>();
  std::unique_ptr<ConvBNFusion> ConvBNFusion_transformer = std::make_unique<ConvBNFusion>();
  std::unique_ptr<ConvMulFusion> ConvMulFusion_transformer = std::make_unique<ConvMulFusion>();
  std::unique_ptr<ConvAddFusion> ConvAddFusion_transformer = std::make_unique<ConvAddFusion>();

  session_object.RegisterGraphTransformer(std::move(Unsqueeze_transformer), {});
  session_object.RegisterGraphTransformer(std::move(ConvBNFusion_transformer), {onnxruntime::kCpuExecutionProvider});
  session_object.RegisterGraphTransformer(std::move(ConvMulFusion_transformer), {onnxruntime::kCpuExecutionProvider});
  session_object.RegisterGraphTransformer(std::move(ConvAddFusion_transformer), {onnxruntime::kCpuExecutionProvider});

  ASSERT_TRUE(session_object.Initialize().IsOK());
}

TEST(GraphTransformationTests, FuseConvActivation) {
  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  std::string activations[] = {"relu", "sigmoid", "softsign", "tanh", "leakyrelu"};

  for (std::string act : activations) {
    InferenceSession session_object{so, &DefaultLoggingManager()};
    std::string model_uri = MODEL_FOLDER + "fusion/conv_" + act + ".onnx";
    ASSERT_TRUE(session_object.Load(model_uri).IsOK());

    std::shared_ptr<Model> p_model;
    ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());
    std::unique_ptr<ConvActivationFusion> ConvActivationFusion_transformer = std::make_unique<ConvActivationFusion>();
    session_object.RegisterGraphTransformer(std::move(ConvActivationFusion_transformer), {onnxruntime::kCpuExecutionProvider});

    ASSERT_TRUE(session_object.Initialize().IsOK());
  }
}

TEST(GraphTransformationTests, FuseConvBNNoBias) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-no-bias.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  std::unique_ptr<ConvBNFusion> ConvBNFusion_transformer = std::make_unique<ConvBNFusion>();

  session_object.RegisterGraphTransformer(std::move(ConvBNFusion_transformer), {onnxruntime::kCpuExecutionProvider});

  ASSERT_TRUE(session_object.Initialize().IsOK());
}

TEST(GraphTransformationTests, FuseConvMulNoBias) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-mul-no-bias.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  std::unique_ptr<UnsqueezeElimination> Unsqueeze_transformer = std::make_unique<UnsqueezeElimination>();
  std::unique_ptr<ConvMulFusion> ConvMulFusion_transformer = std::make_unique<ConvMulFusion>();

  session_object.RegisterGraphTransformer(std::move(Unsqueeze_transformer), {});
  session_object.RegisterGraphTransformer(std::move(ConvMulFusion_transformer), {onnxruntime::kCpuExecutionProvider});
  Status st = session_object.Initialize();
  ASSERT_TRUE(st.IsOK()) << st;
}

TEST(GraphTransformationTests, FuseConvAddNoBias) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-add-no-bias.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  std::unique_ptr<UnsqueezeElimination> Unsqueeze_transformer = std::make_unique<UnsqueezeElimination>();
  std::unique_ptr<ConvAddFusion> ConvAddFusion_transformer = std::make_unique<ConvAddFusion>();

  session_object.RegisterGraphTransformer(std::move(Unsqueeze_transformer), {});
  session_object.RegisterGraphTransformer(std::move(ConvAddFusion_transformer), {onnxruntime::kCpuExecutionProvider});

  Status st = session_object.Initialize();
  ASSERT_TRUE(st.IsOK()) << st;
}

TEST(GraphTransformationTests, FuseConvBNMulAddUnsqueezeNoBias) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-mul-add-unsqueeze-no-bias.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  std::unique_ptr<UnsqueezeElimination> Unsqueeze_transformer = std::make_unique<UnsqueezeElimination>();
  std::unique_ptr<ConvBNFusion> ConvBNFusion_transformer = std::make_unique<ConvBNFusion>();
  std::unique_ptr<ConvMulFusion> ConvMulFusion_transformer = std::make_unique<ConvMulFusion>();
  std::unique_ptr<ConvAddFusion> ConvAddFusion_transformer = std::make_unique<ConvAddFusion>();

  session_object.RegisterGraphTransformer(std::move(Unsqueeze_transformer), {});
  session_object.RegisterGraphTransformer(std::move(ConvBNFusion_transformer), {onnxruntime::kCpuExecutionProvider});
  session_object.RegisterGraphTransformer(std::move(ConvMulFusion_transformer), {onnxruntime::kCpuExecutionProvider});
  session_object.RegisterGraphTransformer(std::move(ConvAddFusion_transformer), {onnxruntime::kCpuExecutionProvider});

  Status st = session_object.Initialize();
  ASSERT_TRUE(st.IsOK()) << st;
}

TEST(GraphTransformationTests, FuseConvAddMul3D) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-add-mul-3d.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  so.graph_optimization_level = TransformerLevel::Level2;
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  //std::unique_ptr<ConvMulFusion> ConvMulFusion_transformer = std::make_unique<ConvMulFusion>();
  //std::unique_ptr<ConvAddFusion> ConvAddFusion_transformer = std::make_unique<ConvAddFusion>();

  //session_object.RegisterGraphTransformer(std::move(ConvMulFusion_transformer), {onnxruntime::kCpuExecutionProvider});
  //session_object.RegisterGraphTransformer(std::move(ConvAddFusion_transformer), {onnxruntime::kCpuExecutionProvider});

  Status st = session_object.Initialize();
  ASSERT_TRUE(st.IsOK()) << st;
}
/*
TEST(GraphTransformationTests, MatMulAddFusion_two_input) {
  string model_uri = MODEL_FOLDER + "matmul_add_fusion/2Input/model.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  std::unique_ptr<MatMulAddFusion> matmul_add_fusion_transformer = std::make_unique<MatMulAddFusion>();

  session_object.RegisterGraphTransformer(std::move(matmul_add_fusion_transformer), {onnxruntime::kCpuExecutionProvider});

  ASSERT_TRUE(session_object.Initialize().IsOK());
}

TEST(GraphTransformationTests, MatMulAddFusion_three_input) {
  string model_uri = MODEL_FOLDER + "matmul_add_fusion/3Input/model.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  std::unique_ptr<MatMulAddFusion> matmul_add_fusion_transformer = std::make_unique<MatMulAddFusion>();

  session_object.RegisterGraphTransformer(std::move(matmul_add_fusion_transformer), {onnxruntime::kCpuExecutionProvider});

  ASSERT_TRUE(session_object.Initialize().IsOK());
}

TEST(GraphTransformationTests, Gemm_Relu_three_input) {
  string model_uri = MODEL_FOLDER + "matmul_add_fusion/3Input/gemm_relu.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  std::unique_ptr<GemmActivationFusion> gemm_activation_fusion_transformer = std::make_unique<GemmActivationFusion>();

  session_object.RegisterGraphTransformer(std::move(gemm_activation_fusion_transformer), {onnxruntime::kCpuExecutionProvider});

  ASSERT_TRUE(session_object.Initialize().IsOK());
}
*/
TEST(GraphTransformationTests, FuseConvBnAddMulFloat16) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-add-mul-float16.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  std::unique_ptr<ConvBNFusion> ConvBNFusion_transformer = std::make_unique<ConvBNFusion>();
  std::unique_ptr<ConvMulFusion> ConvMulFusion_transformer = std::make_unique<ConvMulFusion>();
  std::unique_ptr<ConvAddFusion> ConvAddFusion_transformer = std::make_unique<ConvAddFusion>();
  session_object.RegisterGraphTransformer(std::move(ConvBNFusion_transformer), {onnxruntime::kCpuExecutionProvider});
  session_object.RegisterGraphTransformer(std::move(ConvMulFusion_transformer), {onnxruntime::kCpuExecutionProvider});
  session_object.RegisterGraphTransformer(std::move(ConvAddFusion_transformer), {onnxruntime::kCpuExecutionProvider});

  ASSERT_TRUE(session_object.Initialize().IsOK());

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

  ASSERT_TRUE(session_object.Run(run_options, feeds, output_names, &fetches).IsOK());

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
