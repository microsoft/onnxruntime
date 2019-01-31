// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/graph/graph_transformer.h"
#include "core/graph/graph_transformer_mgr.h"
#include "core/graph/identity_elimination.h"
#include "core/graph/slice_elimination.h"
#include "core/graph/unsqueeze_elimination.h"
#include "core/graph/conv_bn_fusion.h"
#include "core/graph/conv_mul_fusion.h"
#include "core/graph/conv_add_fusion.h"
#include "core/graph/conv_activation_fusion.h"
#include "core/graph/matmul_add_fusion.h"
#include "core/graph/gemm_activation_fusion.h"
#include "core/platform/env.h"

#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "gtest/gtest.h"

using namespace std;
using namespace ONNX_NAMESPACE;

using namespace onnx;

namespace onnxruntime {
namespace test {

static const std::string MODEL_FOLDER = "testdata/transform/";

// Return a map with the number of occurrences of each operator in the graph.
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
  rule_transformer->Register("Identity", std::make_unique<EliminateIdentity>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer));
  ASSERT_TRUE(graph_transformation_mgr.ApplyAll(graph).IsOK());

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
  rule_transformer->Register("Slice", std::make_unique<EliminateSlice>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer));
  ASSERT_TRUE(graph_transformation_mgr.ApplyAll(graph).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Slice"] == 3);
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

  session_object.RegisterGraphTransformer(std::move(Unsqueeze_transformer));
  session_object.RegisterGraphTransformer(std::move(ConvBNFusion_transformer));
  session_object.RegisterGraphTransformer(std::move(ConvMulFusion_transformer));
  session_object.RegisterGraphTransformer(std::move(ConvAddFusion_transformer));

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
    session_object.RegisterGraphTransformer(std::move(ConvActivationFusion_transformer));

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

  session_object.RegisterGraphTransformer(std::move(ConvBNFusion_transformer));

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

  session_object.RegisterGraphTransformer(std::move(Unsqueeze_transformer));
  session_object.RegisterGraphTransformer(std::move(ConvMulFusion_transformer));
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

  session_object.RegisterGraphTransformer(std::move(Unsqueeze_transformer));
  session_object.RegisterGraphTransformer(std::move(ConvAddFusion_transformer));

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

  session_object.RegisterGraphTransformer(std::move(Unsqueeze_transformer));
  session_object.RegisterGraphTransformer(std::move(ConvBNFusion_transformer));
  session_object.RegisterGraphTransformer(std::move(ConvMulFusion_transformer));
  session_object.RegisterGraphTransformer(std::move(ConvAddFusion_transformer));

  Status st = session_object.Initialize();
  ASSERT_TRUE(st.IsOK()) << st;
}

TEST(GraphTransformationTests, FuseConvAddMul3D) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-add-mul-3d.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  std::unique_ptr<ConvMulFusion> ConvMulFusion_transformer = std::make_unique<ConvMulFusion>();
  std::unique_ptr<ConvAddFusion> ConvAddFusion_transformer = std::make_unique<ConvAddFusion>();

  session_object.RegisterGraphTransformer(std::move(ConvMulFusion_transformer));
  session_object.RegisterGraphTransformer(std::move(ConvAddFusion_transformer));

  Status st = session_object.Initialize();
  ASSERT_TRUE(st.IsOK()) << st;
}

TEST(GraphTransformationTests, MatMulAddFusion_two_input) {
  string model_uri = MODEL_FOLDER + "matmul_add_fusion/2Input/model.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  std::unique_ptr<MatMulAddFusion> matmul_add_fusion_transformer = std::make_unique<MatMulAddFusion>();

  session_object.RegisterGraphTransformer(std::move(matmul_add_fusion_transformer));

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

  session_object.RegisterGraphTransformer(std::move(matmul_add_fusion_transformer));

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

  session_object.RegisterGraphTransformer(std::move(gemm_activation_fusion_transformer));

  ASSERT_TRUE(session_object.Initialize().IsOK());
}


}  // namespace test
}  // namespace onnxruntime
