// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/inference_session.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/dropout_elimination.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/optimizer/relu_clip_fusion.h"
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

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(std::make_unique<EliminateIdentity>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 0);
}

TEST(GraphTransformationTests, DropoutEliminationSingleOutput) {
  string model_uri = MODEL_FOLDER + "dropout.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 5);
  ASSERT_TRUE(op_to_count["Dropout"] == 6);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(std::make_unique<EliminateDropout>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1).IsOK());

  op_to_count = CountOpsInGraph(graph);
  // Of the 6 Dropout nodes in the graph, all but the ones named `d1` and `d6` should have been removed.
  // A Dropout node can be removed if its second, optional output `mask` is either missing or unused downstream.
  // `d1` cannot be removed because an Identity node has its `mask` output as an input;
  // `d6` cannot be removed because its `mask` output is marked as a graph output.
  ASSERT_TRUE(op_to_count["Identity"] == 5);
  ASSERT_TRUE(op_to_count["Dropout"] == 2);
}

TEST(GraphTransformationTests, SliceElimination) {
  string model_uri = MODEL_FOLDER + "slice-elim.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Slice"] == 5);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(std::make_unique<EliminateSlice>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Slice"] == 4);
}

TEST(GraphTransformationTests, ConstantFolding1) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::make_unique<ConstantFolding>(), TransformerLevel::Level1);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

// Check transformations in the case of a subgraph with constant inputs.
TEST(GraphTransformationTests, SubgraphWithConstantInputs) {
  string model_uri = MODEL_FOLDER + "constant-subgraph.onnx";

  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::Level2;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  ASSERT_TRUE(session_object.Initialize().IsOK());

  NameMLValMap feeds;
  RunOptions run_options;

  std::vector<std::string> output_names = {"output"};
  std::vector<OrtValue> fetches;

  ASSERT_TRUE(session_object.Run(run_options, feeds, output_names, &fetches).IsOK());
}

TEST(GraphTransformationTests, FuseConvBNNoBias) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L2 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL2");
  rule_transformer_L2->Register(std::make_unique<ConvBNFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L2), TransformerLevel::Level2);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["BatchNormalization"] == 0);
}

TEST(GraphTransformationTests, FuseConvBNMulAddUnsqueeze) {
  std::vector<std::string> test_models = {"fusion/fuse-conv-bn-mul-add-unsqueeze.onnx",
                                          "fusion/fuse-conv-bn-mul-add-unsqueeze-no-bias.onnx"};
  for (const auto& model : test_models) {
    string model_uri = MODEL_FOLDER + model;

    std::shared_ptr<Model> p_model;
    ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());
    Graph& graph = p_model->MainGraph();

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
    rule_transformer_L1->Register(std::make_unique<UnsqueezeElimination>());
    graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

    auto rule_transformer_L2 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL2");
    rule_transformer_L2->Register(std::make_unique<ConvAddFusion>());
    rule_transformer_L2->Register(std::make_unique<ConvBNFusion>());
    rule_transformer_L2->Register(std::make_unique<ConvMulFusion>());
    graph_transformation_mgr.Register(std::move(rule_transformer_L2), TransformerLevel::Level2);

    ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1).IsOK());
    ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2).IsOK());

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count["BatchNormalization"] == 0);
    ASSERT_TRUE(op_to_count["Mul"] == 0);
    ASSERT_TRUE(op_to_count["Add"] == 0);
    ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  }
}

#ifndef DISABLE_CONTRIB_OPS
TEST(GraphTransformationTests, FuseConvActivation) {
  std::unordered_map<std::string, std::string> model_to_op_name{{"fusion/conv_relu.onnx", "Relu"},
                                                                {"fusion/conv_sigmoid.onnx", "Sigmoid"},
                                                                {"fusion/conv_tanh.onnx", "Tanh"},
                                                                {"fusion/conv_leakyrelu.onnx", "LeakyRelu"}};

  for (const auto& model : model_to_op_name) {
    std::string model_uri = MODEL_FOLDER + model.first;
    std::shared_ptr<Model> p_model;
    ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());
    Graph& graph = p_model->MainGraph();

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[model.second] >= 1);

    // Apply transformer
    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level2);
    ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2).IsOK());

    op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[model.second] == 0);
  }
}
#endif

TEST(GraphTransformationTests, FuseConvMulNoBias) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-mul-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(std::make_unique<UnsqueezeElimination>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  auto rule_transformer_L2 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL2");
  rule_transformer_L2->Register(std::make_unique<ConvMulFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L2), TransformerLevel::Level2);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1).IsOK());
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST(GraphTransformationTests, FuseConvAddNoBias) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-add-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(std::make_unique<UnsqueezeElimination>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  auto rule_transformer_L2 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL2");
  rule_transformer_L2->Register(std::make_unique<ConvAddFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L2), TransformerLevel::Level2);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1).IsOK());
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST(GraphTransformationTests, FuseConvAddMul3D) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-add-mul-3d.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L2 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL2");
  rule_transformer_L2->Register(std::make_unique<ConvAddFusion>());
  rule_transformer_L2->Register(std::make_unique<ConvMulFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L2), TransformerLevel::Level2);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
}

TEST(GraphTransformationTests, FuseConvAddMul3D_2) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-add-mul-3d-2.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L2 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL2");
  rule_transformer_L2->Register(std::make_unique<ConvAddFusion>());
  rule_transformer_L2->Register(std::make_unique<ConvMulFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L2), TransformerLevel::Level2);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
}

TEST(GraphTransformationTests, MatMulAddFusion_two_input) {
  string model_uri = MODEL_FOLDER + "matmul_add_fusion/2Input/model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::make_unique<MatMulAddFusion>(), TransformerLevel::Level2);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 1);
}

TEST(GraphTransformationTests, MatMulAddFusion_three_input) {
  string model_uri = MODEL_FOLDER + "matmul_add_fusion/3Input/model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::make_unique<MatMulAddFusion>(), TransformerLevel::Level2);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 1);
}

#ifndef DISABLE_CONTRIB_OPS
TEST(GraphTransformationTests, Gemm_Relu_three_input) {
  string model_uri = MODEL_FOLDER + "matmul_add_fusion/3Input/gemm_relu.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count1 = CountOpsInGraph(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::make_unique<GemmActivationFusion>(), TransformerLevel::Level2);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 0);
}
#endif

TEST(GraphTransformationTests, FuseConvBnAddMulFloat16) {
  string model_uri = MODEL_FOLDER + "fusion/fuse-conv-bn-add-mul-float16.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model).IsOK());

  auto rule_transformer_L2 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformerL2");
  rule_transformer_L2->Register(std::make_unique<ConvAddFusion>());
  rule_transformer_L2->Register(std::make_unique<ConvBNFusion>());
  rule_transformer_L2->Register(std::make_unique<ConvMulFusion>());
  session_object.RegisterGraphTransformer(std::move(rule_transformer_L2), TransformerLevel::Level2);

  ASSERT_TRUE(session_object.Initialize().IsOK());

  NameMLValMap feeds;
  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  OrtValue ml_value_x;

  auto x_f = MLFloat16(math::floatToHalf(1.0));
  std::vector<int64_t> dims_x = {1, 1, 3, 3};
  std::vector<MLFloat16> values_x;
  for (int i = 0; i < 9; ++i) {
    values_x.push_back(x_f);
  }
  CreateMLValue<MLFloat16>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault),
                           dims_x, values_x, &ml_value_x);
  feeds.insert(std::make_pair("X", ml_value_x));

  std::vector<std::string> output_names;
  output_names.push_back("PROD");
  std::vector<OrtValue> fetches;

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
  const std::vector<MLFloat16> found(rtensor.template Data<MLFloat16>(),
                                     rtensor.template Data<MLFloat16>() + expected_dims_prod.size());
  ASSERT_EQ(expected_values_prod, found);
}

TEST(GraphTransformationTests, ReluClipFusion) {
  Model model("ReluClipFusion");
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // 3 paths in the model, each with Relu followed by Clip
  // One has a Clip with min of 0  (remove Relu)
  // One have a Clip with a min > 1 (remove Relu)
  // One has a Clip with min < 0 (remove Relu and update Clip 'min' to 0)
  auto& input0 = graph.GetOrCreateNodeArg("input_0", &input_tensor_type);
  auto& input1 = graph.GetOrCreateNodeArg("input_1", &input_tensor_type);
  auto& input2 = graph.GetOrCreateNodeArg("input_2", &input_tensor_type);

  auto& relu0_output = graph.GetOrCreateNodeArg("relu0_output", &input_tensor_type);
  auto& relu1_output = graph.GetOrCreateNodeArg("relu1_output", &input_tensor_type);
  auto& relu2_output = graph.GetOrCreateNodeArg("relu2_output", &input_tensor_type);

  auto& clip0_output = graph.GetOrCreateNodeArg("clip0_output", &input_tensor_type);
  auto& clip1_output = graph.GetOrCreateNodeArg("clip1_output", &input_tensor_type);
  auto& clip2_output = graph.GetOrCreateNodeArg("clip2_output", &input_tensor_type);

  graph.AddNode("relu0", "Relu", "Relu to eliminate", {&input0}, {&relu0_output});
  graph.AddNode("relu1", "Relu", "Relu to not eliminate", {&input1}, {&relu1_output});
  graph.AddNode("relu2", "Relu", "Relu to eliminate and update 'min' of following Clip", {&input2}, {&relu2_output});

  auto& clip0 = graph.AddNode("clip0", "Clip", "Clip with min 0", {&relu0_output}, {&clip0_output});
  clip0.AddAttribute("min", 0.f);
  clip0.AddAttribute("max", 1.f);

  auto& clip1 = graph.AddNode("clip1", "Clip", "Clip with min 1", {&relu1_output}, {&clip1_output});
  clip1.AddAttribute("min", 1.f);
  clip1.AddAttribute("max", 1.f);

  auto& clip2 = graph.AddNode("clip2", "Clip", "Clip with min -1", {&relu2_output}, {&clip2_output});
  clip2.AddAttribute("min", -1.f);
  clip2.AddAttribute("max", 1.f);

  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 3);

  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(std::make_unique<FuseReluClip>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 0);

  // make sure the Clip nodes were updated to have a 'min' >= 0
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Clip") {
      auto* min = graph_utils::GetNodeAttribute(node, "min");
      ASSERT_TRUE(min->f() >= 0.f);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime
