// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#include "core/graph/onnx_protobuf.h"

#include "core/session/inference_session.h"
#include "core/framework/data_types.h"
#include "core/framework/ml_value.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/dropout_elimination.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/optimizer/bias_gelu_fusion.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/gelu_approximation.h"
#include "core/optimizer/layer_norm_fusion.h"
#include "core/optimizer/skip_layer_norm_fusion.h"
#include "core/optimizer/embed_layer_norm_fusion.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/matmul_transpose_fusion.h"
#include "core/optimizer/relu_clip_fusion.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/shape_to_initializer.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/optimizer/attention_fusion.h"
#include "core/optimizer/fast_gelu_fusion.h"
#include "core/optimizer/utils.h"
#include "core/platform/env.h"
#include "core/util/math.h"
#include "test/capturing_sink.h"
#include "test/framework/test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"

#include "gtest/gtest.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

TEST(GraphTransformationTests, IdentityElimination) {
  auto model_uri = MODEL_FOLDER "abs-id-max.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 1);

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<EliminateIdentity>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 0);
}

TEST(GraphTransformationTests, DropoutElimination) {
  auto model_uri = MODEL_FOLDER "dropout.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 5);
  ASSERT_TRUE(op_to_count["Dropout"] == 6);

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<EliminateDropout>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  op_to_count = CountOpsInGraph(graph);
  // Of the 6 Dropout nodes in the graph, all but the ones named `d1` and `d6` should have been removed.
  // A Dropout node can be removed if its second, optional output `mask` is either missing or unused downstream.
  // `d1` cannot be removed because an Identity node has its `mask` output as an input;
  // `d6` cannot be removed because its `mask` output is marked as a graph output.
  ASSERT_TRUE(op_to_count["Identity"] == 5);
  ASSERT_TRUE(op_to_count["Dropout"] == 2);
}

TEST(GraphTransformationTests, SliceElimination) {
  std::vector<std::basic_string<ORTCHAR_T>> model_names = {ORT_TSTR("slice-v1-elim.onnx"), ORT_TSTR("slice-v11-elim.onnx")};
  for (const auto& model_name : model_names) {
    auto model_uri = MODEL_FOLDER + model_name;
    std::shared_ptr<Model> model;
    ASSERT_TRUE(Model::Load(model_uri, model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    int initial_slice_num = op_to_count["Slice"];

    auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
    rule_transformer_L1->Register(onnxruntime::make_unique<EliminateSlice>());
    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
    ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

    op_to_count = CountOpsInGraph(graph);
    // Only one Slice operator is redundant and is removed.
    ASSERT_TRUE(op_to_count["Slice"] == --initial_slice_num);
  }
}

TEST(GraphTransformationTests, ConstantFolding) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ConstantFolding>(), TransformerLevel::Level1);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST(GraphTransformationTests, ConstantFoldingNodesOnDifferentEP) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 2);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ConstantFolding>(), TransformerLevel::Level1);

  // assign all nodes to CUDA. the constant folding should override this to perform the constant folding on cpu
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kCudaExecutionProvider);
  }

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);

  // all remaining nodes should still be on CUDA
  for (auto& node : graph.Nodes()) {
    EXPECT_STREQ(node.GetExecutionProviderType().c_str(), kCudaExecutionProvider);
  }
}

TEST(GraphTransformationTests, ConstantFoldingSubgraph) {
  TensorProto value_tensor;
  value_tensor.add_dims(1);
  value_tensor.add_float_data(1.f);
  value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  TypeProto float_tensor_type;
  float_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto create_subgraph = [&](GraphProto& graph_proto) {
    // create subgraph that has an Add node to add a local and parent graph initializer
    Model model("ConstantFoldingSubgraphTest_subgraph", false, DefaultLoggingManager().DefaultLogger());
    auto& graph = model.MainGraph();

    TensorProto local_constant(value_tensor);
    local_constant.set_name("local_constant");
    graph.AddInitializedTensor(local_constant);

    auto& local_constant_arg = graph.GetOrCreateNodeArg("local_constant", &float_tensor_type);
    auto& parent_constant_arg = graph.GetOrCreateNodeArg("parent_constant", &float_tensor_type);
    graph.AddOuterScopeNodeArg("parent_constant");

    auto& add_out = graph.GetOrCreateNodeArg("add_out", &float_tensor_type);
    graph.AddNode("add", "Add", "Add two inputs.", {&parent_constant_arg, &local_constant_arg}, {&add_out});

    auto& subgraph_out = graph.GetOrCreateNodeArg("subgraph_out", &float_tensor_type);
    graph.AddNode("identity", "Identity", "So Add isn't providing graph output.", {&add_out}, {&subgraph_out});

    auto status = graph.Resolve();
    ASSERT_TRUE(status.IsOK()) << status;
    graph_proto = graph.ToGraphProto();
  };

  Model model("ConstantFoldingSubgraphTest_main_graph", false, DefaultLoggingManager().DefaultLogger());
  auto& graph = model.MainGraph();

  // add initializer at parent level
  TensorProto parent_value_tensor(value_tensor);
  parent_value_tensor.set_name("parent_constant");
  graph.AddInitializedTensor(parent_value_tensor);

  // put the subgraph in an If node
  TypeProto if_cond_type;
  if_cond_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_BOOL);
  if_cond_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);
  auto& if_cond_input = graph.GetOrCreateNodeArg("if_in", &if_cond_type);
  auto& if_output = graph.GetOrCreateNodeArg("if_out", &float_tensor_type);

  auto& if_node = graph.AddNode("if", "If", "If node", {&if_cond_input}, {&if_output});

  GraphProto subgraph;
  create_subgraph(subgraph);

  if_node.AddAttribute("then_branch", subgraph);
  if_node.AddAttribute("else_branch", subgraph);

  auto status = graph.Resolve();
  ASSERT_TRUE(status.IsOK()) << status;

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 2);  // one in each subgraph

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ConstantFolding>(), TransformerLevel::Level1);

  status = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status;

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0)
      << "Constant folding should have been able to remove the Add node in both subgraphs";
}

TEST(GraphTransformationTests, ShapeToInitializer) {
  auto model_uri = MODEL_FOLDER "shape-add.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 4);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  rule_transformer_L1->Register(onnxruntime::make_unique<ShapeToInitializer>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  op_to_count = CountOpsInGraph(graph);
  // Two of the Shapes are not eliminated because:
  // One includes a symbolic dimension.
  // Another one includes a negative dimension
  ASSERT_TRUE(op_to_count["Shape"] == 2);
}

// Check transformations in the case of a subgraph with constant inputs.
TEST(GraphTransformationTests, SubgraphWithConstantInputs) {
  auto model_uri = MODEL_FOLDER "constant-subgraph.onnx";

  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::Level2;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());

  ASSERT_TRUE(session_object.Initialize().IsOK());

  NameMLValMap feeds;
  RunOptions run_options;

  std::vector<std::string> output_names = {"output"};
  std::vector<OrtValue> fetches;

  ASSERT_TRUE(session_object.Run(run_options, feeds, output_names, &fetches).IsOK());
}

TEST(GraphTransformationTests, FuseConvBNNoBias) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  std::string bn_output_name;

  // add a missing optional output to BN. this should be fusable
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "BatchNormalization") {
      node.MutableOutputDefs().push_back(&graph.GetOrCreateNodeArg("", nullptr));
      bn_output_name = node.OutputDefs()[0]->Name();
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvBNFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["BatchNormalization"] == 0);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      ASSERT_EQ(node.OutputDefs()[0]->Name(), bn_output_name)
          << "fusion should produce the same output name as the last node";
    }
  }
}

TEST(GraphTransformationTests, DontFuseConvWithBNWithOptionalOutputs) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  // add an optional output to the BN node. should not fuse if this is present
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "BatchNormalization") {
      auto mean_input = node.InputDefs()[3];
      auto& mean_output = graph.GetOrCreateNodeArg(mean_input->Name() + ".output", mean_input->TypeAsProto());
      node.MutableOutputDefs().push_back(&mean_output);
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvBNFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["BatchNormalization"] == 1);
}

TEST(GraphTransformationTests, FuseConvBNMulAddUnsqueeze) {
  std::vector<std::basic_string<ORTCHAR_T>> test_models = {ORT_TSTR("fusion/fuse-conv-bn-mul-add-unsqueeze.onnx"),
                                                           ORT_TSTR("fusion/fuse-conv-bn-mul-add-unsqueeze.negative_axes.onnx"),
                                                           ORT_TSTR("fusion/fuse-conv-bn-mul-add-unsqueeze-no-bias.onnx")};
  for (const auto& model : test_models) {
    auto model_uri = MODEL_FOLDER + model;

    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()));
    Graph& graph = p_model->MainGraph();

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
    rule_transformer_L1->Register(onnxruntime::make_unique<UnsqueezeElimination>());
    rule_transformer_L1->Register(onnxruntime::make_unique<ConvAddFusion>());
    rule_transformer_L1->Register(onnxruntime::make_unique<ConvBNFusion>());
    rule_transformer_L1->Register(onnxruntime::make_unique<ConvMulFusion>());
    graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

    ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count["BatchNormalization"] == 0);
    ASSERT_TRUE(op_to_count["Mul"] == 0);
    ASSERT_TRUE(op_to_count["Add"] == 0);
    ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  }
}

#ifndef DISABLE_CONTRIB_OPS
TEST(GraphTransformationTests, FuseConvActivation) {
  std::unordered_map<std::basic_string<ORTCHAR_T>, std::string> model_to_op_name{{ORT_TSTR("fusion/conv_relu.onnx"), "Relu"},
                                                                                 {ORT_TSTR("fusion/conv_clip.onnx"), "Clip"},
                                                                                 {ORT_TSTR("fusion/conv_sigmoid.onnx"), "Sigmoid"},
                                                                                 {ORT_TSTR("fusion/conv_tanh.onnx"), "Tanh"},
                                                                                 {ORT_TSTR("fusion/conv_leakyrelu.onnx"), "LeakyRelu"}};

  for (const auto& model : model_to_op_name) {
    auto model_uri = MODEL_FOLDER + model.first;
    std::shared_ptr<Model> p_model;
    ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
    Graph& graph = p_model->MainGraph();

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[model.second] >= 1);

    // Apply transformer
    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    graph_transformation_mgr.Register(onnxruntime::make_unique<ConvActivationFusion>(), TransformerLevel::Level2);
    ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger()).IsOK());

    op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[model.second] == 0);
  }
}

TEST(GraphTransformationTests, FuseConvClip11Activation) {
  auto model_uri = MODEL_FOLDER "fusion/conv_clip11.onnx";
  std::shared_ptr<Model> p_model;
  auto status = Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status;
  Graph& graph = p_model->MainGraph();

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Clip"], 3);

  // Apply transformer
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ConvActivationFusion>(), TransformerLevel::Level2);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger()).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Clip"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      EXPECT_TRUE(node.Name() == "Conv1") << "Conv1 should not have been fused as 'min' input to Clip was mutable.";
    }

    if (node.OpType() == "FusedConv") {
      const ONNX_NAMESPACE::AttributeProto& attr_proto = node.GetAttributes().at("activation_params");
      const auto& params = attr_proto.floats();
      // check expected values for each. Conv0 is explicitly specified. Conv2 are defaults
      if (node.Name() == "Conv0") {
        EXPECT_TRUE(params.Get(0) == -1.f);
        EXPECT_TRUE(params.Get(1) == 1.f);
      } else if (node.Name() == "Conv2") {
        EXPECT_TRUE(params.Get(0) == std::numeric_limits<float>::lowest());
        EXPECT_TRUE(params.Get(1) == std::numeric_limits<float>::max());
      }
    }
  }
}
#endif

TEST(GraphTransformationTests, FuseConvMulNoBias) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-mul-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<UnsqueezeElimination>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvMulFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST(GraphTransformationTests, FuseConvAddNoBias) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-add-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<UnsqueezeElimination>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvAddFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

// if IR version is 4 or higher the weights can be overridden if there's a matching graph input.
// check that we don't fuse if that is the case
TEST(GraphTransformationTests, NegativeFuseConvAddNoBias) {
  auto model_uri = MODEL_FOLDER "fusion/negative-fuse-conv-add-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<UnsqueezeElimination>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvAddFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  // Nodes are not fused because the weights to conv/add are not constants (they appear in the graph inputs).
  // Unsqueeze is also not eliminated as the initializer that is its input is also not constant
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] != 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] != 0);
}

TEST(GraphTransformationTests, FuseConvAddMul3D) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-add-mul-3d.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvAddFusion>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvMulFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
}

TEST(GraphTransformationTests, FuseConvAddMul3D_2) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-add-mul-3d-2.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvAddFusion>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvMulFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
}

TEST(GraphTransformationTests, MatMulAddFusion_two_input) {
  auto model_uri = MODEL_FOLDER "matmul_add_fusion/2Input/model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MatMulAddFusion>(), TransformerLevel::Level1);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 1);
}

TEST(GraphTransformationTests, MatMulAddFusion_three_input) {
  auto model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MatMulAddFusion>(), TransformerLevel::Level1);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 1);
}

// Matmul+Add with shape [k]*[k,N]+[N], won't do the fusion
// We can do the fusion by changing shape to [1,k]*[k,N]+[1,N], then add a reshape [1,N]=>[N]
// This will bring extra cost. And there's only very limited gain to fuse Matmul+Add to Gemm
// Since the basic implementation is almost same
TEST(GraphTransformationTests, MatMulAddFusion_negitive_case) {
  auto model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/neg_model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MatMulAddFusion>(), TransformerLevel::Level1);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Gemm"] == 0);
}

#ifndef DISABLE_CONTRIB_OPS
TEST(GraphTransformationTests, Gemm_Relu_three_input) {
  auto model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/gemm_relu.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count1 = CountOpsInGraph(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GemmActivationFusion>(), TransformerLevel::Level2);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger()).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 0);
}

TEST(GraphTransformationTests, TransposeMatmulFusion) {
  auto model_uri = MODEL_FOLDER "fusion/transpose_matmul_4d_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["TransposeMatMul"] == 1);
}

TEST(GraphTransformationTests, Gemm_LeakyRelu_Fusion) {
  auto model_uri = MODEL_FOLDER "gemm_activation_fusion/gemm_activation_fusion.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count1 = CountOpsInGraph(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GemmActivationFusion>(), TransformerLevel::Level2);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger()).IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["LeakyRelu"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 0);
  ASSERT_TRUE(op_to_count["FusedGemm"] == 1);
}
#endif

TEST(GraphTransformationTests, FuseConvBnAddMulFloat16) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-add-mul-float16.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, &DefaultLoggingManager()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvAddFusion>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvBNFusion>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvMulFusion>());
  session_object.RegisterGraphTransformer(std::move(rule_transformer_L1), TransformerLevel::Level1);

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

  ASSERT_EQ(1u, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims_prod);
  //Use reinterpret_cast to bypass a gcc bug: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=51213
  ASSERT_EQ(*reinterpret_cast<const std::vector<int64_t>*>(&expected_shape), *reinterpret_cast<const std::vector<int64_t>*>(&rtensor.Shape()));
  const std::vector<MLFloat16> found(rtensor.template Data<MLFloat16>(),
                                     rtensor.template Data<MLFloat16>() + expected_dims_prod.size());
  ASSERT_EQ(expected_values_prod, found);
}

TEST(GraphTransformationTests, ReluClip6Fusion) {
  // Clip op schema changed for opset version 11. Until Clip op is updated in ORT hard coding this model to use
  // older opset.
  Model model("ReluClip6Fusion", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 10}},
              {}, DefaultLoggingManager().DefaultLogger());
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

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<FuseReluClip>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger()).IsOK());

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

// test handling of Clip 11
TEST(GraphTransformationTests, ReluClip11Fusion) {
  Model model("ReluClip6Fusion", false, DefaultLoggingManager().DefaultLogger());  //, true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 11}}, {});
  auto& graph = model.MainGraph();

  std::vector<NodeArg*> inputs;
  std::vector<NodeArg*> outputs;

  TypeProto input_tensor_type;
  input_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  input_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  TypeProto float16_tensor_type;
  float16_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT16);
  float16_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  // 4 paths in the model, each with Relu followed by Clip to test different aspects of Clip 11 handling
  // One has a Clip with mutable 'min' (don't fuse)
  // One has a Clip with constant 'min' < 0 (fuse and update 'min')
  // One has a Clip with constant 'min' > 0 (fuse and leave 'min')
  // One has a Clip with no 'min' (fuse and update to set min to 0 using type info from 'input')
  auto& input0 = graph.GetOrCreateNodeArg("input_0", &input_tensor_type);
  auto& input1 = graph.GetOrCreateNodeArg("input_1", &float16_tensor_type);
  auto& input2 = graph.GetOrCreateNodeArg("input_2", &input_tensor_type);
  auto& input3 = graph.GetOrCreateNodeArg("input_3", &input_tensor_type);

  auto& min_input_0 = graph.GetOrCreateNodeArg("min_input_0", &input_tensor_type);
  auto& min_input_1 = graph.GetOrCreateNodeArg("min_input_1", &float16_tensor_type);
  auto& min_input_2 = graph.GetOrCreateNodeArg("min_input_2", &input_tensor_type);

  // add initializer for min_input_1 so it's constant
  TensorProto const_min_1;
  Initializer i1(TensorProto_DataType_FLOAT16, "min_input_1", {1});
  i1.data<MLFloat16>()->val = math::floatToHalf(-1.f);
  i1.ToProto(const_min_1);
  graph.AddInitializedTensor(const_min_1);

  TensorProto const_min_2;
  Initializer i2(TensorProto_DataType_FLOAT, "min_input_2", {1});
  *i2.data<float>() = 1.f;
  i2.ToProto(const_min_2);
  graph.AddInitializedTensor(const_min_2);

  auto& relu0_output = graph.GetOrCreateNodeArg("relu0_output", &input_tensor_type);
  auto& relu1_output = graph.GetOrCreateNodeArg("relu1_output", &float16_tensor_type);
  auto& relu2_output = graph.GetOrCreateNodeArg("relu2_output", &input_tensor_type);
  auto& relu3_output = graph.GetOrCreateNodeArg("relu3_output", &input_tensor_type);

  auto& clip0_output = graph.GetOrCreateNodeArg("clip0_output", &input_tensor_type);
  auto& clip1_output = graph.GetOrCreateNodeArg("clip1_output", &float16_tensor_type);
  auto& clip2_output = graph.GetOrCreateNodeArg("clip2_output", &input_tensor_type);
  auto& clip3_output = graph.GetOrCreateNodeArg("clip3_output", &input_tensor_type);

  graph.AddNode("relu0", "Relu", "Relu0", {&input0}, {&relu0_output});
  graph.AddNode("relu1", "Relu", "Relu1", {&input1}, {&relu1_output});
  graph.AddNode("relu2", "Relu", "Relu2", {&input2}, {&relu2_output});
  graph.AddNode("relu3", "Relu", "Relu3", {&input3}, {&relu3_output});

  auto& clip0 = graph.AddNode("clip0", "Clip", "Clip with mutable min", {&relu0_output, &min_input_0}, {&clip0_output});
  auto& clip1 = graph.AddNode("clip1", "Clip", "Clip with constant min < 0", {&relu1_output, &min_input_1}, {&clip1_output});
  auto& clip2 = graph.AddNode("clip2", "Clip", "Clip with constant min > 0", {&relu2_output, &min_input_2}, {&clip2_output});
  auto& clip3 = graph.AddNode("clip3", "Clip", "Clip with no min", {&relu3_output}, {&clip3_output});

  graph.SetInputs({&input0, &input1, &input2, &input3, &min_input_0});
  auto status = graph.Resolve();
  EXPECT_EQ(status, Status::OK()) << status;

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 4);

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<FuseReluClip>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  status = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(status.IsOK()) << status;

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 1) << "All except the first Relu should have been fused";

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Relu") {
      EXPECT_TRUE(node.Name() == "relu0") << "relu0 should be the only Relu node left";
    }

    if (node.OpType() == "Clip") {
      auto* min_input = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());

      if (&node == &clip0) {
        EXPECT_TRUE(min_input == nullptr) << "clip0 should not have been fused as min_input_0 is not constant";
      } else {
        EXPECT_TRUE(min_input != nullptr)
            << node.Name() << " should have been fused and have a constant initializer for 'min'";

        auto type = min_input->data_type();

        if (&node == &clip1) {
          // fusion with float16 data and min set to 0
          EXPECT_EQ(type, ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_FLOAT16);
          MLFloat16 value = *Initializer(*min_input, graph.ModelPath()).data<MLFloat16>();
          EXPECT_EQ(math::halfToFloat(value.val), 0.f) << "Min was not 0.f. Got:" << math::halfToFloat(value.val);
        } else if (&node == &clip2) {
          // fusion with float data and min untouched
          EXPECT_EQ(type, ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_FLOAT);
          float value = *Initializer(*min_input, graph.ModelPath()).data<float>();
          EXPECT_EQ(value, 1.0) << "Min should have remained unchanged but is now " << value;
        } else if (&node == &clip3) {
          // fusion with no min so type comes from input
          EXPECT_EQ(type, ONNX_NAMESPACE::TensorProto::DataType::TensorProto_DataType_FLOAT);
          float value = *Initializer(*min_input, graph.ModelPath()).data<float>();
          EXPECT_EQ(value, 0.f) << "Min was not 0.f. Got:" << value;

        } else {
          EXPECT_TRUE(false) << "Unexpected node " << node.Name();
        }
      }
    }
  }
}

// Test Reshape Fusion with 2 constant initializers for Concat inputs.
TEST(GraphTransformationTests, ReshapeFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/reshape.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  ASSERT_TRUE(op_to_count["Concat"] == 0);
  ASSERT_TRUE(op_to_count["Reshape"] == 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 4);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], 12);
      EXPECT_EQ(val[3], 64);
    }
  }
}

// Test Reshape Fusion with one constant initializer for Concat inputs.
TEST(GraphTransformationTests, ReshapeFusionOneConstTest) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_one_const.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 3);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 0);
      EXPECT_EQ(val[2], 768);
    }
  }
}

#ifndef DISABLE_CONTRIB_OPS

static void ValidateAttention(Graph& graph) {
  // Validate the merged weights (initializer) input for Attention node.
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Attention") {
      int64_t expected_heads = 2;
      ASSERT_TRUE(optimizer_utils::IsAttributeWithExpectedValue(node, "num_heads", expected_heads));

      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(initializer->size(), 192);

      // Validate two rows (2x24 items) for sanity check.
      std::vector<double> expected_value = {
          -0.10791015625,
          -0.04193115234375,
          0.09051513671875,
          0.025787353515625,
          -0.11572265625,
          -0.126953125,
          -0.043304443359375,
          -0.02984619140625,
          0.022125244140625,
          -0.017730712890625,
          -0.03265380859375,
          -0.05108642578125,
          0.0423583984375,
          0.112060546875,
          0.080810546875,
          0.09375,
          -0.03643798828125,
          0.02862548828125,
          0.039764404296875,
          0.06097412109375,
          -0.002288818359375,
          -0.10797119140625,
          -0.01171875,
          0.041717529296875,

          0.033538818359375,
          -0.05755615234375,
          -0.04986572265625,
          -0.01558685302734375,
          -0.0352783203125,
          0.03546142578125,
          0.05218505859375,
          0.005565643310546875,
          -0.043182373046875,
          -0.05010986328125,
          -0.063720703125,
          -0.00824737548828125,
          0.1492919921875,
          0.048431396484375,
          -0.0482177734375,
          -0.1123046875,
          0.032196044921875,
          0.0135650634765625,
          0.020233154296875,
          -0.05084228515625,
          -0.011260986328125,
          -0.1241455078125,
          -0.0101165771484375,
          -0.00490570068359375};

      const float* data = initializer->data<float>();
      for (size_t i = 0; i < expected_value.size(); i++) {
        EXPECT_EQ(data[i], static_cast<float>(expected_value[i]));
      }

      tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[2]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      auto initializer2 = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(initializer2->size(), 24);

      std::vector<double> expected_value2 = {
          -0.23681640625,
          -0.16552734375,
          0.2191162109375,
          -0.1756591796875,
          -0.03460693359375,
          -0.05316162109375,
          -0.336181640625,
          -0.253662109375,
          0.0246734619140625,
          0.011993408203125,
          0.0178375244140625,
          0.00998687744140625,
          0.0255126953125,
          0.076416015625,
          -0.040771484375,
          0.0107879638671875,
          -0.005893707275390625,
          -0.00916290283203125,
          0.04541015625,
          0.0159454345703125,
          -0.0029163360595703125,
          -0.03472900390625,
          0.0535888671875,
          0.0091094970703125};

      const float* data2 = initializer2->data<float>();
      for (size_t i = 0; i < expected_value2.size(); i++) {
        EXPECT_EQ(data2[i], static_cast<float>(expected_value2[i]));
      }
    }
  }
}

// Test Attention Fusion with int32 mask
TEST(GraphTransformationTests, AttentionFusionInt32Test) {
  auto model_uri = MODEL_FOLDER "fusion/attention_int32_mask.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<AttentionFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
  EXPECT_EQ(op_to_count["Attention"], 1);

  ValidateAttention(graph);
}

// Test Attention Fusion with int64 mask and symbolic batch dimension
TEST(GraphTransformationTests, AttentionFusionInt64Test) {
  auto model_uri = MODEL_FOLDER "fusion/attention_symbolic_batch.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<AttentionFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Cast"], 1);  // Cast for int64 mask to int32
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
  EXPECT_EQ(op_to_count["Attention"], 1);

  ValidateAttention(graph);
}

TEST(GraphTransformationTests, GeluFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/gelu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Gelu"] == 1);
}

TEST(GraphTransformationTests, BiasGeluTest) {
  auto model_uri = MODEL_FOLDER "fusion/bias_gelu_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<BiasGelu>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Gelu"] == 0);
  ASSERT_TRUE(op_to_count["BiasGelu"] == 1);
}

// Test Gelu -> FastGelu
TEST(GraphTransformationTests, GeluApproximation_Gelu) {
  auto model_uri = MODEL_FOLDER "approximation/gelu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluApproximation>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Gelu"], 0);
  EXPECT_EQ(op_to_count["FastGelu"], 1);
}

// Test AddGeluFusion -> FastGelu
TEST(GraphTransformationTests, GeluApproximation_Gelu_Add_Bias) {
  auto model_uri = MODEL_FOLDER "approximation/gelu_add_bias.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluApproximation>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["BiasGelu"], 0);
  EXPECT_EQ(op_to_count["FastGelu"], 1);
}

// Test MatMul & AddGeluFusion -> MatMul & FastGelu
TEST(GraphTransformationTests, GeluApproximation_Gelu_Add_MatMul) {
  auto model_uri = MODEL_FOLDER "approximation/gelu_add_matmul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluApproximation>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["BiasGelu"], 0);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["FastGelu"], 1);
}

TEST(GraphTransformationTests, FastGeluFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 2);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST(GraphTransformationTests, FastGeluUseGraphInputFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST(GraphTransformationTests, FastGeluWithBiasFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu_with_bias.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<BiasGelu>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST(GraphTransformationTests, FastGeluWithBiasUseGraphInputFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu_with_bias_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<BiasGelu>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST(GraphTransformationTests, FastGeluFusionTest2) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu2.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST(GraphTransformationTests, FastGeluUseGraphInputFusionTest2) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu2_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST(GraphTransformationTests, FastGeluWithBiasFusionTest2) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu2_with_bias.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<BiasGelu>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST(GraphTransformationTests, FastGeluWithBiasUseGraphInputFusionTest2) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu2_with_bias_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<BiasGelu>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST(GraphTransformationTests, LayerNormFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/layer_norm.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<LayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Sub"] == 0);
  ASSERT_TRUE(op_to_count["ReduceMean"] == 0);
  ASSERT_TRUE(op_to_count["Pow"] == 0);
  ASSERT_TRUE(op_to_count["Sqrt"] == 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "LayerNormalization") {
      // LayerNormalization should have three inputs.
      EXPECT_EQ(node.InputDefs().size(), 3u) << "LayerNormalization number of inputs does not equal to 3. Got:" << node.InputDefs().size();
      // LayerNormalization input "scale" and "bias" should have the same dimension.
      const TensorShapeProto* scale_shape = node.InputDefs()[1]->Shape();
      const TensorShapeProto* bias_shape = node.InputDefs()[2]->Shape();
      EXPECT_EQ(scale_shape->dim_size(), 1) << "LayerNormalization scale should be 1D. Got: " << scale_shape->dim_size();
      EXPECT_EQ(bias_shape->dim_size(), 1) << "LayerNormalization bias should be 1D. Got: " << bias_shape->dim_size();
      EXPECT_EQ(scale_shape->dim(0).dim_value(), bias_shape->dim(0).dim_value());
    } else {
      EXPECT_TRUE(false) << "Unexpected node " << node.Name();
    }
  }
}

TEST(GraphTransformationTests, LayerNormWithSubDupFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/layer_norm_sub_dup.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<LayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Sub"] == 0);
  ASSERT_TRUE(op_to_count["ReduceMean"] == 0);
  ASSERT_TRUE(op_to_count["Pow"] == 0);
  ASSERT_TRUE(op_to_count["Sqrt"] == 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "LayerNormalization") {
      // LayerNormalization should have three inputs.
      EXPECT_EQ(node.InputDefs().size(), 3u) << "LayerNormalization number of inputs does not equal to 3. Got:" << node.InputDefs().size();
      // LayerNormalization input "scale" and "bias" should have the same dimension.
      const TensorShapeProto* scale_shape = node.InputDefs()[1]->Shape();
      const TensorShapeProto* bias_shape = node.InputDefs()[2]->Shape();
      EXPECT_EQ(scale_shape->dim_size(), 1) << "LayerNormalization scale should be 1D. Got: " << scale_shape->dim_size();
      EXPECT_EQ(bias_shape->dim_size(), 1) << "LayerNormalization bias should be 1D. Got: " << bias_shape->dim_size();
      EXPECT_EQ(scale_shape->dim(0).dim_value(), bias_shape->dim(0).dim_value());
    } else {
      EXPECT_TRUE(false) << "Unexpected node " << node.Name();
    }
  }
}

static void TestSkipLayerNormFusion(const std::basic_string<ORTCHAR_T>& file_path, int add_count, int ln_count, int skip_ln_count) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<LayerNormFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<SkipLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == add_count);
  ASSERT_TRUE(op_to_count["Sub"] == 0);
  ASSERT_TRUE(op_to_count["ReduceMean"] == 0);
  ASSERT_TRUE(op_to_count["Pow"] == 0);
  ASSERT_TRUE(op_to_count["Sqrt"] == 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == ln_count);
  ASSERT_TRUE(op_to_count["SkipLayerNormalization"] == skip_ln_count);
}

TEST(GraphTransformationTests, SkipLayerNormFusionTest) {
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format1.onnx", 0, 0, 1);
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format2.onnx", 0, 0, 1);
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format3.onnx", 0, 0, 1);
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format1_partial.onnx", 1, 0, 1);
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format2_partial.onnx", 1, 0, 1);
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format3_no_fusion.onnx", 1, 1, 0);
}

TEST(GraphTransformationTests, EmbedLayerNormFusionFormat1) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format1.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["ReduceSum"] == 0);
  ASSERT_TRUE(op_to_count["Attention"] == 1);
  ASSERT_TRUE(op_to_count["SkipLayerNormalization"] == 0);
  ASSERT_TRUE(op_to_count["EmbedLayerNormalization"] == 1);
}

TEST(GraphTransformationTests, EmbedLayerNormFusionFormat2) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format2.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["Expand"] == 0);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 0);
  ASSERT_TRUE(op_to_count["NonZero"] == 0);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
  ASSERT_TRUE(op_to_count["Squeeze"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["ReduceSum"] == 0);
  ASSERT_TRUE(op_to_count["Attention"] == 1);
  ASSERT_TRUE(op_to_count["SkipLayerNormalization"] == 0);
  ASSERT_TRUE(op_to_count["EmbedLayerNormalization"] == 1);
}

TEST(GraphTransformationTests, EmbedLayerNormFusionFormat3) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format3.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Expand"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["LayerNormalization"], 0);
  EXPECT_EQ(op_to_count["SkipLayerNormalization"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 0);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Cast"], 3);
  EXPECT_EQ(op_to_count["Attention"], 1);
  EXPECT_EQ(op_to_count["EmbedLayerNormalization"], 1);
}

TEST(GraphTransformationTests, EmbedLayerNormFusionFormat4) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format4.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["Expand"] == 0);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Concat"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 0);
  ASSERT_TRUE(op_to_count["NonZero"] == 0);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
  ASSERT_TRUE(op_to_count["Squeeze"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["ReduceSum"] == 0);
  ASSERT_TRUE(op_to_count["Attention"] == 1);
  ASSERT_TRUE(op_to_count["SkipLayerNormalization"] == 0);
  ASSERT_TRUE(op_to_count["EmbedLayerNormalization"] == 1);
}

TEST(GraphTransformationTests, EmbedLayerNormFusionFormat5) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format5.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, DefaultLoggingManager().DefaultLogger()).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, DefaultLoggingManager().DefaultLogger());
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["LayerNormalization"], 0);
  EXPECT_EQ(op_to_count["SkipLayerNormalization"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 0);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Cast"], 3);
  EXPECT_EQ(op_to_count["Attention"], 1);
  EXPECT_EQ(op_to_count["EmbedLayerNormalization"], 1);

  // Validate the position embedding input.
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "EmbedLayerNormalization") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[3]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(initializer->size(), 12);

      std::vector<double> expected_value = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0};

      const float* data = initializer->data<float>();
      for (size_t i = 0; i < expected_value.size(); i++) {
        EXPECT_EQ(data[i], static_cast<float>(expected_value[i]));
      }
    }
  }
}

#endif

}  // namespace test
}  // namespace onnxruntime
