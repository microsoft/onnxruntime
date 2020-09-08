// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#include <random>
#include "core/graph/onnx_protobuf.h"

#include "asserts.h"
#include "core/framework/data_types.h"
#include "core/framework/ml_value.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/attention_fusion.h"
#include "core/optimizer/bias_gelu_fusion.h"
#include "core/optimizer/computation_reduction.h"
#include "core/optimizer/cast_elimination.h"
#include "core/optimizer/constant_folding.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_bn_fusion.h"
#include "core/optimizer/conv_mul_fusion.h"
#include "core/optimizer/dropout_elimination.h"
#include "core/optimizer/dynamic_quantize_matmul_fusion.h"
#include "core/optimizer/embed_layer_norm_fusion.h"
#include "core/optimizer/expand_elimination.h"
#include "core/optimizer/fast_gelu_fusion.h"
#include "core/optimizer/gelu_approximation.h"
#include "core/optimizer/gelu_fusion.h"
#include "core/optimizer/gemm_activation_fusion.h"
#include "core/optimizer/graph_transformer.h"
#include "core/optimizer/graph_transformer_mgr.h"
#include "core/optimizer/identity_elimination.h"
#include "core/optimizer/initializer.h"
#include "core/optimizer/layer_norm_fusion.h"
#include "core/optimizer/matmul_add_fusion.h"
#include "core/optimizer/matmul_scale_fusion.h"
#include "core/optimizer/matmul_transpose_fusion.h"
#include "core/optimizer/relu_clip_fusion.h"
#include "core/optimizer/reshape_fusion.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "core/optimizer/shape_to_initializer.h"
#include "core/optimizer/skip_layer_norm_fusion.h"
#include "core/optimizer/slice_elimination.h"
#include "core/optimizer/unsqueeze_elimination.h"
#include "core/optimizer/utils.h"
#include "core/platform/env.h"
#include "core/session/inference_session.h"
#include "core/util/math.h"
#include "gtest/gtest.h"
#include "test/capturing_sink.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/compare_ortvalue.h"
#include "test/framework/test_utils.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/providers/provider_test_utils.h"
#include "test/test_environment.h"
#include "test/util/include/default_providers.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

TEST_F(GraphTransformationTests, IdentityElimination) {
  auto model_uri = MODEL_FOLDER "abs-id-max.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 1);

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<EliminateIdentity>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 0);
}

TEST_F(GraphTransformationTests, DropoutElimination) {
  auto model_uri = MODEL_FOLDER "dropout.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 5);
  ASSERT_TRUE(op_to_count["Dropout"] == 6);

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<EliminateDropout>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  // Of the 6 Dropout nodes in the graph, all but the ones named `d1` and `d6` should have been removed.
  // A Dropout node can be removed if its second, optional output `mask` is either missing or unused downstream.
  // `d1` cannot be removed because an Identity node has its `mask` output as an input;
  // `d6` cannot be removed because its `mask` output is marked as a graph output.
  ASSERT_TRUE(op_to_count["Identity"] == 5);
  ASSERT_TRUE(op_to_count["Dropout"] == 2);
}

TEST_F(GraphTransformationTests, SliceElimination) {
  std::vector<std::basic_string<ORTCHAR_T>> model_names = {ORT_TSTR("slice-v1-elim.onnx"), ORT_TSTR("slice-v11-elim.onnx")};
  for (const auto& model_name : model_names) {
    auto model_uri = MODEL_FOLDER + model_name;
    std::shared_ptr<Model> model;
    ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
    Graph& graph = model->MainGraph();
    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    int initial_slice_num = op_to_count["Slice"];

    auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
    rule_transformer_L1->Register(onnxruntime::make_unique<EliminateSlice>());
    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    op_to_count = CountOpsInGraph(graph);
    // Only one Slice operator is redundant and is removed.
    ASSERT_TRUE(op_to_count["Slice"] == --initial_slice_num);
  }
}

TEST_F(GraphTransformationTests, ConstantFolding) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 2);
  std::unique_ptr<CPUExecutionProvider> e =
      onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ConstantFolding>(*e.get()), TransformerLevel::Level1);

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST_F(GraphTransformationTests, ConstantFoldingNodesOnDifferentEP) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-mul-add-unsqueeze.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 2);
  std::unique_ptr<CPUExecutionProvider> e =
      onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ConstantFolding>(*e.get()), TransformerLevel::Level1);

  // assign all nodes to CUDA. the constant folding should override this to perform the constant folding on cpu
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kCudaExecutionProvider);
  }

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);

  // all remaining nodes should still be on CUDA
  for (auto& node : graph.Nodes()) {
    EXPECT_STREQ(node.GetExecutionProviderType().c_str(), kCudaExecutionProvider);
  }
}

TEST_F(GraphTransformationTests, ConstantFoldingSubgraph) {
  TensorProto value_tensor;
  value_tensor.add_dims(1);
  value_tensor.add_float_data(1.f);
  value_tensor.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

  TypeProto float_tensor_type;
  float_tensor_type.mutable_tensor_type()->set_elem_type(TensorProto_DataType_FLOAT);
  float_tensor_type.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(1);

  auto create_subgraph = [&](GraphProto& graph_proto) {
    // create subgraph that has an Add node to add a local and parent graph initializer
    Model model("ConstantFoldingSubgraphTest_subgraph", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, *logger_);
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

    ASSERT_STATUS_OK(graph.Resolve());
    graph_proto = graph.ToGraphProto();
  };

  Model model("ConstantFoldingSubgraphTest_main_graph", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{kOnnxDomain, 12}}, {}, *logger_);
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

  ASSERT_STATUS_OK(graph.Resolve());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 2);  // one in each subgraph
  std::unique_ptr<CPUExecutionProvider> e =
      onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ConstantFolding>(*e.get()), TransformerLevel::Level1);

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0)
      << "Constant folding should have been able to remove the Add node in both subgraphs";
}

TEST_F(GraphTransformationTests, ConstantFoldingWithShapeToInitializer) {
  auto model_uri = MODEL_FOLDER "fusion/constant_folding_with_shape_to_initializer.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 2);
  ASSERT_TRUE(op_to_count["MatMul"] == 2);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 3);

  std::unordered_set<std::string> compatible_eps;
  std::unordered_set<std::string> excluded_initializers;
  excluded_initializers.insert("matmul_weight");
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unique_ptr<CPUExecutionProvider> e =
      onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  graph_transformation_mgr.Register(onnxruntime::make_unique<ConstantFolding>(*e.get(), compatible_eps, excluded_initializers), TransformerLevel::Level1);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 2);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST_F(GraphTransformationTests, ConstantFoldingWithScalarShapeToInitializer) {
  auto model_uri = MODEL_FOLDER "fusion/constant_folding_with_scalar_shape_to_initializer.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 1);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);

  std::unordered_set<std::string> compatible_eps;
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  std::unique_ptr<CPUExecutionProvider> e =
      onnxruntime::make_unique<CPUExecutionProvider>(CPUExecutionProviderInfo());
  graph_transformation_mgr.Register(onnxruntime::make_unique<ConstantFolding>(*e.get(), compatible_eps), TransformerLevel::Level1);

  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 0);
  ASSERT_TRUE(op_to_count["ConstantOfShape"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 1);
}

TEST_F(GraphTransformationTests, ShapeToInitializer) {
  auto model_uri = MODEL_FOLDER "shape-add.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Shape"] == 4);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  rule_transformer_L1->Register(onnxruntime::make_unique<ShapeToInitializer>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  op_to_count = CountOpsInGraph(graph);
  // Two of the Shapes are not eliminated because:
  // One includes a symbolic dimension.
  // Another one includes a negative dimension
  ASSERT_TRUE(op_to_count["Shape"] == 2);
}

// Check transformations in the case of a subgraph with constant inputs.
TEST_F(GraphTransformationTests, SubgraphWithConstantInputs) {
  auto model_uri = MODEL_FOLDER "constant-subgraph.onnx";

  SessionOptions so;
  so.graph_optimization_level = TransformerLevel::Level2;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_STATUS_OK(session_object.Load(model_uri));

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));

  ASSERT_STATUS_OK(session_object.Initialize());

  NameMLValMap feeds;
  RunOptions run_options;

  std::vector<std::string> output_names = {"output"};
  std::vector<OrtValue> fetches;

  ASSERT_STATUS_OK(session_object.Run(run_options, feeds, output_names, &fetches));
}

TEST_F(GraphTransformationTests, FuseConvBNNoBias) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
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

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["BatchNormalization"] == 0);

  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      ASSERT_EQ(node.OutputDefs()[0]->Name(), bn_output_name)
          << "fusion should produce the same output name as the last node";
    }
  }
}

TEST_F(GraphTransformationTests, DontFuseConvWithBNWithOptionalOutputs) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
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

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["BatchNormalization"] == 1);
}

TEST_F(GraphTransformationTests, FuseConvBNMulAddUnsqueeze) {
  std::vector<std::basic_string<ORTCHAR_T>> test_models = {ORT_TSTR("fusion/fuse-conv-bn-mul-add-unsqueeze.onnx"),
                                                           ORT_TSTR("fusion/fuse-conv-bn-mul-add-unsqueeze.negative_axes.onnx"),
                                                           ORT_TSTR("fusion/fuse-conv-bn-mul-add-unsqueeze-no-bias.onnx")};
  for (const auto& model : test_models) {
    auto model_uri = MODEL_FOLDER + model;

    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
    rule_transformer_L1->Register(onnxruntime::make_unique<UnsqueezeElimination>());
    rule_transformer_L1->Register(onnxruntime::make_unique<ConvAddFusion>());
    rule_transformer_L1->Register(onnxruntime::make_unique<ConvBNFusion>());
    rule_transformer_L1->Register(onnxruntime::make_unique<ConvMulFusion>());
    graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count["BatchNormalization"] == 0);
    ASSERT_TRUE(op_to_count["Mul"] == 0);
    ASSERT_TRUE(op_to_count["Add"] == 0);
    ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
  }
}

#ifndef DISABLE_CONTRIB_OPS
TEST_F(GraphTransformationTests, FuseConvActivation) {
  std::unordered_map<std::basic_string<ORTCHAR_T>, std::string> model_to_op_name{{ORT_TSTR("fusion/conv_relu.onnx"), "Relu"},
                                                                                 {ORT_TSTR("fusion/conv_clip.onnx"), "Clip"},
                                                                                 {ORT_TSTR("fusion/conv_sigmoid.onnx"), "Sigmoid"},
                                                                                 {ORT_TSTR("fusion/conv_tanh.onnx"), "Tanh"},
                                                                                 {ORT_TSTR("fusion/conv_leakyrelu.onnx"), "LeakyRelu"}};

  for (const auto& model : model_to_op_name) {
    auto model_uri = MODEL_FOLDER + model.first;
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[model.second] >= 1);

    // Apply transformer
    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    graph_transformation_mgr.Register(onnxruntime::make_unique<ConvActivationFusion>(), TransformerLevel::Level2);
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

    op_to_count = CountOpsInGraph(graph);
    ASSERT_TRUE(op_to_count[model.second] == 0);
  }
}

TEST_F(GraphTransformationTests, FuseConvClip11Activation) {
  auto model_uri = MODEL_FOLDER "fusion/conv_clip11.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Clip"], 3);

  // Apply transformer
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ConvActivationFusion>(), TransformerLevel::Level2);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

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

TEST_F(GraphTransformationTests, FuseConvMulNoBias) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-mul-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<UnsqueezeElimination>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvMulFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

TEST_F(GraphTransformationTests, FuseConvAddNoBias) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-add-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<UnsqueezeElimination>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvAddFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] == 0);
}

// if IR version is 4 or higher the weights can be overridden if there's a matching graph input.
// check that we don't fuse if that is the case
TEST_F(GraphTransformationTests, NegativeFuseConvAddNoBias) {
  auto model_uri = MODEL_FOLDER "fusion/negative-fuse-conv-add-no-bias.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<UnsqueezeElimination>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvAddFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // Nodes are not fused because the weights to conv/add are not constants (they appear in the graph inputs).
  // Unsqueeze is also not eliminated as the initializer that is its input is also not constant
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] != 0);
  ASSERT_TRUE(op_to_count["Unsqueeze"] != 0);
}

TEST_F(GraphTransformationTests, FuseConvAddMul3D) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-add-mul-3d.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvAddFusion>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvMulFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
}

TEST_F(GraphTransformationTests, FuseConvAddMul3D_2) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-add-mul-3d-2.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvAddFusion>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvMulFusion>());
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
}

TEST_F(GraphTransformationTests, MatMulAddFusion_two_input) {
  auto model_uri = MODEL_FOLDER "matmul_add_fusion/2Input/model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MatMulAddFusion>(), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 1);
}

TEST_F(GraphTransformationTests, MatMulAddFusion_three_input) {
  auto model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MatMulAddFusion>(), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 1);
}

// Matmul+Add with shape [k]*[k,N]+[N], won't do the fusion
// We can do the fusion by changing shape to [1,k]*[k,N]+[1,N], then add a reshape [1,N]=>[N]
// This will bring extra cost. And there's only very limited gain to fuse Matmul+Add to Gemm
// Since the basic implementation is almost same
TEST_F(GraphTransformationTests, MatMulAddFusion_negitive_case) {
  auto model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/neg_model.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MatMulAddFusion>(), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Gemm"] == 0);
}

// Matmul+Add with shape [M,k]*[k,N]+[1,4], won't do the fusion
// 1,4 is not uni-directionally broadcast
TEST_F(GraphTransformationTests, MatMulAddFusion_NotBroadcastable) {
  auto model_uri = MODEL_FOLDER "matmul_add_fusion/matmul_add_not_broadcastable.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MatMulAddFusion>(), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["MatMul"] == 1);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Gemm"] == 0);
}

#ifndef DISABLE_CONTRIB_OPS
TEST_F(GraphTransformationTests, Gemm_Relu_three_input) {
  auto model_uri = MODEL_FOLDER "matmul_add_fusion/3Input/gemm_relu.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count1 = CountOpsInGraph(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GemmActivationFusion>(), TransformerLevel::Level2);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 0);
}

TEST_F(GraphTransformationTests, TransposeMatmulFusion) {
  auto model_uri = MODEL_FOLDER "fusion/transpose_matmul_4d_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["TransposeScaleMatMul"] == 1);
}

TEST_F(GraphTransformationTests, TransposeMatmulFusionOnTwoTranspose) {
  auto model_uri = MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_2_transpose.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["TransposeScaleMatMul"] == 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "TransposeScaleMatMul");
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
}

TEST_F(GraphTransformationTests, TransposeMatmulFusionOnThreeTranspose) {
  auto model_uri = MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_3_transpose.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Transpose"] == 0);
  ASSERT_TRUE(op_to_count["MatMul"] == 0);
  ASSERT_TRUE(op_to_count["TransposeScaleMatMul"] == 1);

  auto& node = *graph.Nodes().begin();
  ASSERT_TRUE(node.OpType() == "TransposeScaleMatMul");
  ASSERT_FALSE(static_cast<bool>(node.GetAttributes().at("transA").i()));
  ASSERT_TRUE(static_cast<bool>(node.GetAttributes().at("transB").i()));
}

TEST_F(GraphTransformationTests, TransposeMatmulNoFusionOnInvalidPerm) {
  const std::vector<PathString> model_uris = {
      MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_invalid_perm.onnx",
      MODEL_FOLDER "fusion/transpose_matmul_4d_fusion_invalid_default_perm.onnx",
  };
  for (const auto& model_uri : model_uris) {
    std::shared_ptr<Model> p_model;
    ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
    Graph& graph = p_model->MainGraph();

    onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
    ASSERT_STATUS_OK(graph_transformation_mgr.Register(
        onnxruntime::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
    ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

    std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
    ASSERT_EQ(op_to_count["Transpose"], 1);
    ASSERT_EQ(op_to_count["MatMul"], 1);
    ASSERT_EQ(op_to_count["TransposeScaleMatMul"], 0);
  }
}

TEST_F(GraphTransformationTests, TransposeMatmulFusionFromTransposeScaleMatMul) {
  auto model_uri = MODEL_FOLDER "fusion/transpose_matmul_2d_fusion_from_transpose_scale_matmul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  float expected_alpha;
  {
    auto transpose_scale_matmul_node =
        std::find_if(
            graph.Nodes().cbegin(), graph.Nodes().cend(),
            [](const Node& node) { return node.Name() == "TransposeScaleMatMul"; });
    ASSERT_NE(transpose_scale_matmul_node, graph.Nodes().cend());
    expected_alpha = transpose_scale_matmul_node->GetAttributes().at("alpha").f();
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      onnxruntime::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 0);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["TransposeScaleMatMul"], 1);

  auto& transpose_scale_matmul_node = *graph.Nodes().begin();
  ASSERT_EQ(transpose_scale_matmul_node.OpType(), "TransposeScaleMatMul");
  ASSERT_FALSE(static_cast<bool>(transpose_scale_matmul_node.GetAttributes().at("transA").i()));
  ASSERT_FALSE(static_cast<bool>(transpose_scale_matmul_node.GetAttributes().at("transB").i()));
  ASSERT_EQ(transpose_scale_matmul_node.GetAttributes().at("alpha").f(), expected_alpha);
}

TEST_F(GraphTransformationTests, TransposeMatmulFusionWithPreservedTranspose) {
  auto model_uri = MODEL_FOLDER "fusion/transpose_matmul_2d_fusion_with_preserved_transpose.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      onnxruntime::make_unique<MatmulTransposeFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Transpose"], 1);
  ASSERT_EQ(op_to_count["MatMul"], 0);
  ASSERT_EQ(op_to_count["TransposeScaleMatMul"], 1);

  ASSERT_FALSE(graph.GraphResolveNeeded());
}

TEST_F(GraphTransformationTests, Gemm_LeakyRelu_Fusion) {
  auto model_uri = MODEL_FOLDER "gemm_activation_fusion/gemm_activation_fusion.onnx";

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count1 = CountOpsInGraph(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GemmActivationFusion>(), TransformerLevel::Level2);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["LeakyRelu"] == 0);
  ASSERT_TRUE(op_to_count["Gemm"] == 0);
  ASSERT_TRUE(op_to_count["FusedGemm"] == 1);
}
#endif

TEST_F(GraphTransformationTests, FuseConvBnAddMulFloat16) {
  auto model_uri = MODEL_FOLDER "fusion/fuse-conv-bn-add-mul-float16.onnx";

  SessionOptions so;
  so.session_logid = "GraphTransformationTests.LoadModelToTransform";
  InferenceSession session_object{so, GetEnvironment()};
  ASSERT_TRUE(session_object.Load(model_uri).IsOK());

  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformerL1");
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvAddFusion>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvBNFusion>());
  rule_transformer_L1->Register(onnxruntime::make_unique<ConvMulFusion>());
  ASSERT_STATUS_OK(session_object.RegisterGraphTransformer(std::move(rule_transformer_L1), TransformerLevel::Level1));

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

TEST_F(GraphTransformationTests, ReluClip6Fusion) {
  // Clip op schema changed for opset version 11. Until Clip op is updated in ORT hard coding this model to use
  // older opset.
  Model model("ReluClip6Fusion", true, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 10}},
              {}, *logger_);
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
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_).IsOK());

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
TEST_F(GraphTransformationTests, ReluClip11Fusion) {
  std::unordered_map<std::string, int> domain_to_version;
  domain_to_version[kOnnxDomain] = 11;
  Model model("ReluClip6Fusion", false, ModelMetaData(), PathString(), IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version, std::vector<ONNX_NAMESPACE::FunctionProto>(),
              *logger_);  //, true, ModelMetaData(), IOnnxRuntimeOpSchemaRegistryList(), {{"", 11}}, {});
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
  status = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
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
TEST_F(GraphTransformationTests, ReshapeFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/reshape.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
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
TEST_F(GraphTransformationTests, ReshapeFusionOneConstTest) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_one_const.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
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

// Test Reshape Fusion with an internal node being the output of the graph.
TEST_F(GraphTransformationTests, ReshapeFusionInternalNodeIsOutput) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_internal_node_is_graph_output.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 1);
  ASSERT_EQ(op_to_count["Gather"], 1);
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
      EXPECT_EQ(val[2], -1);
    }
  }
}

// Test Reshape Fusion where some of the internal nodes are reused:
// A Shape is used in two Gather's, and the third Gather is the graph output.
TEST_F(GraphTransformationTests, ReshapeFusionInternalReuseTest) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_internal_nodes_reused.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 1);
  ASSERT_EQ(op_to_count["Gather"], 1);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);

      auto initializer = onnxruntime::make_unique<Initializer>(*tensor_proto, graph.ModelPath());
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_INT64);
      EXPECT_EQ(initializer->size(), 5);

      const int64_t* val = initializer->data<int64_t>();
      EXPECT_EQ(val[0], 0);
      EXPECT_EQ(val[1], 128);
      EXPECT_EQ(val[2], 0);
      EXPECT_EQ(val[3], 0);
      EXPECT_EQ(val[4], -1);
    } else if (node.OpType() == "Shape") {
      EXPECT_EQ(node.Name(), "shape2");
    } else if (node.OpType() == "Gather") {
      EXPECT_EQ(node.Name(), "gather3");
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionGraphInputsTest) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_with_graph_inputs.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 1);
  ASSERT_EQ(op_to_count["Gather"], 1);
  ASSERT_EQ(op_to_count["Unsqueeze"], 1);
  ASSERT_EQ(op_to_count["Concat"], 1);
  ASSERT_EQ(op_to_count["Reshape"], 1);
}

TEST_F(GraphTransformationTests, ReshapeFusionMultipleValuesInInitializerSubgraphTest) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_multiple_values_in_initializer_tensor_1.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count_orig = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  // The optimization does not apply.
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
      EXPECT_EQ(val[0], 1);
      EXPECT_EQ(val[1], 200);
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionMultipleValuesInInitializerAppliesTest) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_multiple_values_in_initializer_tensor_2.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
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
      EXPECT_EQ(val[0], 1);
      EXPECT_EQ(val[1], 200);
      EXPECT_EQ(val[2], 0);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionAnotherGraphInput) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_input_is_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  // The optimization does not apply.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Concat"], 0);
  ASSERT_EQ(op_to_count["Reshape"], 1);
}

TEST_F(GraphTransformationTests, ReshapeFusionOverridableInitializer) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_overridable_initializer.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();
  std::map<std::string, int> op_to_count_orig = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  // The optimization does not apply.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count_orig, op_to_count);
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraphMultipleOutputs) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph_multiple_outputs.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  // The optimization applies but certain paths with multiple outputs/graph outputs are not removed.
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 3);
  ASSERT_EQ(op_to_count["Gather"], 1);
  ASSERT_EQ(op_to_count["Unsqueeze"], 1);
  ASSERT_EQ(op_to_count["Slice"], 1);
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
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraph) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Slice"], 0);
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
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraphNotTriggered) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph_not_triggered.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  // Two of the branches leading to Concat are candidates to trigger the optimization
  // (Shape -> Gather -> Unsqueeze -> Concat).
  // But one of the subgraphs leading to the Concat node will not trigger the optimization
  // as an additional pad value of 1 is inserted thus making the inputs to the Concat -
  // [10], [20], and [1, 30]
  // Since the third branch will match the subgraph fusion, (it has more than 1 value in the tensor)
  // and hence the optimization will not be triggered eventually

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 3);
  ASSERT_EQ(op_to_count["Gather"], 2);
  ASSERT_EQ(op_to_count["Unsqueeze"], 2);
  ASSERT_EQ(op_to_count["Slice"], 1);
  ASSERT_EQ(op_to_count["Concat"], 1);
  ASSERT_EQ(op_to_count["Pad"], 1);
  ASSERT_EQ(op_to_count["Reshape"], 1);
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "Reshape") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[1]->Name());
      ASSERT_TRUE(tensor_proto == nullptr);  // No initializer as optimizer is not triggered
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraphWithDiv) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph_div.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Slice"], 0);
  ASSERT_EQ(op_to_count["Div"], 0);
  ASSERT_EQ(op_to_count["Squeeze"], 0);
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
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionConcatSubgraphWithMul) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_concat_subgraph_mul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(model_uri, p_model, nullptr, *logger_).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Shape"], 0);
  ASSERT_EQ(op_to_count["Gather"], 0);
  ASSERT_EQ(op_to_count["Unsqueeze"], 0);
  ASSERT_EQ(op_to_count["Slice"], 0);
  ASSERT_EQ(op_to_count["Mul"], 0);
  ASSERT_EQ(op_to_count["Squeeze"], 0);
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
      EXPECT_EQ(val[2], -1);
    }
  }
}

TEST_F(GraphTransformationTests, ReshapeFusionDistilBertTest) {
  auto model_uri = MODEL_FOLDER "fusion/reshape_fusion_distillbert.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ReshapeFusion>(), TransformerLevel::Level1);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_);
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
      EXPECT_EQ(val[1], -1);
      EXPECT_EQ(val[2], 2);
      EXPECT_EQ(val[3], 4);
    }
  }
}

TEST_F(GraphTransformationTests, ExpandElimination) {
  auto model_uri = MODEL_FOLDER "expand_elimination.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Expand"] == 6);

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<ExpandElimination>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Expand"] == 3);
}

TEST_F(GraphTransformationTests, CastElimination) {
  auto model_uri = MODEL_FOLDER "cast_elimination.onnx";
  std::shared_ptr<Model> model;
  ASSERT_TRUE(Model::Load(model_uri, model, nullptr, *logger_).IsOK());
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 7);

  auto rule_transformer_L1 = onnxruntime::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  rule_transformer_L1->Register(onnxruntime::make_unique<CastElimination>());
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level1);
  ASSERT_TRUE(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_).IsOK());

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Cast"] == 4);
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
TEST_F(GraphTransformationTests, AttentionFusionInt32Test) {
  auto model_uri = MODEL_FOLDER "fusion/attention_int32_mask.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<AttentionFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Attention"], 1);

  ValidateAttention(graph);
}

// Test Attention Fusion with int64 mask and symbolic batch dimension
TEST_F(GraphTransformationTests, AttentionFusionInt64Test) {
  auto model_uri = MODEL_FOLDER "fusion/attention_symbolic_batch.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<AttentionFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Cast"], 1);  // Cast for int64 mask to int32
  EXPECT_EQ(op_to_count["Attention"], 1);

  ValidateAttention(graph);
}

// Test Attention Fusion with float32 mask and no "cast" node in mask path
TEST_F(GraphTransformationTests, AttentionFusionFloat32Test) {
  auto model_uri = MODEL_FOLDER "fusion/attention_mask_no_cast.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<AttentionFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["Div"], 0);
  EXPECT_EQ(op_to_count["Sub"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["Attention"], 1);

  ValidateAttention(graph);
}

// Test GPT-2 Attention Fusion with float32 mask
TEST_F(GraphTransformationTests, AttentionFusionGPTWithPastAndMaskTest) {
  auto model_uri = MODEL_FOLDER "fusion/gpt2_past_mask_one_layer.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<AttentionFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["Attention"], 1);
}

// Test GPT-2 Attention Fusion without input mask
TEST_F(GraphTransformationTests, AttentionFusionGPTWithPastNoMaskTest) {
  auto model_uri = MODEL_FOLDER "fusion/gpt2_past_one_layer.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<AttentionFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["Attention"], 1);
}

// Test GPT-2 Attention Fusion without input mask and past state
TEST_F(GraphTransformationTests, AttentionFusionGPTTest) {
  auto model_uri = MODEL_FOLDER "fusion/gpt2_one_layer.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<AttentionFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["Attention"], 1);
}

TEST_F(GraphTransformationTests, AttentionFusionDistilBertTest) {
  auto model_uri = MODEL_FOLDER "fusion/attention_distilbert.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<AttentionFusion>(), TransformerLevel::Level2);
  auto ret1 = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret1.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["ReduceSum"], 0);
  EXPECT_EQ(op_to_count["Attention"], 1);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["Concat"], 0);
  EXPECT_EQ(op_to_count["Transpose"], 0);
  EXPECT_EQ(op_to_count["Softmax"], 0);
  EXPECT_EQ(op_to_count["Shape"], 0);
}

TEST_F(GraphTransformationTests, GeluFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/gelu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTestSwitchOrderFormat2) {
  auto model_uri = MODEL_FOLDER "fusion/gelu_format2_0.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTestFormat2) {
  auto model_uri = MODEL_FOLDER "fusion/gelu_format2_1.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTestFormat2GraphInput) {
  auto model_uri = MODEL_FOLDER "fusion/gelu_format2_1_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Gelu"] == 1);
}

TEST_F(GraphTransformationTests, GeluFusionTestFormat2GraphOutput) {
  auto model_uri = MODEL_FOLDER "fusion/gelu_format2_0_with_bias_use_graph_output.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<BiasGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Gelu"] == 0);
  ASSERT_TRUE(op_to_count["BiasGelu"] == 0);
}

TEST_F(GraphTransformationTests, BiasGeluTest) {
  auto model_uri = MODEL_FOLDER "fusion/bias_gelu_fusion.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<BiasGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Erf"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["Gelu"] == 0);
  ASSERT_TRUE(op_to_count["BiasGelu"] == 1);
}

// BiasGelu allows input switching based on input dimensions.
// This test validates the input edges are plugged correct in the optimized graph.
TEST_F(GraphTransformationTests, BiasGeluSwitchedInputOrder) {
  auto model_uri = MODEL_FOLDER "fusion/bias_gelu_fusion_format_2.onnx";

  // create inputs and outputs
  RandomValueGenerator random{};
  NameMLValMap feeds;

  OrtValue mlvalue_b_i;
  std::vector<int64_t> dims_b_i = {3072};
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_b_i,
                       random.Uniform<float>(dims_b_i, 0.0f, 1.0f), &mlvalue_b_i);
  feeds.insert(std::make_pair("B_I", mlvalue_b_i));

  OrtValue mlvalue_a_i;
  std::vector<int64_t> dims_a_i = {3, 512, 3072};
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_a_i,
                       random.Uniform<float>(dims_a_i, 0.0f, 1.0f), &mlvalue_a_i);
  feeds.insert(std::make_pair("A_I", mlvalue_a_i));

  std::vector<std::string> output_names;
  output_names.push_back("C");

  auto run_model_test = [&](TransformerLevel level, std::vector<OrtValue>& fetches) {
    SessionOptions session_options;
    session_options.graph_optimization_level = level;
    session_options.session_logid = "OptimizerTests";
    InferenceSession session{session_options, GetEnvironment()};
    ASSERT_TRUE(session.Load(model_uri).IsOK());
    ASSERT_TRUE(session.Initialize().IsOK());

    RunOptions run_options;
    ASSERT_STATUS_OK(session.Run(run_options, feeds, output_names, &fetches));
  };

  // run model with and w/o optimizations and compare the results
  std::vector<OrtValue> unoptimized_fetches;
  run_model_test(TransformerLevel::Default, unoptimized_fetches);

  std::vector<OrtValue> optimized_fetches;
  run_model_test(TransformerLevel::MaxLevel, optimized_fetches);

  // Compare results
  double per_sample_tolerance = 1e-3;
  double relative_per_sample_tolerance = 0.0;
  auto ret = CompareOrtValue(optimized_fetches[0], unoptimized_fetches[0], per_sample_tolerance, relative_per_sample_tolerance, false);
  EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
}

// Test Gelu -> FastGelu
TEST_F(GraphTransformationTests, GeluApproximation_Gelu) {
  auto model_uri = MODEL_FOLDER "approximation/gelu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluApproximation>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Gelu"], 0);
  EXPECT_EQ(op_to_count["FastGelu"], 1);
}

// Test AddGeluFusion -> FastGelu
TEST_F(GraphTransformationTests, GeluApproximation_Gelu_Add_Bias) {
  auto model_uri = MODEL_FOLDER "approximation/gelu_add_bias.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluApproximation>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["BiasGelu"], 0);
  EXPECT_EQ(op_to_count["FastGelu"], 1);
}

// Test MatMul & AddGeluFusion -> MatMul & FastGelu
TEST_F(GraphTransformationTests, GeluApproximation_Gelu_Add_MatMul) {
  auto model_uri = MODEL_FOLDER "approximation/gelu_add_matmul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<GeluApproximation>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["BiasGelu"], 0);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["FastGelu"], 1);
}

TEST_F(GraphTransformationTests, FastGeluFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Identity"] == 2);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluUseGraphInputFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluWithBiasFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu_with_bias.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<BiasGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluWithBiasUseGraphInputFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu_with_bias_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<BiasGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluFusionTest2) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu2.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluUseGraphInputFusionTest2) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu2_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluWithBiasFusionTest2) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu2_with_bias.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<BiasGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, FastGeluWithBiasUseGraphInputFusionTest2) {
  auto model_uri = MODEL_FOLDER "fusion/fast_gelu2_with_bias_use_graph_input.onnx";
  std::shared_ptr<Model> p_model;
  auto load_ret = Model::Load(model_uri, p_model, nullptr, *logger_);
  ASSERT_TRUE(load_ret.IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<FastGeluFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<BiasGeluFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["Tanh"] == 0);
  ASSERT_TRUE(op_to_count["Mul"] == 0);
  ASSERT_TRUE(op_to_count["FastGelu"] == 1);
}

TEST_F(GraphTransformationTests, LayerNormFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/layer_norm.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<LayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
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

TEST_F(GraphTransformationTests, LayerNormWithCastFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/layer_norm_with_cast.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<LayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

#ifdef ENABLE_TRAINING
  ASSERT_TRUE(op_to_count["Cast"] == 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 1);
#else
  ASSERT_TRUE(op_to_count["Cast"] == 1);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 0);
#endif
}

TEST_F(GraphTransformationTests, LayerNormWithSubDupFusionTest) {
  auto model_uri = MODEL_FOLDER "fusion/layer_norm_sub_dup.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<LayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
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

static void TestSkipLayerNormFusion(const std::basic_string<ORTCHAR_T>& file_path, int add_count, int ln_count,
                                    int skip_ln_count, logging::Logger* logger) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, *logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<LayerNormFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<SkipLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger);
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

TEST_F(GraphTransformationTests, SkipLayerNormFusionTest) {
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format1.onnx", 0, 0, 1, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format2.onnx", 0, 0, 1, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format3.onnx", 0, 0, 1, logger_.get());

  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format1_partial.onnx", 1, 0, 1, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format2_partial.onnx", 1, 0, 1, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format3_no_fusion.onnx", 1, 1, 0, logger_.get());

  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format1_graph_output.onnx", 1, 0, 1, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format2_graph_output.onnx", 1, 0, 1, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format3_graph_output.onnx", 1, 1, 0, logger_.get());
}

TEST_F(GraphTransformationTests, SkipLayerNormFusion_Input_Output_Check) {
  auto model_uri = MODEL_FOLDER "fusion/skip_layer_norm_input_output_check.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<LayerNormFusion>(), TransformerLevel::Level2);
  graph_transformation_mgr.Register(onnxruntime::make_unique<SkipLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  for (Node& node : graph.Nodes()) {
    if (node.OpType() == "SkipLayerNormalization") {
      // check inputs
      std::vector<NodeArg*>& input_defs = node.MutableInputDefs();
      EXPECT_EQ(input_defs.size(), 5u) << "SkipLayerNormalization number of inputs does not equal to 5. Got:" << node.InputDefs().size();
      EXPECT_EQ(input_defs[0]->Name(), "input.1");
      EXPECT_EQ(input_defs[1]->Name(), "6");
      EXPECT_EQ(input_defs[2]->Name(), "1");
      EXPECT_EQ(input_defs[3]->Name(), "2");
      EXPECT_EQ(input_defs[4]->Name(), "4");

      // check outputs
      std::vector<NodeArg*>& output_defs = node.MutableOutputDefs();
#ifdef ENABLE_TRAINING
      EXPECT_EQ(node.OutputDefs().size(), 3u) << "SkipLayerNormalization number of outputs does not equal to 3. Got:" << node.OutputDefs().size();
#else
      EXPECT_EQ(node.OutputDefs().size(), 1u) << "SkipLayerNormalization number of outputs does not equal to 1. Got:" << node.OutputDefs().size();
#endif
      EXPECT_EQ(output_defs[0]->Name(), "19");
    } else {
      EXPECT_EQ(node.OpType(), "MatMul") << "Unexpected node: " << node.OpType() << "," << node.Name();
    }
  }
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat1) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format1.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["ReduceSum"] == 1);
  ASSERT_TRUE(op_to_count["Attention"] == 1);
  ASSERT_TRUE(op_to_count["SkipLayerNormalization"] == 0);
  ASSERT_TRUE(op_to_count["EmbedLayerNormalization"] == 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat2) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format2.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
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
  ASSERT_TRUE(op_to_count["ReduceSum"] == 1);
  ASSERT_TRUE(op_to_count["Attention"] == 1);
  ASSERT_TRUE(op_to_count["SkipLayerNormalization"] == 0);
  ASSERT_TRUE(op_to_count["EmbedLayerNormalization"] == 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat3) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format3.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Expand"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["LayerNormalization"], 0);
  EXPECT_EQ(op_to_count["SkipLayerNormalization"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Cast"], 3);
  EXPECT_EQ(op_to_count["Attention"], 1);
  EXPECT_EQ(op_to_count["EmbedLayerNormalization"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat3NoCast) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format3_no_cast.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Expand"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["LayerNormalization"], 0);
  EXPECT_EQ(op_to_count["SkipLayerNormalization"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Cast"], 3);
  EXPECT_EQ(op_to_count["Attention"], 1);
  EXPECT_EQ(op_to_count["EmbedLayerNormalization"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat4) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format4.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
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
  ASSERT_TRUE(op_to_count["ReduceSum"] == 1);
  ASSERT_TRUE(op_to_count["Attention"] == 1);
  ASSERT_TRUE(op_to_count["SkipLayerNormalization"] == 0);
  ASSERT_TRUE(op_to_count["EmbedLayerNormalization"] == 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat5) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format5.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["LayerNormalization"], 0);
  EXPECT_EQ(op_to_count["SkipLayerNormalization"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
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

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat6) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format6.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Expand"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Equal"], 0);
  EXPECT_EQ(op_to_count["Where"], 0);
  EXPECT_EQ(op_to_count["LayerNormalization"], 0);
  EXPECT_EQ(op_to_count["SkipLayerNormalization"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Cast"], 3);
  EXPECT_EQ(op_to_count["Attention"], 1);
  EXPECT_EQ(op_to_count["EmbedLayerNormalization"], 1);
}

static void TestEmbedLayerNormFusionDistilBert(const std::basic_string<ORTCHAR_T>& model_uri,
                                               std::map<std::string, int>& op_to_count,
                                               logging::Logger* logger) {
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret1 = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger);
  ASSERT_TRUE(ret1.IsOK());

  op_to_count = CountOpsInGraph(graph);
}

//DistilBert
TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat7) {
  std::map<std::string, int> op_to_count;
  TestEmbedLayerNormFusionDistilBert(MODEL_FOLDER "fusion/embed_layer_norm_format7.onnx", op_to_count, logger_.get());
  EXPECT_EQ(op_to_count["EmbedLayerNormalization"], 1);
  EXPECT_EQ(op_to_count["Attention"], 1);
  EXPECT_EQ(op_to_count["Cast"], 2);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat8) {
  std::map<std::string, int> op_to_count;
  TestEmbedLayerNormFusionDistilBert(MODEL_FOLDER "fusion/embed_layer_norm_format8.onnx", op_to_count, logger_.get());
  EXPECT_EQ(op_to_count["EmbedLayerNormalization"], 1);
  EXPECT_EQ(op_to_count["Attention"], 1);
  EXPECT_EQ(op_to_count["Cast"], 2);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat9) {
  std::map<std::string, int> op_to_count;
  TestEmbedLayerNormFusionDistilBert(MODEL_FOLDER "fusion/embed_layer_norm_format9.onnx", op_to_count, logger_.get());
  EXPECT_EQ(op_to_count["EmbedLayerNormalization"], 1);
  EXPECT_EQ(op_to_count["Attention"], 1);
  EXPECT_EQ(op_to_count["Cast"], 2);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Gather"], 2);
  EXPECT_EQ(op_to_count["Unsqueeze"], 2);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionMultiple) {
  auto model_uri = MODEL_FOLDER "fusion/embed_layer_norm_multiple.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Expand"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["LayerNormalization"], 0);
  EXPECT_EQ(op_to_count["SkipLayerNormalization"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 2);
  EXPECT_EQ(op_to_count["MatMul"], 2);
  EXPECT_EQ(op_to_count["Add"], 5);
  EXPECT_EQ(op_to_count["Cast"], 6);
  EXPECT_EQ(op_to_count["Attention"], 2);
  EXPECT_EQ(op_to_count["EmbedLayerNormalization"], 2);
}

TEST_F(GraphTransformationTests, DynamicQuantizeMatMulTest) {
  auto model_uri = MODEL_FOLDER "fusion/dynamic_quantize_matmul.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 0);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["DynamicQuantizeMatMul"], 1);
}

TEST_F(GraphTransformationTests, DynamicQuantizeMatMulTest_With_Bias) {
  auto model_uri = MODEL_FOLDER "fusion/dynamic_quantize_matmul_bias.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 0);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["DynamicQuantizeMatMul"], 1);
}

TEST_F(GraphTransformationTests, DynamicQuantizeMatMulTest_With_ND_bias) {
  auto model_uri = MODEL_FOLDER "fusion/dynamic_quantize_matmul_bias_ND.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 0);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["DynamicQuantizeMatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 1);
}

TEST_F(GraphTransformationTests, DynamicQuantizeMatMulTest_With_Bias_No_B_ZP) {
  auto model_uri = MODEL_FOLDER "fusion/dynamic_quantize_matmul_bias_b_no_zp.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 0);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["DynamicQuantizeMatMul"], 1);
}

TEST_F(GraphTransformationTests, MatMulIntegerToFloatTest) {
  auto model_uri = MODEL_FOLDER "fusion/matmul_integer_to_float.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<DynamicQuantizeMatMulFusion>(), TransformerLevel::Level2);
  auto ret = graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_);
  ASSERT_TRUE(ret.IsOK());

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["DynamicQuantizeLinear"], 1);
  EXPECT_EQ(op_to_count["MatMulInteger"], 0);
  EXPECT_EQ(op_to_count["Cast"], 0);
  EXPECT_EQ(op_to_count["Mul"], 0);
  EXPECT_EQ(op_to_count["MatMulIntegerToFloat"], 3);
  EXPECT_EQ(op_to_count["Add"], 1);
}

#endif

// LayerNormalization implementation is in contrib namespace (OnnxDomain 1), so
// Without contib_ops enabled, we cannot parse the graph correctly.
#ifndef DISABLE_CONTRIB_OPS
// We used Opset 12 for testing to make sure we are not using GatherND OnnxDomain Opset 1.
static void GatherNDComputationReductionTest(const std::string op_type, logging::Logger& logger) {
  std::string op_type_lower = op_type;
  std::transform(op_type_lower.begin(), op_type_lower.end(), op_type_lower.begin(), [](unsigned char c) { return std::tolower(c); });
  std::string file_path = std::string("testdata/transform/computation_reduction/gathernd_") + op_type_lower + std::string(".onnx");
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(ToPathString(file_path), model, nullptr, logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ComputationReductionTransformer>(), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, logger));

  GraphViewer graph_viewer(graph);
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

  Node* gathernd_node = nullptr;
  for (auto node_index : node_topology_list) {
    Node* p_node = graph.GetNode(node_index);
    ASSERT_FALSE(p_node == nullptr);
    if (p_node->OpType().compare("GatherND") == 0) {
      gathernd_node = p_node;
      EXPECT_EQ(gathernd_node->MutableInputDefs()[0]->Name(), "input");
      const auto& consumers = graph.GetConsumerNodes(gathernd_node->MutableOutputDefs()[0]->Name());
      EXPECT_EQ(consumers[0]->OpType(), op_type);
    }
  }

  ASSERT_FALSE(gathernd_node == nullptr);
}

TEST_F(GraphTransformationTests, ComputationReductionTransformer_GatherND_Gelu) {
  GatherNDComputationReductionTest("Gelu", *logger_);
}

TEST_F(GraphTransformationTests, ComputationReductionTransformer_GatherND_Add) {
  GatherNDComputationReductionTest("Add", *logger_);
}

TEST_F(GraphTransformationTests, ComputationReductionTransformer_GatherND_LayerNormalization) {
  GatherNDComputationReductionTest("LayerNormalization", *logger_);
}

TEST_F(GraphTransformationTests, ComputationReductionTransformer_GatherND_MatMul) {
  GatherNDComputationReductionTest("MatMul", *logger_);
}

static void RunGatherNDE2EGraph(std::vector<OrtValue>& run_results, const PathString& model_uri,
                                const std::string session_log_id, const std::string& provider_type,
                                const std::vector<int64_t>& dims_input,
                                const std::vector<float>& input_values,
                                const std::vector<int64_t>& dims_unsqueezed_masked_lm_positions,
                                const std::vector<int64_t>& values_unsqueezed_masked_lm_positions) {
  SessionOptions so;
  // we don't want any transformation here.
  so.graph_optimization_level = TransformerLevel::Default;
  so.session_logid = session_log_id;

  InferenceSession session_object{so, GetEnvironment()};
  std::unique_ptr<IExecutionProvider> execution_provider;
  if (provider_type == onnxruntime::kCpuExecutionProvider)
    execution_provider = DefaultCpuExecutionProvider();
  else if (provider_type == onnxruntime::kCudaExecutionProvider)
    execution_provider = DefaultCudaExecutionProvider();
  EXPECT_TRUE(session_object.RegisterExecutionProvider(std::move(execution_provider)).IsOK());

  Status st;
  ASSERT_TRUE((st = session_object.Load(model_uri)).IsOK()) << st;
  ASSERT_TRUE((st = session_object.Initialize()).IsOK()) << st;

  OrtValue input1;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_input, input_values, &input1);
  OrtValue input2;
  CreateMLValue<int64_t>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_unsqueezed_masked_lm_positions,
                         values_unsqueezed_masked_lm_positions, &input2);

  NameMLValMap feeds;
  feeds.insert(std::make_pair("input", input1));
  feeds.insert(std::make_pair("unsqueezed_masked_lm_positions", input2));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("output");
  output_names.push_back("gather_output");

  // Now run
  RunOptions run_options;
  st = session_object.Run(run_options, feeds, output_names, &run_results);

  EXPECT_TRUE(st.IsOK());
}

TEST_F(GraphTransformationTests, ComputationReductionTransformer_GatherND_E2E) {
  auto model_uri = MODEL_FOLDER "computation_reduction/e2e.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger_));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  graph_transformation_mgr.Register(onnxruntime::make_unique<ComputationReductionTransformer>(), TransformerLevel::Level1);
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));

  // check the expected node orders.
  {
    GraphViewer graph_viewer(graph);
    const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();

    Node* gathernd_node = nullptr;
    for (auto node_index : node_topology_list) {
      Node* p_node = graph.GetNode(node_index);
      ASSERT_FALSE(p_node == nullptr);
      if (p_node->OpType().compare("GatherND") == 0) {
        gathernd_node = p_node;
        const Node* layer_norm_node = graph.GetProducerNode(gathernd_node->MutableInputDefs()[0]->Name());
        EXPECT_EQ(layer_norm_node->OpType(), "LayerNormalization");
        EXPECT_EQ(layer_norm_node->Name(), "layer_norm_1");
        const auto& consumers = graph.GetConsumerNodes(gathernd_node->MutableOutputDefs()[0]->Name());
        EXPECT_EQ(consumers[0]->OpType(), "MatMul");
        EXPECT_EQ(consumers[0]->Name(), "matmul_1");
        break;
      }
    }

    ASSERT_FALSE(gathernd_node == nullptr);
  }

  // check result diff after the re-order
  auto new_model_uri = "computation_reduction_transformer_after.onnx";
  Model::Save(*model, new_model_uri);

  float scale = 1.f;
  float mean = 0.f;
  float seed = 123.f;
  std::default_random_engine generator_float{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution_float{mean, scale};

  int batch_size = 8;
  int sequence = 128;
  int hidden_size = 128;
  int dynamic_predict_count = 20;
  const std::vector<int64_t> dims_input = {batch_size, sequence, hidden_size};
  std::vector<float> input_values(TensorShape(dims_input).Size());
  std::for_each(input_values.begin(), input_values.end(),
                [&generator_float, &distribution_float](float& value) { value = distribution_float(generator_float); });

  const std::vector<int64_t> dims_unsqueezed_masked_lm_positions = {batch_size, dynamic_predict_count, 1};
  std::vector<int64_t> values_unsqueezed_masked_lm_positions(TensorShape(dims_unsqueezed_masked_lm_positions).Size());

  std::random_device rd;                                   // obtain a random number from hardware
  std::mt19937 eng(rd());                                  // seed the generator
  std::uniform_int_distribution<> distr(0, sequence - 1);  // define the range
  std::for_each(values_unsqueezed_masked_lm_positions.begin(), values_unsqueezed_masked_lm_positions.end(),
                [&distr, &eng](int64_t& value) { value = distr(eng); });

  static const std::string all_provider_types[] = {
      onnxruntime::kCpuExecutionProvider,
#ifdef USE_CUDA
      onnxruntime::kCudaExecutionProvider,
#endif
  };

  for (auto& provider_type : all_provider_types) {
    std::vector<OrtValue> expected_ort_values;
    RunGatherNDE2EGraph(expected_ort_values, model_uri, std::string("RawGraphRun"), provider_type,
                        dims_input, input_values, dims_unsqueezed_masked_lm_positions,
                        values_unsqueezed_masked_lm_positions);

    std::vector<OrtValue> actual_ort_values;
    RunGatherNDE2EGraph(actual_ort_values, ToPathString(new_model_uri), std::string("OptimizedGraphRun"), provider_type,
                        dims_input, input_values, dims_unsqueezed_masked_lm_positions,
                        values_unsqueezed_masked_lm_positions);

    ASSERT_TRUE(expected_ort_values.size() == actual_ort_values.size());
    const double per_sample_tolerance = 1e-4;
    const double relative_per_sample_tolerance = 1e-4;
    for (size_t i = 0; i < expected_ort_values.size(); i++) {
      auto ret = CompareOrtValue(actual_ort_values[i], expected_ort_values[i],
                                 per_sample_tolerance, relative_per_sample_tolerance, false);
      EXPECT_EQ(ret.first, COMPARE_RESULT::SUCCESS) << ret.second;
    }
  }
}
#endif

#ifndef DISABLE_CONTRIB_OPS
template <typename GraphTransformationCheckFn>
static void TestMatMulScaleFusion(
    const PathString& model_path, const Logger& logger,
    GraphTransformationCheckFn graph_transformation_check,
    const std::unordered_set<std::string>& excluded_initializer_names = {}) {
  SCOPED_TRACE(ORT_TSTR("model path: ") + model_path);

  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_path, model, nullptr, logger));
  Graph& graph = model->MainGraph();

  auto original_op_counts = CountOpsInGraph(graph);

  onnxruntime::GraphTransformerManager graph_transformer_manager{5};
  ASSERT_STATUS_OK(graph_transformer_manager.Register(
      make_unique<MatMulScaleFusion>(std::unordered_set<std::string>{}, excluded_initializer_names),
      TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformer_manager.ApplyTransformers(graph, TransformerLevel::Level2, logger));

  auto transformed_op_counts = CountOpsInGraph(graph);

  graph_transformation_check(graph, original_op_counts, transformed_op_counts);
}

TEST_F(GraphTransformationTests, MatMulScaleFusionFusableModels) {
  const std::vector<PathString> one_fusion_model_paths{
      MODEL_FOLDER "fusion/matmul_scale_in0.onnx",
      MODEL_FOLDER "fusion/matmul_scale_in0_in1.onnx",
      MODEL_FOLDER "fusion/matmul_scale_in0_in1_out.onnx",
      MODEL_FOLDER "fusion/matmul_scale_transposescalematmul_in0_in1_out.onnx",
  };

  for (const auto& path : one_fusion_model_paths) {
    TestMatMulScaleFusion(
        path, *logger_,
        [](const Graph& graph,
           std::map<std::string, int> original_op_counts,
           std::map<std::string, int> transformed_op_counts) {
          EXPECT_EQ(transformed_op_counts["Mul"], 0);
          EXPECT_EQ(transformed_op_counts["Div"], 0);
          EXPECT_EQ(transformed_op_counts["MatMul"], 0);
          EXPECT_EQ(transformed_op_counts["TransposeScaleMatMul"], 1);

          // check combined scale, individual scales should all have the same value
          const float scale_value = 3.0f;

          const int num_scales =
              original_op_counts["Mul"] + original_op_counts["Div"] + original_op_counts["TransposeScaleMatMul"];

          auto fused_node = std::find_if(
              graph.Nodes().cbegin(), graph.Nodes().cend(),
              [](const Node& node) { return node.OpType() == "TransposeScaleMatMul"; });
          ASSERT_NE(fused_node, graph.Nodes().cend());

          auto alpha_attr = fused_node->GetAttributes().find("alpha");
          ASSERT_NE(alpha_attr, fused_node->GetAttributes().end());

          EXPECT_EQ(alpha_attr->second.f(), pow(scale_value, num_scales));
        });
  }
}

TEST_F(GraphTransformationTests, MatMulScaleFusionUnfusableModels) {
  const std::vector<PathString> unfusable_model_paths{
      MODEL_FOLDER "fusion/matmul_scale_unfusable_div_not_scale.onnx",
      MODEL_FOLDER "fusion/matmul_scale_unfusable_scale_not_scalar.onnx",
      MODEL_FOLDER "fusion/matmul_scale_unfusable_scale_not_constant.onnx",
  };

  for (const auto& path : unfusable_model_paths) {
    TestMatMulScaleFusion(
        path, *logger_,
        [](const Graph&,
           const std::map<std::string, int>& original_op_counts,
           const std::map<std::string, int>& transformed_op_counts) {
          EXPECT_EQ(original_op_counts, transformed_op_counts);
        });
  }
}

TEST_F(GraphTransformationTests, MatMulScaleFusionReusedInputScale) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_reused_input_scale.onnx", *logger_,
      [](const Graph&,
         const std::map<std::string, int>&,
         std::map<std::string, int> transformed_op_counts) {
        EXPECT_EQ(transformed_op_counts["Mul"], 0);
        EXPECT_EQ(transformed_op_counts["Div"], 0);
        EXPECT_EQ(transformed_op_counts["MatMul"], 0);
        EXPECT_EQ(transformed_op_counts["TransposeScaleMatMul"], 2);
      });
}

TEST_F(GraphTransformationTests, MatMulScaleFusionExcludedInitializerName) {
  TestMatMulScaleFusion(
      MODEL_FOLDER "fusion/matmul_scale_in0.onnx", *logger_,
      [](const Graph&,
         const std::map<std::string, int>& original_op_counts,
         const std::map<std::string, int>& transformed_op_counts) {
        EXPECT_EQ(original_op_counts, transformed_op_counts);
      },
      {"scale"});
}
#endif

}  // namespace test
}  // namespace onnxruntime
