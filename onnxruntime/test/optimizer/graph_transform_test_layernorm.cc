// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#include <algorithm>

#include "gtest/gtest.h"

#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/initializer.h"

#include "core/optimizer/bias_skip_layer_norm_fusion.h"
#include "core/optimizer/embed_layer_norm_fusion.h"
#include "core/optimizer/group_query_attention_fusion.h"
#include "core/optimizer/layer_norm_fusion.h"
#include "core/optimizer/skip_layer_norm_fusion.h"

#include "test/capturing_sink.h"
#include "test/unittest_util/framework_test_utils.h"
#include "test/optimizer/graph_transform_test_fixture.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")

#ifndef DISABLE_CONTRIB_OPS

TEST_F(GraphTransformationTests, LayerNormFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/layer_norm.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<LayerNormFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

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
      EXPECT_EQ(node.InputDefs().size(), 3u)
          << "LayerNormalization number of inputs does not equal to 3. Got:" << node.InputDefs().size();
      // LayerNormalization input "scale" and "bias" should have the same dimension.
      const TensorShapeProto* scale_shape = node.InputDefs()[1]->Shape();
      const TensorShapeProto* bias_shape = node.InputDefs()[2]->Shape();
      EXPECT_EQ(scale_shape->dim_size(), 1)
          << "LayerNormalization scale should be 1D. Got: " << scale_shape->dim_size();
      EXPECT_EQ(bias_shape->dim_size(), 1)
          << "LayerNormalization bias should be 1D. Got: " << bias_shape->dim_size();
      EXPECT_EQ(scale_shape->dim(0).dim_value(), bias_shape->dim(0).dim_value());
    } else {
      EXPECT_TRUE(false) << "Unexpected node " << node.Name();
    }
  }
}

TEST_F(GraphTransformationTests, TwoLayerNormShareSameInput) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/layer_norm_shared_input.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<LayerNormFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count.size() == 1);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 2);
}

TEST_F(GraphTransformationTests, LayerNormWithCastFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/layer_norm_with_cast.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<LayerNormFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

#ifdef ENABLE_TRAINING_CORE
  ASSERT_TRUE(op_to_count["Cast"] == 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 1);
#else
  ASSERT_TRUE(op_to_count["Cast"] == 1);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 0);
#endif
}

TEST_F(GraphTransformationTests, LayerNormWithCastFusionTest_2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/layer_norm_with_cast_2.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<LayerNormFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  ASSERT_TRUE(op_to_count["Cast"] == 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 1);
}

TEST_F(GraphTransformationTests, LayerNormWithCastFusionTest_3) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/layer_norm_with_cast_3.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<LayerNormFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  ASSERT_TRUE(op_to_count["Cast"] == 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 1);
}

TEST_F(GraphTransformationTests, LayerNormWithCastFusionTest_4) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/layer_norm_with_cast_4.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<LayerNormFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);

  ASSERT_TRUE(op_to_count["Cast"] == 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 1);
}

/*
ReduceMean:
  axes - INTS : A list of integers, along which to reduce.
  The default is to reduce over all the dimensions of the input tensor.
  Accepted range is [-r, r-1] where r = rank(data).
*/
TEST_F(GraphTransformationTests, LayerNormWithSubDupFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/layer_norm_sub_dup.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<LayerNormFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] > 0);
  ASSERT_TRUE(op_to_count["Add"] > 0);
  ASSERT_TRUE(op_to_count["Sub"] > 0);
  ASSERT_TRUE(op_to_count["ReduceMean"] > 0);
  ASSERT_TRUE(op_to_count["Pow"] > 0);
  ASSERT_TRUE(op_to_count["Sqrt"] > 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 0);
  /*
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
  */
}

void BuildLayerNorm(ModelTestBuilder& builder, std::vector<int64_t> reduce1_axes = {-1},
                    std::vector<int64_t> reduce2_axes = {-1}) {
  std::vector<int64_t> input_shape = {2, 3, 3, 3};
  auto* data_arg = builder.MakeInput<MLFloat16>(input_shape);
  auto* pow_initializer = builder.MakeInitializer<float>({}, {2.0f});
  auto* add_initializer = builder.MakeInitializer<float>({}, {1e-5f});
  std::vector<int64_t> normalized_shape = {};
  int64_t normalized_shape_size = 1;
  auto raxes = reduce1_axes;
  std::transform(raxes.begin(), raxes.end(), raxes.begin(), [&input_shape](int64_t i) {
    return i < 0 ? i + input_shape.size() : i;
  });
  sort(raxes.begin(), raxes.end());
  for (auto axis : raxes) {
    normalized_shape.push_back(input_shape[axis]);
    normalized_shape_size *= input_shape[axis];
  }

  auto* weight_initializer = builder.MakeInitializer<MLFloat16>(
      normalized_shape, std::vector<MLFloat16>(normalized_shape_size, MLFloat16(1.0f)));
  auto* bias_initializer = builder.MakeInitializer<MLFloat16>(
      normalized_shape, std::vector<MLFloat16>(normalized_shape_size, MLFloat16(0.0f)));
  auto* reduce_mean_out_1 = builder.MakeIntermediate();
  auto* sub_out = builder.MakeIntermediate();
  auto* cast_out_1 = builder.MakeIntermediate();
  auto* pow_out = builder.MakeIntermediate();
  auto* reduce_mean_out_2 = builder.MakeIntermediate();
  auto* add_out_1 = builder.MakeIntermediate();
  auto* sqrt_out = builder.MakeIntermediate();
  auto* div_out = builder.MakeIntermediate();
  auto* cast_out_2 = builder.MakeIntermediate();
  auto* mul_out = builder.MakeIntermediate();
  auto* add_out_2 = builder.MakeOutput();
  auto opset = builder.DomainToVersionMap().find(kOnnxDomain)->second;

  if (opset >= 18) {
    int64_t rsize = static_cast<int64_t>(reduce1_axes.size());
    onnxruntime::NodeArg* axes = builder.MakeInitializer<int64_t>({rsize}, reduce1_axes);
    builder.AddNode("ReduceMean", {data_arg, axes}, {reduce_mean_out_1});
  } else {
    builder.AddNode("ReduceMean", {data_arg}, {reduce_mean_out_1}).AddAttribute("axes", reduce1_axes);
  }
  builder.AddNode("Sub", {data_arg, reduce_mean_out_1}, {sub_out});
  builder.AddNode("Cast", {sub_out}, {cast_out_1})
      .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
  builder.AddNode("Pow", {cast_out_1, pow_initializer}, {pow_out});
  if (opset >= 18) {
    int64_t rsize = static_cast<int64_t>(reduce2_axes.size());
    onnxruntime::NodeArg* axes = builder.MakeInitializer<int64_t>({rsize}, reduce2_axes);
    builder.AddNode("ReduceMean", {pow_out, axes}, {reduce_mean_out_2});
  } else {
    builder.AddNode("ReduceMean", {pow_out}, {reduce_mean_out_2}).AddAttribute("axes", reduce2_axes);
  }
  builder.AddNode("Add", {reduce_mean_out_2, add_initializer}, {add_out_1});
  builder.AddNode("Sqrt", {add_out_1}, {sqrt_out});
  builder.AddNode("Div", {cast_out_1, sqrt_out}, {div_out});
  builder.AddNode("Cast", {div_out}, {cast_out_2})
      .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16));
  builder.AddNode("Mul", {cast_out_2, weight_initializer}, {mul_out});
  builder.AddNode("Add", {mul_out, bias_initializer}, {add_out_2});
}

TEST_F(GraphTransformationTests, LayerNormWithCastFusionTest_5) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildLayerNorm(builder, {-1}, {-1});
  };

  auto pre_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["ReduceMean"] == 2);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sub"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Pow"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] == 2);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sqrt"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Div"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["ReduceMean"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sub"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Pow"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sqrt"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Div"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 0);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["LayerNormalization"] == 1);
    return Status::OK();
  };

  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  std::unique_ptr<GraphTransformer> transformer_1 = std::make_unique<LayerNormFusion>();
  std::unique_ptr<GraphTransformer> transformer_2 =
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2);
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(transformer_1),
                                        TransformerLevel::Level1, 1, pre_graph_checker, post_graph_checker));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer_2),
                                        TransformerLevel::Level2, 1, pre_graph_checker, post_graph_checker));
}

TEST_F(GraphTransformationTests, LayerNormWithCastFusionTest_6) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildLayerNorm(builder, {-2}, {-1});
  };

  int num_of_layer_norm = 0;
  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["ReduceMean"] == 2 - 2 * num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sub"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2 - 2 * num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Pow"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] == 2 - 2 * num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sqrt"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Div"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["LayerNormalization"] == num_of_layer_norm);
    return Status::OK();
  };

  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  std::unique_ptr<GraphTransformer> transformer_1 = std::make_unique<LayerNormFusion>();
  std::unique_ptr<GraphTransformer> transformer_2 =
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2);
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(transformer_1),
                                        TransformerLevel::Level1, 1, nullptr, post_graph_checker));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer_2),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

TEST_F(GraphTransformationTests, LayerNormWithCastFusionTest_7) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildLayerNorm(builder, {-2, -1}, {-1, -2});
  };
#ifdef ENABLE_TRAINING_CORE
  int num_of_layer_norm = 1;
#else
  int num_of_layer_norm = 0;
#endif
  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["ReduceMean"] == 2 - 2 * num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sub"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2 - 2 * num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Pow"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] == 2 - 2 * num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sqrt"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Div"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["LayerNormalization"] == num_of_layer_norm);
    return Status::OK();
  };

  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  std::unique_ptr<GraphTransformer> transformer_1 = std::make_unique<LayerNormFusion>();
  std::unique_ptr<GraphTransformer> transformer_2 =
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2);
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(transformer_1),
                                        TransformerLevel::Level1, 1, nullptr, post_graph_checker));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer_2),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

TEST_F(GraphTransformationTests, LayerNormWithCastFusionTest_8) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildLayerNorm(builder, {-3, -2, -1}, {-1, -2});
  };

  int num_of_layer_norm = 0;
  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["ReduceMean"] == 2 - 2 * num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sub"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2 - 2 * num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Pow"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] == 2 - 2 * num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sqrt"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Div"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["LayerNormalization"] == num_of_layer_norm);
    return Status::OK();
  };

  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  std::unique_ptr<GraphTransformer> transformer_1 = std::make_unique<LayerNormFusion>();
  std::unique_ptr<GraphTransformer> transformer_2 =
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2);
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(transformer_1),
                                        TransformerLevel::Level1, 1, nullptr, post_graph_checker));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer_2),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

TEST_F(GraphTransformationTests, LayerNormWithCastFusionTest_9) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    BuildLayerNorm(builder, {2, -1}, {-1, -2});
  };

#ifdef ENABLE_TRAINING_CORE
  int num_of_layer_norm = 1;
#else
  int num_of_layer_norm = 0;
#endif
  auto post_graph_checker = [&](Graph& graph) {
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["ReduceMean"] == 2 - 2 * num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sub"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Cast"] == 2 - 2 * num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Pow"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] == 2 - 2 * num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Sqrt"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Div"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Mul"] == 1 - num_of_layer_norm);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["LayerNormalization"] == num_of_layer_norm);
    return Status::OK();
  };

  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  std::unique_ptr<GraphTransformer> transformer_1 = std::make_unique<LayerNormFusion>();
  std::unique_ptr<GraphTransformer> transformer_2 =
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2);
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 18, *logger_, std::move(transformer_1),
                                        TransformerLevel::Level1, 1, nullptr, post_graph_checker));
  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 14, *logger_, std::move(transformer_2),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

TEST_F(GraphTransformationTests, SimplifiedLayerNormFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/layer_norm_t5.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<SimplifiedLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["ReduceMean"] == 0);
  ASSERT_TRUE(op_to_count["Pow"] == 0);
  ASSERT_TRUE(op_to_count["Sqrt"] == 0);
  ASSERT_TRUE(op_to_count["SimplifiedLayerNormalization"] == 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "SimplifiedLayerNormalization") {
      // LayerNormalization should have two inputs.
      EXPECT_EQ(node.InputDefs().size(), 2u) << "LayerNormalization number of inputs does not equal to 2. Got:" << node.InputDefs().size();
      // LayerNormalization input "scale" and "bias" should have the same dimension.
      const TensorShapeProto* scale_shape = node.InputDefs()[1]->Shape();
      EXPECT_EQ(scale_shape->dim_size(), 1) << "LayerNormalization scale should be 1D. Got: " << scale_shape->dim_size();
    } else {
      EXPECT_TRUE(false) << "Unexpected node " << node.Name();
    }
  }
}

// It tests the scenario when scale or bias are not Graph Inputs and not initialized in Graph
// To test this added a Identity node after Scale and Bias terms to ensure LayerNormFusion works properly
TEST_F(GraphTransformationTests, LayerNormScaleBiasTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/layer_norm_fusion_scale_bias.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<LayerNormFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger_));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["ReduceMean"], 0);
  ASSERT_EQ(op_to_count["Sub"], 0);
  ASSERT_EQ(op_to_count["Cast"], 0);
  ASSERT_EQ(op_to_count["Pow"], 0);
  ASSERT_EQ(op_to_count["Add"], 0);
  ASSERT_EQ(op_to_count["Sqrt"], 0);
  ASSERT_EQ(op_to_count["Div"], 0);
  ASSERT_EQ(op_to_count["Mul"], 0);
  ASSERT_EQ(op_to_count["LayerNormalization"], 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "LayerNormalization") {
      // LayerNormalization should have three inputs.
      EXPECT_EQ(node.InputDefs().size(), 3u) << "LayerNormalization number of inputs does not equal to 3. Got:" << node.InputDefs().size();
      // LayerNormalization input "scale" and "bias" should have the same dimension.
      const TensorShapeProto* scale_shape = node.InputDefs()[1]->Shape();
      EXPECT_EQ(scale_shape->dim_size(), 1) << "LayerNormalization scale should be 1D. Got: " << scale_shape->dim_size();
    }
  }
}

// If EP is non-GPU EP or unknown, the sub-graph will be not fused because CPU impl for SimplifiedLayerNormalization
// doesn't support input and scale having different data types.
TEST_F(GraphTransformationTests, SimplifiedLayerNormWithCastsFusionTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/simplified_layer_norm_with_casts.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  InlinedHashSet<std::string_view> compatible_eps;
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<SimplifiedLayerNormFusion>(compatible_eps),
                                                     TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["SimplifiedLayerNormalization"] == 0);
}

TEST_F(GraphTransformationTests, SimplifiedLayerNormWithCastsFusionTestCudaEp) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/simplified_layer_norm_with_casts.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();
  for (auto& node : graph.Nodes()) {
    node.SetExecutionProviderType(kCudaExecutionProvider);
  }

  InlinedHashSet<std::string_view> compatible_eps;
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<SimplifiedLayerNormFusion>(compatible_eps),
                                                     TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["ReduceMean"] == 0);
  ASSERT_TRUE(op_to_count["Pow"] == 0);
  ASSERT_TRUE(op_to_count["Sqrt"] == 0);
  ASSERT_TRUE(op_to_count["Cast"] == 0);
  ASSERT_TRUE(op_to_count["SimplifiedLayerNormalization"] == 1);

  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "SimplifiedLayerNormalization") {
      // LayerNormalization should have two inputs.
      EXPECT_EQ(node.InputDefs().size(), 2u)
          << "LayerNormalization number of inputs does not equal to 2. Got:" << node.InputDefs().size();
      // LayerNormalization input "scale" and "bias" should have the same dimension.
      const TensorShapeProto* scale_shape = node.InputDefs()[1]->Shape();
      EXPECT_EQ(scale_shape->dim_size(), 1)
          << "LayerNormalization scale should be 1D. Got: " << scale_shape->dim_size();
    } else if (node.OpType() == "Cast") {
      continue;
    } else {
      EXPECT_TRUE(false) << "Unexpected node " << node.Name();
    }
  }
}

static void TestGQAFusion(const std::basic_string<ORTCHAR_T>& file_path, int matmulnbits_count, int matmul_count, logging::Logger* logger) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, *logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{3};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<GroupQueryAttentionFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["com.microsoft.RotaryEmbedding"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.MatMulNBits"] == matmulnbits_count);
  ASSERT_TRUE(op_to_count["MatMul"] == matmul_count);
  ASSERT_TRUE(op_to_count["com.microsoft.GroupQueryAttention"] == 1);
}

static void TestSkipLayerNormFusion(const std::basic_string<ORTCHAR_T>& file_path, int add_count, int ln_count,
                                    int skip_ln_count, int cast_count, logging::Logger* logger) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, *logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<LayerNormFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<SkipLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Div"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == add_count);
  ASSERT_TRUE(op_to_count["Sub"] == 0);
  ASSERT_TRUE(op_to_count["ReduceMean"] == 0);
  ASSERT_TRUE(op_to_count["Pow"] == 0);
  ASSERT_TRUE(op_to_count["Sqrt"] == 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == ln_count);
  ASSERT_TRUE(op_to_count["com.microsoft.SkipLayerNormalization"] == skip_ln_count);
  ASSERT_TRUE(op_to_count["Cast"] == cast_count);
}

TEST_F(GraphTransformationTests, SkipLayerNormFusionTest) {
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format1.onnx", 0, 0, 1, 0, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format2.onnx", 0, 0, 1, 0, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format3.onnx", 0, 0, 1, 0, logger_.get());

  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format1_partial.onnx", 1, 0, 1, 0, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format2_partial.onnx", 1, 0, 1, 0, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format3_no_fusion.onnx", 1, 1, 0, 0, logger_.get());

  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format1_graph_output.onnx", 1, 0, 1, 0, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format2_graph_output.onnx", 1, 0, 1, 0, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format3_graph_output.onnx", 1, 1, 0, 0, logger_.get());
}

// SkipLayerNorm fusion should not be applied when gamma/beta have more than 1 dimension,
// because the SkipLayerNormalization kernel requires 1D gamma/beta.
TEST_F(GraphTransformationTests, SkipLayerNormFusion_3DGamma_NoFusion) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    // Inputs: A and B are 3D [16, 32, 4]
    auto* input_a = builder.MakeInput<float>({16, 32, 4}, -1.0f, 1.0f);
    auto* input_b = builder.MakeInput<float>({16, 32, 4}, -1.0f, 1.0f);
    // gamma and beta have 3D shape [1, 1, 4] (not 1D)
    auto* gamma = builder.MakeInitializer<float>({1, 1, 4}, {1.0f, 2.0f, 3.0f, 4.0f});
    auto* beta = builder.MakeInitializer<float>({1, 1, 4}, {0.1f, 0.2f, 0.3f, 0.4f});
    auto* add_out = builder.MakeIntermediate();
    auto* ln_out = builder.MakeOutput();

    builder.AddNode("Add", {input_a, input_b}, {add_out});
    builder.AddNode("LayerNormalization", {add_out, gamma, beta}, {ln_out})
        .AddAttribute("axis", static_cast<int64_t>(-1));
  };

  auto post_graph_checker = [](Graph& graph) {
    // SkipLayerNormalization should NOT have been created because gamma/beta are 3D.
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["Add"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["LayerNormalization"] == 1);
    TEST_RETURN_IF_NOT(CountOpsInGraph(graph)["com.microsoft.SkipLayerNormalization"] == 0);
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 17, *logger_,
                                        std::make_unique<SkipLayerNormFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

TEST_F(GraphTransformationTests, GroupQueryAttentionFusionTest) {
  TestGQAFusion(MODEL_FOLDER "fusion/gqa_fusion_quantized_simple.onnx", 1, 0, logger_.get());
  TestGQAFusion(MODEL_FOLDER "fusion/gqa_fusion_different_head_sizes.onnx", 0, 1, logger_.get());
  TestGQAFusion(MODEL_FOLDER "fusion/gqa_fusion_quantized_different_head_sizes.onnx", 1, 0, logger_.get());
}

TEST_F(GraphTransformationTests, SkipLayerNormFusionWithCastTest) {
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format1_with_cast.onnx", 0, 0, 1, 3, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format2_with_cast.onnx", 0, 0, 1, 3, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format3_with_cast.onnx", 0, 0, 1, 2, logger_.get());

  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format1_partial_with_cast.onnx", 1, 0, 1, 2, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format2_partial_with_cast.onnx", 1, 0, 1, 2, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format3_no_fusion_with_cast.onnx", 1, 1, 0, 0, logger_.get());

  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format1_graph_output_with_cast.onnx", 1, 0, 1, 2, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format2_graph_output_with_cast.onnx", 1, 0, 1, 2, logger_.get());
  TestSkipLayerNormFusion(MODEL_FOLDER "fusion/skip_layer_norm_format3_graph_output_with_cast.onnx", 1, 1, 0, 0, logger_.get());
}

static void TestSkipLayerNormFusionInputOutputCheck(const std::basic_string<ORTCHAR_T>& model_uri, bool with_cast, logging::Logger* logger) {
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  const InlinedHashSet<std::string_view> no_limit_empty_ep_list = {};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<LayerNormFusion>(), TransformerLevel::Level1));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<LayerNormFusion>(no_limit_empty_ep_list, TransformerLevel::Level2), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<SkipLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level1, *logger));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

  for (Node& node : graph.Nodes()) {
    if (node.OpType() == "SkipLayerNormalization") {
      // check inputs
      std::vector<NodeArg*>& input_defs = node.MutableInputDefs();
      EXPECT_EQ(input_defs.size(), 5u) << "SkipLayerNormalization number of inputs does not equal to 5. Got:" << node.InputDefs().size();
      EXPECT_EQ(input_defs[0]->Name(), ((with_cast) ? "input.1_Float" : "input.1"));
      EXPECT_EQ(input_defs[1]->Name(), ((with_cast) ? "6_Float" : "6"));
      EXPECT_EQ(input_defs[2]->Name(), "1");
      EXPECT_EQ(input_defs[3]->Name(), "2");
      EXPECT_EQ(input_defs[4]->Name(), ((with_cast) ? "4_Float" : "4"));

      // check outputs
      std::vector<NodeArg*>& output_defs = node.MutableOutputDefs();
#ifdef ENABLE_TRAINING_CORE
      EXPECT_EQ(node.OutputDefs().size(), 3u) << "SkipLayerNormalization number of outputs does not equal to 3. Got:" << node.OutputDefs().size();
#else
      EXPECT_EQ(node.OutputDefs().size(), 1u) << "SkipLayerNormalization number of outputs does not equal to 1. Got:" << node.OutputDefs().size();
#endif
      EXPECT_EQ(output_defs[0]->Name(), "19");
    } else if (node.OpType() == "Cast") {
      EXPECT_TRUE(with_cast) << "Unexpected node: " << node.OpType() << "," << node.Name();
    } else {
      EXPECT_EQ(node.OpType(), "MatMul") << "Unexpected node: " << node.OpType() << "," << node.Name();
    }
  }
}

TEST_F(GraphTransformationTests, SkipLayerNormFusion_Input_Output_Check) {
  TestSkipLayerNormFusionInputOutputCheck(MODEL_FOLDER "fusion/skip_layer_norm_input_output_check.onnx", false, logger_.get());
  TestSkipLayerNormFusionInputOutputCheck(MODEL_FOLDER "fusion/skip_layer_norm_input_output_with_cast_check.onnx", true, logger_.get());
}

static void TestSkipLayerNormFusionNoBeta(const std::basic_string<ORTCHAR_T>& model_uri, bool with_cast, logging::Logger* logger) {
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<SkipLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["LayerNormalization"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.SkipLayerNormalization"] == 1);
  ASSERT_TRUE(op_to_count["Cast"] == ((with_cast) ? 2 : 0));
}

TEST_F(GraphTransformationTests, SkipLayerNormFusion_NoBeta) {
  TestSkipLayerNormFusionNoBeta(MODEL_FOLDER "fusion/skip_layer_norm_no_beta.onnx", false, logger_.get());
  TestSkipLayerNormFusionNoBeta(MODEL_FOLDER "fusion/skip_layer_norm_no_beta_with_cast.onnx", true, logger_.get());
}

// ---- BiasSkipLayerNormFusion tests ----
//
// All tests start with a pre-existing 4-input com.microsoft.SkipLayerNormalization node,
// mirroring the scenario where a model was already exported with SkipLayerNormalization (e.g.,
// via the Python transformer optimizer), and a bias Add upstream still needs to be absorbed.

// Verify that Add(MatMul_out, bias_1D) → SLN(4 inputs) is fused into SLN(5 inputs).
// Pattern: Add at SLN input[0], bias as Add input[1].
TEST_F(GraphTransformationTests, BiasSkipLayerNormFusion_AddAtInput0) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* matmul_a = builder.MakeInput<float>({2, 4, 8}, -1.0f, 1.0f);
    auto* matmul_b = builder.MakeInitializer<float>({8, 4}, -1.0f, 1.0f);
    auto* skip = builder.MakeInput<float>({2, 4, 4}, -1.0f, 1.0f);
    auto* gamma = builder.MakeInitializer<float>({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto* beta = builder.MakeInitializer<float>({4}, {0.0f, 0.0f, 0.0f, 0.0f});
    auto* bias = builder.MakeInitializer<float>({4}, {0.1f, 0.2f, 0.3f, 0.4f});

    auto* matmul_out = builder.MakeIntermediate();
    auto* add_out = builder.MakeIntermediate();
    auto* sln_out = builder.MakeOutput();

    builder.AddNode("MatMul", {matmul_a, matmul_b}, {matmul_out});
    builder.AddNode("Add", {matmul_out, bias}, {add_out});
    // 4-input SLN: add_out at input[0], skip at input[1]
    builder.AddNode("SkipLayerNormalization", {add_out, skip, gamma, beta}, {sln_out}, kMSDomain);
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count["Add"] == 0);
    TEST_RETURN_IF_NOT(op_count["com.microsoft.SkipLayerNormalization"] == 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "SkipLayerNormalization") {
        // Bias absorbed as 5th input
        TEST_RETURN_IF_NOT(node.InputDefs().size() == 5u);

        // Verify wiring: input[0] is produced by MatMul, input[1] is the original skip input,
        // and input[4] is an initializer (the fused bias).
        const auto& input_defs = node.InputDefs();
        auto* input0 = input_defs[0];
        auto* input1 = input_defs[1];
        auto* input4 = input_defs[4];

        // input[0] should come from MatMul
        const Node* input0_producer = graph.GetProducerNode(input0->Name());
        TEST_RETURN_IF_NOT(input0_producer != nullptr);
        TEST_RETURN_IF_NOT(input0_producer->OpType() == "MatMul");

        // input[1] should be the skip connection: a graph input (no producer)
        const Node* input1_producer = graph.GetProducerNode(input1->Name());
        TEST_RETURN_IF_NOT(input1_producer == nullptr);
        bool is_graph_input1 = false;
        for (const auto* gi : graph.GetInputs()) {
          if (gi->Name() == input1->Name()) {
            is_graph_input1 = true;
            break;
          }
        }
        TEST_RETURN_IF_NOT(is_graph_input1);

        // input[4] should be an initializer (the fused bias), identified by name
        const Node* input4_producer = graph.GetProducerNode(input4->Name());
        TEST_RETURN_IF_NOT(input4_producer == nullptr);
        const ONNX_NAMESPACE::TensorProto* bias_initializer = nullptr;
        TEST_RETURN_IF_NOT(graph.GetInitializedTensor(input4->Name(), bias_initializer));
        TEST_RETURN_IF_NOT(bias_initializer != nullptr);
      }
    }
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 17, *logger_,
                                        std::make_unique<BiasSkipLayerNormFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

// Same as above, but bias is Add input[0] (not input[1]).
TEST_F(GraphTransformationTests, BiasSkipLayerNormFusion_BiasAsFirstAddInput) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* matmul_a = builder.MakeInput<float>({2, 4, 8}, -1.0f, 1.0f);
    auto* matmul_b = builder.MakeInitializer<float>({8, 4}, -1.0f, 1.0f);
    auto* skip = builder.MakeInput<float>({2, 4, 4}, -1.0f, 1.0f);
    auto* gamma = builder.MakeInitializer<float>({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto* beta = builder.MakeInitializer<float>({4}, {0.0f, 0.0f, 0.0f, 0.0f});
    auto* bias = builder.MakeInitializer<float>({4}, {0.1f, 0.2f, 0.3f, 0.4f});

    auto* matmul_out = builder.MakeIntermediate();
    auto* add_out = builder.MakeIntermediate();
    auto* sln_out = builder.MakeOutput();

    builder.AddNode("MatMul", {matmul_a, matmul_b}, {matmul_out});
    // bias is Add input[0], MatMul output is Add input[1]
    builder.AddNode("Add", {bias, matmul_out}, {add_out});
    builder.AddNode("SkipLayerNormalization", {add_out, skip, gamma, beta}, {sln_out}, kMSDomain);
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count["Add"] == 0);
    TEST_RETURN_IF_NOT(op_count["com.microsoft.SkipLayerNormalization"] == 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "SkipLayerNormalization") {
        TEST_RETURN_IF_NOT(node.InputDefs().size() == 5u);

        // Verify wiring for this scenario as well: input[0] from MatMul, input[1] is skip input,
        // and input[4] is an initializer (the fused bias).
        const auto& input_defs = node.InputDefs();
        auto* input0 = input_defs[0];
        auto* input1 = input_defs[1];
        auto* input4 = input_defs[4];

        // input[0] should come from MatMul
        const Node* input0_producer = graph.GetProducerNode(input0->Name());
        TEST_RETURN_IF_NOT(input0_producer != nullptr);
        TEST_RETURN_IF_NOT(input0_producer->OpType() == "MatMul");

        // input[1] should be the skip connection: a graph input (no producer)
        const Node* input1_producer = graph.GetProducerNode(input1->Name());
        TEST_RETURN_IF_NOT(input1_producer == nullptr);
        bool is_graph_input1 = false;
        for (const auto* gi : graph.GetInputs()) {
          if (gi->Name() == input1->Name()) {
            is_graph_input1 = true;
            break;
          }
        }
        TEST_RETURN_IF_NOT(is_graph_input1);

        // input[4] should be an initializer (the fused bias), identified by name
        const Node* input4_producer = graph.GetProducerNode(input4->Name());
        TEST_RETURN_IF_NOT(input4_producer == nullptr);
        const ONNX_NAMESPACE::TensorProto* bias_initializer = nullptr;
        TEST_RETURN_IF_NOT(graph.GetInitializedTensor(input4->Name(), bias_initializer));
        TEST_RETURN_IF_NOT(bias_initializer != nullptr);
      }
    }
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 17, *logger_,
                                        std::make_unique<BiasSkipLayerNormFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

// Add(MatMul_out, bias_1D) is connected to SLN input[1] (the "skip" input).
TEST_F(GraphTransformationTests, BiasSkipLayerNormFusion_AddAtSkipInput) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({2, 4, 4}, -1.0f, 1.0f);
    auto* matmul_a = builder.MakeInput<float>({2, 4, 8}, -1.0f, 1.0f);
    auto* matmul_b = builder.MakeInitializer<float>({8, 4}, -1.0f, 1.0f);
    auto* gamma = builder.MakeInitializer<float>({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto* beta = builder.MakeInitializer<float>({4}, {0.0f, 0.0f, 0.0f, 0.0f});
    auto* bias = builder.MakeInitializer<float>({4}, {0.1f, 0.2f, 0.3f, 0.4f});

    auto* matmul_out = builder.MakeIntermediate();
    auto* add_out = builder.MakeIntermediate();
    auto* sln_out = builder.MakeOutput();

    builder.AddNode("MatMul", {matmul_a, matmul_b}, {matmul_out});
    builder.AddNode("Add", {matmul_out, bias}, {add_out});
    // add_out at SLN input[1] (skip), primary input at input[0]
    builder.AddNode("SkipLayerNormalization", {input, add_out, gamma, beta}, {sln_out}, kMSDomain);
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count["Add"] == 0);
    TEST_RETURN_IF_NOT(op_count["com.microsoft.SkipLayerNormalization"] == 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "SkipLayerNormalization") {
        TEST_RETURN_IF_NOT(node.InputDefs().size() == 5u);

        const auto& input_defs = node.InputDefs();

        // input[0] should be the original graph input (unchanged – Add fed SLN.input[1], so
        // only SLN.input[1] is replaced with the MatMul output; input[0] keeps its original value).
        const Node* input0_producer = graph.GetProducerNode(input_defs[0]->Name());
        TEST_RETURN_IF_NOT(input0_producer == nullptr);
        bool is_graph_input0 = false;
        for (const auto* gi : graph.GetInputs()) {
          if (gi->Name() == input_defs[0]->Name()) {
            is_graph_input0 = true;
            break;
          }
        }
        TEST_RETURN_IF_NOT(is_graph_input0);

        // input[1] should come from MatMul (the bias-Add was at SLN.input[1])
        const Node* input1_producer = graph.GetProducerNode(input_defs[1]->Name());
        TEST_RETURN_IF_NOT(input1_producer != nullptr);
        TEST_RETURN_IF_NOT(input1_producer->OpType() == "MatMul");

        // input[4] should be an initializer (the fused bias)
        const Node* input4_producer = graph.GetProducerNode(input_defs[4]->Name());
        TEST_RETURN_IF_NOT(input4_producer == nullptr);
        const ONNX_NAMESPACE::TensorProto* bias_initializer = nullptr;
        TEST_RETURN_IF_NOT(graph.GetInitializedTensor(input_defs[4]->Name(), bias_initializer));
        TEST_RETURN_IF_NOT(bias_initializer != nullptr);
      }
    }
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 17, *logger_,
                                        std::make_unique<BiasSkipLayerNormFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

// Cast variant: MatMul → Cast → Add(bias_1D) → SLN(4 inputs).
// Models using fp16 precision commonly insert a Cast between MatMul and the bias Add.
TEST_F(GraphTransformationTests, BiasSkipLayerNormFusion_WithCast) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* matmul_a = builder.MakeInput<onnxruntime::MLFloat16>({2, 4, 8}, MLFloat16(-1.0f), MLFloat16(1.0f));
    auto* matmul_b = builder.MakeInitializer<onnxruntime::MLFloat16>({8, 4}, MLFloat16(-1.0f), MLFloat16(1.0f));
    auto* skip = builder.MakeInput<float>({2, 4, 4}, -1.0f, 1.0f);
    auto* gamma = builder.MakeInitializer<float>({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto* beta = builder.MakeInitializer<float>({4}, {0.0f, 0.0f, 0.0f, 0.0f});
    auto* bias = builder.MakeInitializer<float>({4}, {0.1f, 0.2f, 0.3f, 0.4f});

    auto* matmul_out = builder.MakeIntermediate();
    auto* cast_out = builder.MakeIntermediate();
    auto* add_out = builder.MakeIntermediate();
    auto* sln_out = builder.MakeOutput();

    builder.AddNode("MatMul", {matmul_a, matmul_b}, {matmul_out});
    builder.AddNode("Cast", {matmul_out}, {cast_out})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    builder.AddNode("Add", {cast_out, bias}, {add_out});
    builder.AddNode("SkipLayerNormalization", {add_out, skip, gamma, beta}, {sln_out}, kMSDomain);
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count["Add"] == 0);
    TEST_RETURN_IF_NOT(op_count["com.microsoft.SkipLayerNormalization"] == 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "SkipLayerNormalization") {
        TEST_RETURN_IF_NOT(node.InputDefs().size() == 5u);

        const auto& input_defs = node.InputDefs();

        // input[0] should come from Cast (MatMul → Cast → fused SLN)
        const Node* input0_producer = graph.GetProducerNode(input_defs[0]->Name());
        TEST_RETURN_IF_NOT(input0_producer != nullptr);
        TEST_RETURN_IF_NOT(input0_producer->OpType() == "Cast");

        // input[1] should be the skip connection: a graph input (no producer)
        const Node* input1_producer = graph.GetProducerNode(input_defs[1]->Name());
        TEST_RETURN_IF_NOT(input1_producer == nullptr);
        bool is_graph_input1 = false;
        for (const auto* gi : graph.GetInputs()) {
          if (gi->Name() == input_defs[1]->Name()) {
            is_graph_input1 = true;
            break;
          }
        }
        TEST_RETURN_IF_NOT(is_graph_input1);

        // input[4] should be an initializer (the fused bias)
        const Node* input4_producer = graph.GetProducerNode(input_defs[4]->Name());
        TEST_RETURN_IF_NOT(input4_producer == nullptr);
        const ONNX_NAMESPACE::TensorProto* bias_initializer = nullptr;
        TEST_RETURN_IF_NOT(graph.GetInitializedTensor(input_defs[4]->Name(), bias_initializer));
        TEST_RETURN_IF_NOT(bias_initializer != nullptr);
      }
    }
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 17, *logger_,
                                        std::make_unique<BiasSkipLayerNormFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

// Cast variant negative test: bias is 1D but its length is incompatible with gamma/beta.
// This guards against fusing dimension-mismatched biases when hidden-size validation is applied
// on the Cast path.
TEST_F(GraphTransformationTests, BiasSkipLayerNormFusion_WithCast_BiasHiddenSizeMismatch) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* matmul_a = builder.MakeInput<onnxruntime::MLFloat16>({2, 4, 8}, MLFloat16(-1.0f), MLFloat16(1.0f));
    auto* matmul_b = builder.MakeInitializer<onnxruntime::MLFloat16>({8, 4}, MLFloat16(-1.0f), MLFloat16(1.0f));
    auto* skip = builder.MakeInput<float>({2, 4, 4}, -1.0f, 1.0f);
    auto* gamma = builder.MakeInitializer<float>({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto* beta = builder.MakeInitializer<float>({4}, {0.0f, 0.0f, 0.0f, 0.0f});
    // Intentionally use a 1D bias whose length does not match gamma/beta (size 4).
    // bias{1} broadcasts validly with cast_out{2,4,4}, but bias_hidden_size(1) != sln_hidden_size(4)
    // so the fusion is blocked.
    auto* bias = builder.MakeInitializer<float>({1}, {0.5f});

    auto* matmul_out = builder.MakeIntermediate();
    auto* cast_out = builder.MakeIntermediate();
    auto* add_out = builder.MakeIntermediate();
    auto* sln_out = builder.MakeOutput();

    builder.AddNode("MatMul", {matmul_a, matmul_b}, {matmul_out});
    builder.AddNode("Cast", {matmul_out}, {cast_out})
        .AddAttribute("to", static_cast<int64_t>(ONNX_NAMESPACE::TensorProto_DataType_FLOAT));
    builder.AddNode("Add", {cast_out, bias}, {add_out});
    builder.AddNode("SkipLayerNormalization", {add_out, skip, gamma, beta}, {sln_out}, kMSDomain);
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count = CountOpsInGraph(graph);
    // Fusion should not occur: Add must remain, and SkipLayerNormalization must keep 4 inputs.
    TEST_RETURN_IF_NOT(op_count["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count["com.microsoft.SkipLayerNormalization"] == 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "SkipLayerNormalization") {
        TEST_RETURN_IF_NOT(node.InputDefs().size() == 4u);
      }
    }
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 17, *logger_,
                                        std::make_unique<BiasSkipLayerNormFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

// Fusion must NOT occur when the bias is 2D (not 1D).
TEST_F(GraphTransformationTests, BiasSkipLayerNormFusion_NoFusion_2DBias) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* matmul_a = builder.MakeInput<float>({2, 4, 8}, -1.0f, 1.0f);
    auto* matmul_b = builder.MakeInitializer<float>({8, 4}, -1.0f, 1.0f);
    auto* skip = builder.MakeInput<float>({2, 4, 4}, -1.0f, 1.0f);
    auto* gamma = builder.MakeInitializer<float>({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto* beta = builder.MakeInitializer<float>({4}, {0.0f, 0.0f, 0.0f, 0.0f});
    // 2D bias – should prevent fusion
    auto* bias_2d = builder.MakeInitializer<float>({1, 4}, {0.1f, 0.2f, 0.3f, 0.4f});

    auto* matmul_out = builder.MakeIntermediate();
    auto* add_out = builder.MakeIntermediate();
    auto* sln_out = builder.MakeOutput();

    builder.AddNode("MatMul", {matmul_a, matmul_b}, {matmul_out});
    builder.AddNode("Add", {matmul_out, bias_2d}, {add_out});
    builder.AddNode("SkipLayerNormalization", {add_out, skip, gamma, beta}, {sln_out}, kMSDomain);
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count = CountOpsInGraph(graph);
    // Graph should be unchanged: Add and 4-input SLN both remain.
    TEST_RETURN_IF_NOT(op_count["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count["com.microsoft.SkipLayerNormalization"] == 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "SkipLayerNormalization") {
        TEST_RETURN_IF_NOT(node.InputDefs().size() == 4u);
      }
    }
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 17, *logger_,
                                        std::make_unique<BiasSkipLayerNormFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

// Fusion must NOT occur when the SLN node already has 5 inputs (bias already absorbed).
TEST_F(GraphTransformationTests, BiasSkipLayerNormFusion_NoFusion_SLNHas5Inputs) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input = builder.MakeInput<float>({2, 4, 4}, -1.0f, 1.0f);
    auto* skip = builder.MakeInput<float>({2, 4, 4}, -1.0f, 1.0f);
    auto* gamma = builder.MakeInitializer<float>({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto* beta = builder.MakeInitializer<float>({4}, {0.0f, 0.0f, 0.0f, 0.0f});
    auto* bias = builder.MakeInitializer<float>({4}, {0.1f, 0.2f, 0.3f, 0.4f});
    auto* sln_out = builder.MakeOutput();

    // SLN already has 5 inputs – no further fusion should happen.
    builder.AddNode("SkipLayerNormalization", {input, skip, gamma, beta, bias}, {sln_out}, kMSDomain);
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count["com.microsoft.SkipLayerNormalization"] == 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "SkipLayerNormalization") {
        TEST_RETURN_IF_NOT(node.InputDefs().size() == 5u);
      }
    }
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 17, *logger_,
                                        std::make_unique<BiasSkipLayerNormFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

// Fusion must NOT occur when the Add node feeds multiple consumers (the output is used both by
// SLN and by another node, so removing Add would drop the other consumer's input).
TEST_F(GraphTransformationTests, BiasSkipLayerNormFusion_NoFusion_AddHasMultipleConsumers) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* matmul_a = builder.MakeInput<float>({2, 4, 8}, -1.0f, 1.0f);
    auto* matmul_b = builder.MakeInitializer<float>({8, 4}, -1.0f, 1.0f);
    auto* skip = builder.MakeInput<float>({2, 4, 4}, -1.0f, 1.0f);
    auto* gamma = builder.MakeInitializer<float>({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto* beta = builder.MakeInitializer<float>({4}, {0.0f, 0.0f, 0.0f, 0.0f});
    auto* bias = builder.MakeInitializer<float>({4}, {0.1f, 0.2f, 0.3f, 0.4f});

    auto* matmul_out = builder.MakeIntermediate();
    auto* add_out = builder.MakeIntermediate();
    auto* sln_out = builder.MakeOutput();
    auto* identity_out = builder.MakeOutput();

    builder.AddNode("MatMul", {matmul_a, matmul_b}, {matmul_out});
    builder.AddNode("Add", {matmul_out, bias}, {add_out});
    // add_out feeds both SLN and an Identity node – Add has 2 consumers.
    builder.AddNode("SkipLayerNormalization", {add_out, skip, gamma, beta}, {sln_out}, kMSDomain);
    builder.AddNode("Identity", {add_out}, {identity_out});
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count = CountOpsInGraph(graph);
    // Add must NOT be removed because it has multiple consumers.
    TEST_RETURN_IF_NOT(op_count["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count["com.microsoft.SkipLayerNormalization"] == 1);
    for (auto& node : graph.Nodes()) {
      if (node.OpType() == "SkipLayerNormalization") {
        TEST_RETURN_IF_NOT(node.InputDefs().size() == 4u);
      }
    }
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 17, *logger_,
                                        std::make_unique<BiasSkipLayerNormFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

// Verify that fusion preserves downstream edges when the SLN output feeds another node.
// This exercises the edge-rewiring code path: the fused SLN node must inherit all consumers
// of the original SLN node.
TEST_F(GraphTransformationTests, BiasSkipLayerNormFusion_DownstreamConsumerPreserved) {
  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* matmul_a = builder.MakeInput<float>({2, 4, 8}, -1.0f, 1.0f);
    auto* matmul_b = builder.MakeInitializer<float>({8, 4}, -1.0f, 1.0f);
    auto* skip = builder.MakeInput<float>({2, 4, 4}, -1.0f, 1.0f);
    auto* gamma = builder.MakeInitializer<float>({4}, {1.0f, 1.0f, 1.0f, 1.0f});
    auto* beta = builder.MakeInitializer<float>({4}, {0.0f, 0.0f, 0.0f, 0.0f});
    auto* bias = builder.MakeInitializer<float>({4}, {0.1f, 0.2f, 0.3f, 0.4f});

    auto* matmul_out = builder.MakeIntermediate();
    auto* add_out = builder.MakeIntermediate();
    auto* sln_out = builder.MakeIntermediate();  // intermediate: SLN output feeds Identity
    auto* identity_out = builder.MakeOutput();

    builder.AddNode("MatMul", {matmul_a, matmul_b}, {matmul_out});
    builder.AddNode("Add", {matmul_out, bias}, {add_out});
    builder.AddNode("SkipLayerNormalization", {add_out, skip, gamma, beta}, {sln_out}, kMSDomain);
    // Downstream consumer of the SLN output: must be preserved after fusion.
    builder.AddNode("Identity", {sln_out}, {identity_out});
  };

  auto post_graph_checker = [](Graph& graph) {
    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count["Add"] == 0);
    TEST_RETURN_IF_NOT(op_count["com.microsoft.SkipLayerNormalization"] == 1);
    TEST_RETURN_IF_NOT(op_count["Identity"] == 1);

    // The Identity node must still be wired to the fused SLN output.
    for (const auto& node : graph.Nodes()) {
      if (node.OpType() == "Identity") {
        TEST_RETURN_IF_NOT(node.InputDefs().size() == 1u);
        const Node* identity_input_producer = graph.GetProducerNode(node.InputDefs()[0]->Name());
        TEST_RETURN_IF_NOT(identity_input_producer != nullptr);
        TEST_RETURN_IF_NOT(identity_input_producer->OpType() == "SkipLayerNormalization");
        // The fused SLN must have 5 inputs.
        TEST_RETURN_IF_NOT(identity_input_producer->InputDefs().size() == 5u);
      }
    }
    return Status::OK();
  };

  ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, 17, *logger_,
                                        std::make_unique<BiasSkipLayerNormFusion>(),
                                        TransformerLevel::Level2, 1, nullptr, post_graph_checker));
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat1) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format1.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Gather"] == 0);
  ASSERT_TRUE(op_to_count["Add"] == 0);
  ASSERT_TRUE(op_to_count["ReduceSum"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.Attention"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.SkipLayerNormalization"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.EmbedLayerNormalization"] == 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat2) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format2.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

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
  ASSERT_TRUE(op_to_count["com.microsoft.Attention"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.SkipLayerNormalization"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.EmbedLayerNormalization"] == 1);
}

static void EmbedLayerNormFusionFormat3(const std::basic_string<ORTCHAR_T>& file_path, logging::Logger* logger) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, *logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Expand"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["LayerNormalization"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.SkipLayerNormalization"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Cast"], 3);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
  EXPECT_EQ(op_to_count["com.microsoft.EmbedLayerNormalization"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat3) {
  EmbedLayerNormFusionFormat3(MODEL_FOLDER "fusion/embed_layer_norm_format3.onnx", logger_.get());
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat3_OpSet13) {
  EmbedLayerNormFusionFormat3(MODEL_FOLDER "fusion/embed_layer_norm_format3_opset13.onnx", logger_.get());
}

static void EmbedLayerNormFusionFormat3NoCast(const std::basic_string<ORTCHAR_T>& file_path, logging::Logger* logger) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, *logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Expand"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["LayerNormalization"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.SkipLayerNormalization"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Cast"], 3);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
  EXPECT_EQ(op_to_count["com.microsoft.EmbedLayerNormalization"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat3NoCast) {
  EmbedLayerNormFusionFormat3NoCast(MODEL_FOLDER "fusion/embed_layer_norm_format3_no_cast.onnx", logger_.get());
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat3NoCast_OpSet13) {
  EmbedLayerNormFusionFormat3NoCast(MODEL_FOLDER "fusion/embed_layer_norm_format3_no_cast_opset13.onnx", logger_.get());
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat4) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/embed_layer_norm_format4.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger_));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger_));

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
  ASSERT_TRUE(op_to_count["com.microsoft.Attention"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.SkipLayerNormalization"] == 0);
  ASSERT_TRUE(op_to_count["com.microsoft.EmbedLayerNormalization"] == 1);
}

static void EmbedLayerNormFusionFormat5(const std::basic_string<ORTCHAR_T>& file_path, logging::Logger* logger) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, *logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["LayerNormalization"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.SkipLayerNormalization"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Cast"], 3);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
  EXPECT_EQ(op_to_count["com.microsoft.EmbedLayerNormalization"], 1);

  // Validate the position embedding input.
  double expected_value[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 8.0, 7.0, 6.0};
  for (const Node& node : graph.Nodes()) {
    if (node.OpType() == "EmbedLayerNormalization") {
      const ONNX_NAMESPACE::TensorProto* tensor_proto = graph_utils::GetConstantInitializer(graph, node.InputDefs()[3]->Name());
      ASSERT_TRUE(tensor_proto != nullptr);
      EXPECT_EQ(tensor_proto->data_type(), ONNX_NAMESPACE::TensorProto_DataType_FLOAT);

      Initializer initializer{graph, *tensor_proto, graph.ModelPath()};
      EXPECT_EQ(initializer.size(), std::size(expected_value));

      const float* data = initializer.data<float>();
      for (size_t i = 0; i < std::size(expected_value); i++) {
        EXPECT_EQ(data[i], static_cast<float>(expected_value[i]));
      }
    }
  }
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat5) {
  EmbedLayerNormFusionFormat5(MODEL_FOLDER "fusion/embed_layer_norm_format5.onnx", logger_.get());
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat5_OpSet13) {
  EmbedLayerNormFusionFormat5(MODEL_FOLDER "fusion/embed_layer_norm_format5_opset13.onnx", logger_.get());
}

static void EmbedLayerNormFusionFormat6(const std::basic_string<ORTCHAR_T>& file_path, logging::Logger* logger) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, *logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Expand"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["Reshape"], 0);
  EXPECT_EQ(op_to_count["Equal"], 0);
  EXPECT_EQ(op_to_count["Where"], 0);
  EXPECT_EQ(op_to_count["LayerNormalization"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.SkipLayerNormalization"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
  EXPECT_EQ(op_to_count["MatMul"], 1);
  EXPECT_EQ(op_to_count["Add"], 2);
  EXPECT_EQ(op_to_count["Cast"], 3);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
  EXPECT_EQ(op_to_count["com.microsoft.EmbedLayerNormalization"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat6) {
  EmbedLayerNormFusionFormat6(MODEL_FOLDER "fusion/embed_layer_norm_format6.onnx", logger_.get());
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat6_OpSet13) {
  EmbedLayerNormFusionFormat6(MODEL_FOLDER "fusion/embed_layer_norm_format6_opset13.onnx", logger_.get());
}

static void TestEmbedLayerNormFusionDistilBert(const std::basic_string<ORTCHAR_T>& model_uri,
                                               std::map<std::string, int>& op_to_count,
                                               logging::Logger* logger) {
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger));
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

  op_to_count = CountOpsInGraph(graph);
}

// DistilBert
TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat7) {
  std::map<std::string, int> op_to_count;
  TestEmbedLayerNormFusionDistilBert(MODEL_FOLDER "fusion/embed_layer_norm_format7.onnx", op_to_count, logger_.get());
  EXPECT_EQ(op_to_count["com.microsoft.EmbedLayerNormalization"], 1);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
  EXPECT_EQ(op_to_count["Cast"], 2);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat7_OpSet13) {
  std::map<std::string, int> op_to_count;
  TestEmbedLayerNormFusionDistilBert(MODEL_FOLDER "fusion/embed_layer_norm_format7_opset13.onnx", op_to_count, logger_.get());
  EXPECT_EQ(op_to_count["com.microsoft.EmbedLayerNormalization"], 1);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
  EXPECT_EQ(op_to_count["Cast"], 2);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat8) {
  std::map<std::string, int> op_to_count;
  TestEmbedLayerNormFusionDistilBert(MODEL_FOLDER "fusion/embed_layer_norm_format8.onnx", op_to_count, logger_.get());
  EXPECT_EQ(op_to_count["com.microsoft.EmbedLayerNormalization"], 1);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
  EXPECT_EQ(op_to_count["Cast"], 2);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat8_OpSet13) {
  std::map<std::string, int> op_to_count;
  TestEmbedLayerNormFusionDistilBert(MODEL_FOLDER "fusion/embed_layer_norm_format8_opset13.onnx", op_to_count, logger_.get());
  EXPECT_EQ(op_to_count["com.microsoft.EmbedLayerNormalization"], 1);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
  EXPECT_EQ(op_to_count["Cast"], 2);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat9) {
  std::map<std::string, int> op_to_count;
  TestEmbedLayerNormFusionDistilBert(MODEL_FOLDER "fusion/embed_layer_norm_format9.onnx", op_to_count, logger_.get());
  EXPECT_EQ(op_to_count["com.microsoft.EmbedLayerNormalization"], 1);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
  EXPECT_EQ(op_to_count["Cast"], 2);
  EXPECT_EQ(op_to_count["Shape"], 1);
  EXPECT_EQ(op_to_count["Gather"], 2);
  EXPECT_EQ(op_to_count["Unsqueeze"], 2);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionFormat9_OpSet13) {
  std::map<std::string, int> op_to_count;
  TestEmbedLayerNormFusionDistilBert(MODEL_FOLDER "fusion/embed_layer_norm_format9_opset13.onnx", op_to_count, logger_.get());
  EXPECT_EQ(op_to_count["com.microsoft.EmbedLayerNormalization"], 1);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 1);
  EXPECT_EQ(op_to_count["Cast"], 2);
  EXPECT_EQ(op_to_count["Shape"], 1);
  EXPECT_EQ(op_to_count["Gather"], 2);
  EXPECT_EQ(op_to_count["Unsqueeze"], 2);
  EXPECT_EQ(op_to_count["ReduceSum"], 1);
}

static void EmbedLayerNormFusionFormatMultiple(const std::basic_string<ORTCHAR_T>& file_path, logging::Logger* logger) {
  std::shared_ptr<Model> p_model;
  ASSERT_TRUE(Model::Load(file_path, p_model, nullptr, *logger).IsOK());
  Graph& graph = p_model->MainGraph();

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<EmbedLayerNormFusion>(), TransformerLevel::Level2));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  EXPECT_EQ(op_to_count["Shape"], 0);
  EXPECT_EQ(op_to_count["Expand"], 0);
  EXPECT_EQ(op_to_count["Gather"], 0);
  EXPECT_EQ(op_to_count["Unsqueeze"], 0);
  EXPECT_EQ(op_to_count["LayerNormalization"], 0);
  EXPECT_EQ(op_to_count["com.microsoft.SkipLayerNormalization"], 0);
  EXPECT_EQ(op_to_count["ReduceSum"], 2);
  EXPECT_EQ(op_to_count["MatMul"], 2);
  EXPECT_EQ(op_to_count["Add"], 5);
  EXPECT_EQ(op_to_count["Cast"], 6);
  EXPECT_EQ(op_to_count["com.microsoft.Attention"], 2);
  EXPECT_EQ(op_to_count["com.microsoft.EmbedLayerNormalization"], 2);
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionMultiple) {
  EmbedLayerNormFusionFormatMultiple(MODEL_FOLDER "fusion/embed_layer_norm_multiple.onnx", logger_.get());
}

TEST_F(GraphTransformationTests, EmbedLayerNormFusionMultiple_OpSet13) {
  EmbedLayerNormFusionFormatMultiple(MODEL_FOLDER "fusion/embed_layer_norm_multiple_opset13.onnx", logger_.get());
}

#endif

}  // namespace test
}  // namespace onnxruntime
