// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <vector>
#include "core/graph/graph.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_add_act_fusion.h"
#include "core/optimizer/rule_based_graph_transformer.h"
#include "gtest/gtest.h"
#include "graph_transform_test_builder.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
#ifndef DISABLE_CONTRIB_OPS
#define MODEL_FOLDER ORT_TSTR("testdata/transform/")
class ResNet50FusionTests : public ::testing::Test {
 protected:
  ResNet50FusionTests() : logger(DefaultLoggingManager().CreateLogger("ResNet50FusionTest")) {

  }
  std::unique_ptr<logging::Logger> logger;
};
TEST_F(ResNet50FusionTests, Fusion) {
  std::basic_string<ORTCHAR_T> const resnet_fp16_model = ORT_TSTR("fusion/resnet50.onnx");
  PathString const model_uri = PathString(MODEL_FOLDER) + resnet_fp16_model;
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger));
  Graph& graph = p_model->MainGraph();
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvAddFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger));
  ASSERT_STATUS_OK(Model::Save(*p_model, "resnet50_fused.onnx"));
}

TEST_F(ResNet50FusionTests, FuseConvAddRelu) {
  std::basic_string<ORTCHAR_T> const resnet_fp16_model = ORT_TSTR("fusion/resnet50.onnx");
  PathString const model_uri = PathString(MODEL_FOLDER) + resnet_fp16_model;
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvAddActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger));
  ASSERT_STATUS_OK(Model::Save(*p_model, "resnet50_fused_CAR.onnx"));
}

TEST_F(ResNet50FusionTests, Fp16Fusion) {
  std::basic_string<ORTCHAR_T> const resnet_fp16_model = ORT_TSTR("fusion/resnet50.fp16.onnx");
  PathString const model_uri = PathString(MODEL_FOLDER) + resnet_fp16_model;
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger));
  Graph& graph = p_model->MainGraph();
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  auto rule_transformer_L1 = std::make_unique<RuleBasedGraphTransformer>("RuleTransformer1");
  ASSERT_STATUS_OK(rule_transformer_L1->Register(std::make_unique<ConvAddFusion>()));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::move(rule_transformer_L1), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger));
  ASSERT_STATUS_OK(Model::Save(*p_model, "resnet50_fp16_fused.onnx"));
}

TEST_F(ResNet50FusionTests, Fp16FuseConvAddRelu) {
  std::basic_string<ORTCHAR_T> const resnet_fp16_model = ORT_TSTR("fusion/resnet50.fp16.onnx");
    PathString const model_uri = PathString(MODEL_FOLDER) + resnet_fp16_model;
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvAddActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger));
  ASSERT_STATUS_OK(Model::Save(*p_model, "resnet50_fp16_fused_CAR.onnx"));
}

TEST_F(ResNet50FusionTests, FuseCpuConvAddRelu) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add_relu.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  ASSERT_TRUE(op_to_count["Relu"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvAddActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);   // Add removed from graph
  ASSERT_TRUE(op_to_count["Relu"] == 0);  // Relu removed from graph
  ASSERT_STATUS_OK(Model::Save(*p_model, "fused_conv_add_relu.onnx"));
}

TEST_F(ResNet50FusionTests, FuseCpuConvAdd) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvAddActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Add"] == 0);  // Add removed
  ASSERT_STATUS_OK(Model::Save(*p_model, "fused_conv_add.onnx"));
}
#endif  // DISABLE_CONTRIB_OPS

}  // namespace test
}  // namespace onnxruntime