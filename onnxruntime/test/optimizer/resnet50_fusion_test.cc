// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <vector>
#include "core/graph/graph.h"
#include "core/optimizer/conv_activation_fusion.h"
#include "core/optimizer/conv_add_fusion.h"
#include "core/optimizer/conv_add_act_fusion.h"
#include "core/mlas/inc/mlas.h"
#include "gtest/gtest.h"
#include "graph_transform_test_builder.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

#include "test/optimizer/graph_transform_test_builder.h"
#include "test/optimizer/graph_transform_test_fixture.h"

namespace onnxruntime {
namespace test {
// #define MLAS_F16VEC_INTRINSICS_SUPPORTED

#define MODEL_FOLDER ORT_TSTR("testdata/transform/")
#if defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && !defined(DISABLE_CONTRIB_OPS)

class ResNet50FusionTests : public ::testing::Test {
 protected:
  ResNet50FusionTests() : logger(DefaultLoggingManager().CreateLogger("ResNet50FusionTest")) {
  }
  std::unique_ptr<logging::Logger> logger;
};

TEST_F(ResNet50FusionTests, FuseConvAddReluUnitTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add_relu_fp16.onnx";
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
  ASSERT_STATUS_OK(Model::Save(*p_model, ORT_TSTR("conv_add_relu_fp16_fused.onnx")));
  ASSERT_TRUE(op_to_count["Add"] == 0);   // Add removed from graph
  ASSERT_TRUE(op_to_count["Relu"] == 0);  // Relu removed from graph
}
TEST_F(ResNet50FusionTests, FuseConvAddUnitTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_add_fp16.onnx";
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
  ASSERT_STATUS_OK(Model::Save(*p_model, ORT_TSTR("conv_add_fp16_fused.onnx")));
  ASSERT_TRUE(op_to_count["Add"] == 0);  // Add removed from graph
}
TEST_F(ResNet50FusionTests, FuseConvReluUnitTest) {
  constexpr const ORTCHAR_T* model_uri = MODEL_FOLDER "fusion/conv_relu_fp16.onnx";
  std::shared_ptr<Model> p_model;
  ASSERT_STATUS_OK(Model::Load(model_uri, p_model, nullptr, *logger));
  Graph& graph = p_model->MainGraph();
  for (auto& node : p_model->MainGraph().Nodes()) {
    node.SetExecutionProviderType(kCpuExecutionProvider);
  }
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Relu"] == 1);
  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(std::make_unique<ConvActivationFusion>(), TransformerLevel::Level3));
  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger));
  op_to_count = CountOpsInGraph(graph);
  ASSERT_STATUS_OK(Model::Save(*p_model, ORT_TSTR("conv_relu_fp16_fused.onnx")));
  ASSERT_TRUE(op_to_count["Relu"] == 0);  // Add removed from graph
}
#endif  // defined(MLAS_F16VEC_INTRINSICS_SUPPORTED) && !defined(DISABLE_CONTRIB_OPS)
}  // namespace test
}  // namespace onnxruntime
