// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4244)
#endif

#include <random>
#include "core/graph/onnx_protobuf.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include "asserts.h"
#include "core/common/span_utils.h"
#include "core/framework/data_types.h"
#include "core/framework/ort_value.h"
#include "core/graph/graph_utils.h"
#include "core/graph/graph_viewer.h"
#include "core/graph/model.h"
#include "core/optimizer/utils.h"
#include "core/platform/env.h"
#include "core/session/inference_session.h"
#include "core/util/math.h"
#include "test/framework/test_utils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "orttraining/core/optimizer/memory_optimizer/memory_optimizer.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/recompute/")

TEST(MemoryOptimizerTests, GeluRecompute) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "recompute_gelu.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Gemm"] == 5);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.YieldOp"] == 1);
  ASSERT_TRUE(op_to_count["ReduceSum"] == 2);
  ASSERT_TRUE(op_to_count["com.microsoft.GeluGrad"] == 1);

  std::string gelu_node_name;
  for (auto& node : graph.Nodes()) {
    if (node.OpType().compare("Gelu") == 0) {
      gelu_node_name = node.Name();
      break;
    }
  }

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};

  const std::string alleviation_config("Gelu+:1:-1");
  const std::string alleviation_level("1");
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MemoryOptimizer>(alleviation_config, alleviation_level), TransformerLevel::Level3));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Gemm"] == 5);
  ASSERT_TRUE(op_to_count["com.microsoft.Gelu"] == 2);
  ASSERT_TRUE(op_to_count["com.microsoft.YieldOp"] == 1);
  ASSERT_TRUE(op_to_count["ReduceSum"] == 2);
  ASSERT_TRUE(op_to_count["com.microsoft.GeluGrad"] == 1);

  Node* recompute_gelu_node{nullptr};
  Node* original_gelu_node{nullptr};
  for (auto& node : graph.Nodes()) {
    if (node.OpType().compare("Gelu") == 0) {
      if (node.Name() != gelu_node_name) {
        recompute_gelu_node = &node;
      } else {
        original_gelu_node = &node;
      }
    }
  }

  ASSERT_EQ(recompute_gelu_node->MutableInputDefs()[0]->Name(), original_gelu_node->MutableInputDefs()[0]->Name());
  ASSERT_EQ(recompute_gelu_node->Priority(), static_cast<int>(ExecutionPriority::LOCAL_LOW));
  ASSERT_EQ(original_gelu_node->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
}

TEST(MemoryOptimizerTests, TileRecompute) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "recompute_tile.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Tile"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.YieldOp"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == 3);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};

  const std::string alleviation_config("Tile+:1:-1");
  const std::string alleviation_level("1");
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MemoryOptimizer>(alleviation_config, alleviation_level), TransformerLevel::Level3));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Tile"] == 2);
  ASSERT_TRUE(op_to_count["com.microsoft.YieldOp"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == 3);

  Node* recompute_tile_node{nullptr};
  Node* original_tile_node{nullptr};
  for (auto& node : graph.Nodes()) {
    if (node.Priority() == static_cast<int>(ExecutionPriority::LOCAL_LOW)) {
      if (node.OpType().compare("Tile") == 0) {
        recompute_tile_node = &node;
      }
    } else if (node.Priority() == static_cast<int>(ExecutionPriority::DEFAULT)) {
      if (node.OpType().compare("Tile") == 0) {
        original_tile_node = &node;
      }
    }
  }

  const Node* query_layer_grad_node = graph.GetProducerNode("query_layer_grad");

  ASSERT_TRUE(recompute_tile_node);
  ASSERT_TRUE(original_tile_node);
  ASSERT_TRUE(query_layer_grad_node);

  ASSERT_EQ(recompute_tile_node->MutableInputDefs()[0]->Name(), original_tile_node->MutableInputDefs()[0]->Name());
  ASSERT_EQ(query_layer_grad_node->InputDefs()[1]->Name(), recompute_tile_node->MutableOutputDefs()[0]->Name());

  ASSERT_EQ(recompute_tile_node->Priority(), static_cast<int>(ExecutionPriority::LOCAL_LOW));
  ASSERT_EQ(original_tile_node->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
  ASSERT_EQ(query_layer_grad_node->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
}

}  // namespace test
}  // namespace onnxruntime
