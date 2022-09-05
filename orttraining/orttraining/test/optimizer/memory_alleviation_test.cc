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
#include "orttraining/core/optimizer/memory_alleviation.h"

using namespace std;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace test {

#define MODEL_FOLDER ORT_TSTR("testdata/transform/recompute/")

TEST(MemoryAlleviationTests, GeluRecompute) {
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

  const std::string alleviation_config("Gelu:1");
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MemoryAlleviation>(alleviation_config), TransformerLevel::Level2));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

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

TEST(MemoryAlleviationTests, DropoutRecompute) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "recompute_dropout.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  std::map<std::string, int> op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Where"] == 4);
  ASSERT_TRUE(op_to_count["com.microsoft.BitmaskDropout"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.YieldOp"] == 1);
  ASSERT_TRUE(op_to_count["Softmax"] == 1);
  ASSERT_TRUE(op_to_count["MatMul"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == 2);
  ASSERT_TRUE(op_to_count["Reshape"] == 2);
  ASSERT_TRUE(op_to_count["com.microsoft.SoftmaxGrad_13"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.BitmaskDropoutGrad"] == 1);

  onnxruntime::GraphTransformerManager graph_transformation_mgr{5};

  const std::string alleviation_config("Dropout:1");
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MemoryAlleviation>(alleviation_config), TransformerLevel::Level2));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_TRUE(op_to_count["Where"] == 5);
  ASSERT_TRUE(op_to_count["com.microsoft.BitmaskDropout"] == 2);
  ASSERT_TRUE(op_to_count["com.microsoft.YieldOp"] == 1);
  ASSERT_TRUE(op_to_count["Softmax"] == 1);
  ASSERT_TRUE(op_to_count["MatMul"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == 2);
  ASSERT_TRUE(op_to_count["Reshape"] == 3);
  ASSERT_TRUE(op_to_count["com.microsoft.SoftmaxGrad_13"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.BitmaskDropoutGrad"] == 1);

  Node* recompute_where_node{nullptr};
  Node* recompute_dropout_node{nullptr};
  Node* recompute_reshape_node{nullptr};
  Node* original_dropout_node{nullptr};
  Node* original_softmax_node{nullptr};
  for (auto& node : graph.Nodes()) {
    if (node.Priority() == static_cast<int>(ExecutionPriority::LOCAL_LOW)) {
      if (node.OpType().compare("Reshape") == 0) {
        recompute_reshape_node = &node;
      } else if (node.OpType().compare("BitmaskDropout") == 0) {
        recompute_dropout_node = &node;
      } else if (node.OpType().compare("Where") == 0) {
        recompute_where_node = &node;
      }
    } else if (node.Priority() == static_cast<int>(ExecutionPriority::DEFAULT)) {
      if (node.OpType().compare("BitmaskDropout") == 0) {
        original_dropout_node = &node;
      } else if (node.OpType().compare("Softmax") == 0) {
        original_softmax_node = &node;
      }
    }
  }

  const Node* value_layer_grad_node = graph.GetProducerNode("value_layer_grad");

  ASSERT_TRUE(recompute_where_node);
  ASSERT_TRUE(recompute_dropout_node);
  ASSERT_TRUE(recompute_reshape_node);
  ASSERT_TRUE(original_dropout_node);
  ASSERT_TRUE(original_softmax_node);
  ASSERT_TRUE(value_layer_grad_node);

  ASSERT_EQ(recompute_where_node->MutableInputDefs()[1]->Name(), original_softmax_node->MutableOutputDefs()[0]->Name());
  ASSERT_EQ(recompute_dropout_node->MutableInputDefs()[0]->Name(), recompute_where_node->MutableOutputDefs()[0]->Name());
  ASSERT_EQ(recompute_reshape_node->MutableInputDefs()[0]->Name(), recompute_dropout_node->MutableOutputDefs()[0]->Name());
  ASSERT_EQ(value_layer_grad_node->InputDefs()[0]->Name(), recompute_reshape_node->MutableOutputDefs()[0]->Name());

  ASSERT_EQ(recompute_dropout_node->GetAttributes().at("seed").i(), original_dropout_node->GetAttributes().at("seed").i());
  ASSERT_EQ(recompute_where_node->Priority(), static_cast<int>(ExecutionPriority::LOCAL_LOW));
  ASSERT_EQ(recompute_dropout_node->Priority(), static_cast<int>(ExecutionPriority::LOCAL_LOW));
  ASSERT_EQ(recompute_reshape_node->Priority(), static_cast<int>(ExecutionPriority::LOCAL_LOW));
  ASSERT_EQ(original_dropout_node->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
  ASSERT_EQ(original_softmax_node->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
  ASSERT_EQ(value_layer_grad_node->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
}

TEST(MemoryAlleviationTests, TileRecompute) {
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

  const std::string alleviation_config("Tile:1");
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MemoryAlleviation>(alleviation_config), TransformerLevel::Level2));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level2, *logger));

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
