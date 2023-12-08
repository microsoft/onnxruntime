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
#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "orttraining/core/optimizer/memory_optimizer/memory_optimizer.h"
#include "orttraining/core/optimizer/memory_optimizer/memory_insight.h"

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
  const std::string probe_config("1:0");
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MemoryOptimizer>(alleviation_config, probe_config), TransformerLevel::Level3));

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

  const std::string alleviation_config("Expand+Tile+:1:-1");
  const std::string probe_config("1:0");
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MemoryOptimizer>(alleviation_config, probe_config), TransformerLevel::Level3));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Tile"], 2);
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

  const Node* recompute_expand_node = graph.GetProducerNode(recompute_tile_node->InputDefs()[0]->Name());
  ASSERT_TRUE(recompute_expand_node);

  const Node* original_expand_node = graph.GetProducerNode(original_tile_node->InputDefs()[0]->Name());
  ASSERT_TRUE(original_expand_node);

  ASSERT_EQ(recompute_expand_node->InputDefs()[0]->Name(), original_expand_node->InputDefs()[0]->Name());
  ASSERT_EQ(query_layer_grad_node->InputDefs()[1]->Name(), recompute_tile_node->OutputDefs()[0]->Name());

  ASSERT_EQ(recompute_tile_node->Priority(), static_cast<int>(ExecutionPriority::LOCAL_LOW));
  ASSERT_EQ(original_tile_node->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
  ASSERT_EQ(query_layer_grad_node->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
}

TEST(MemoryOptimizerTests, TransformerPerLayerRecompute) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "3layer_bloom_optimized_training.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();

  // Find all optimizable subgraphs
  GraphViewer graph_viewer(graph);
  const std::string initial_mem_config("");
  const std::string probe_config("1:1");
  std::map<std::string, std::pair<std::string, int>>
      cluster_id_combinations_to_saved_symbolic_byte_map;
  std::string record_str =
      optimizer::memory_optimizer::GetSerializedORTModuleMemoryStat(graph_viewer,
                                                                    initial_mem_config,
                                                                    probe_config,
                                                                    *logger,
                                                                    cluster_id_combinations_to_saved_symbolic_byte_map,
                                                                    nullptr,
                                                                    nullptr);

  InlinedHashMap<std::string, optimizer::memory_optimizer::UserConfig> cluster_id_to_config_map;
  for (auto it = cluster_id_combinations_to_saved_symbolic_byte_map.begin();
       it != cluster_id_combinations_to_saved_symbolic_byte_map.end(); ++it) {
    std::string cluster_id = it->first;
    ORT_ENFORCE(optimizer::memory_optimizer::ParseOptimizationConfigFromString(cluster_id, cluster_id_to_config_map)
                    .IsOK());
  }
  std::ostringstream oss;
  int index = 0;
  for (auto it = cluster_id_to_config_map.begin(); it != cluster_id_to_config_map.end(); ++it) {
    if (it->second.type == optimizer::memory_optimizer::OptimizationType::Recompute) {
      oss << (index == 0 ? "" : ",") << it->first << ":1:-1";
      ++index;
    }
  }

  // Apply the transformer
  GraphTransformerManager graph_transformation_mgr{5};
  const std::string layer_wise_recompute_config(oss.str());
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MemoryOptimizer>(layer_wise_recompute_config, probe_config), TransformerLevel::Level3));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger));

  std::vector<const Node*> bw_nodes_in_expected_order;
  const Node* yield_op_node = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType().compare("YieldOp") == 0) {
      yield_op_node = &node;
    }
  }
  ASSERT_TRUE(yield_op_node != nullptr);
  bw_nodes_in_expected_order.push_back(yield_op_node);

  for (int layer_index = 2; layer_index >= 0; --layer_index) {
    const Node* input_layer_norm_grad_node = nullptr;
    {
      // The input of LayerNormalization node in Attention should not be recomputed for the transformer layerwise probe.
      auto consumers = graph.GetConsumerNodes("_original_module._original_model.transformer.h." +
                                              std::to_string(layer_index) + ".input_layernorm.weight");
      // Check there are two LayerNormalization nodes, one of them is the original one,
      // and the other is the recomputed one
      const Node* original_ln_node = nullptr;
      const Node* recompute_ln_node = nullptr;
      const Node* original_ln_node_parent_add_or_ln_node = nullptr;
      const Node* recompute_ln_node_parent_add_or_ln_node = nullptr;

      for (auto& consumer : consumers) {
        if (consumer->OpType().compare("LayerNormalization") == 0) {
          if (consumer->Name().find("_recompute") != std::string::npos) {
            recompute_ln_node = consumer;
            ASSERT_EQ(consumer->Priority(), static_cast<int>(ExecutionPriority::LOCAL_LOW));
            recompute_ln_node_parent_add_or_ln_node = graph.GetProducerNode(consumer->InputDefs()[0]->Name());
            ASSERT_TRUE(recompute_ln_node_parent_add_or_ln_node != nullptr);
            ASSERT_EQ(recompute_ln_node_parent_add_or_ln_node->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
            ASSERT_TRUE(recompute_ln_node_parent_add_or_ln_node->Name().find("_recompute") == std::string::npos);
          } else {
            original_ln_node = consumer;
            ASSERT_EQ(consumer->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
            original_ln_node_parent_add_or_ln_node = graph.GetProducerNode(consumer->InputDefs()[0]->Name());
            ASSERT_TRUE(original_ln_node_parent_add_or_ln_node);
            ASSERT_EQ(original_ln_node_parent_add_or_ln_node->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
            ASSERT_TRUE(original_ln_node_parent_add_or_ln_node->Name().find("_recompute") == std::string::npos);
          }
        } else if (consumer->OpType().compare("LayerNormalizationGrad") == 0) {
          input_layer_norm_grad_node = consumer;
          ASSERT_EQ(consumer->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
        }
      }

      ASSERT_TRUE(recompute_ln_node);
      ASSERT_TRUE(original_ln_node);
      ASSERT_TRUE(input_layer_norm_grad_node);
    }

    {
      auto consumers = graph.GetConsumerNodes("_original_module._original_model.transformer.h." +
                                              std::to_string(layer_index) + ".post_attention_layernorm.weight");
      // Check there are two LayerNormalization nodes, one of them is the original one,
      // and the other is the recomputed one
      const Node* original_ln_node = nullptr;
      const Node* recompute_ln_node = nullptr;
      const Node* original_ln_node_parent_add_node = nullptr;
      const Node* recompute_ln_node_parent_add_node = nullptr;
      const Node* ln_grad_node = nullptr;

      for (auto& consumer : consumers) {
        if (consumer->OpType().compare("LayerNormalization") == 0) {
          if (consumer->Name().find("_recompute") != std::string::npos) {
            recompute_ln_node = consumer;
            ASSERT_EQ(consumer->Priority(), static_cast<int>(ExecutionPriority::LOCAL_LOW));
            recompute_ln_node_parent_add_node = graph.GetProducerNode(consumer->InputDefs()[0]->Name());
            ASSERT_TRUE(recompute_ln_node_parent_add_node);
            ASSERT_EQ(recompute_ln_node_parent_add_node->OpType(), "Add");
            ASSERT_EQ(recompute_ln_node_parent_add_node->Priority(), static_cast<int>(ExecutionPriority::LOCAL_LOW));
            ASSERT_TRUE(recompute_ln_node_parent_add_node->Name().find("_recompute") != std::string::npos);
          } else {
            original_ln_node = consumer;
            ASSERT_EQ(consumer->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
            original_ln_node_parent_add_node = graph.GetProducerNode(consumer->InputDefs()[0]->Name());
            ASSERT_TRUE(original_ln_node_parent_add_node);
          }
        } else if (consumer->OpType().compare("LayerNormalizationGrad") == 0) {
          ln_grad_node = consumer;
          ASSERT_EQ(consumer->Priority(), static_cast<int>(ExecutionPriority::DEFAULT));
        }
      }

      ASSERT_TRUE(recompute_ln_node);
      ASSERT_TRUE(original_ln_node);
      ASSERT_TRUE(ln_grad_node);

      bw_nodes_in_expected_order.push_back(recompute_ln_node_parent_add_node);
      bw_nodes_in_expected_order.push_back(ln_grad_node);  // ln gradient need the recomputed ln node's add node as input
    }
    bw_nodes_in_expected_order.push_back(input_layer_norm_grad_node);
  }

  std::vector<size_t> nodes_in_topological_order;
  nodes_in_topological_order.reserve(bw_nodes_in_expected_order.size());
  const auto& node_topology_list = graph_viewer.GetNodesInTopologicalOrder();  // ExecutionOrder::PRIORITY_BASED

  size_t j = 0;
  for (auto node_index : node_topology_list) {
    auto* node_ptr = graph.GetNode(node_index);
    if (!node_ptr) continue;  // Node was removed.

    if (std::find(bw_nodes_in_expected_order.begin(), bw_nodes_in_expected_order.end(), node_ptr) !=
        bw_nodes_in_expected_order.end()) {
      nodes_in_topological_order.push_back(j);
      j++;
    }
  }

  for (size_t i = 1; i < nodes_in_topological_order.size(); ++i) {
    ASSERT_TRUE(nodes_in_topological_order[i - 1] < nodes_in_topological_order[i]);
  }
}

}  // namespace test
}  // namespace onnxruntime
