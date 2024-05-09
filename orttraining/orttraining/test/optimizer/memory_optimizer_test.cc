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
#include "test/optimizer/graph_transform_test_builder.h"
#include "core/optimizer/utils.h"
#include "core/platform/env.h"
#include "core/session/inference_session.h"
#include "core/util/math.h"
#include "test/framework/test_utils.h"
#include "test/capturing_sink.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/temp_dir.h"
#include "orttraining/core/optimizer/memory_optimizer/common.h"
#include "orttraining/core/optimizer/memory_optimizer/memory_optimizer.h"
#include "orttraining/core/optimizer/memory_optimizer/memory_insight.h"
#include "orttraining/core/optimizer/memory_optimizer/transformer_specific.h"

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

  onnxruntime::GraphTransformerManager graph_transformation_mgr{1};

  const std::string alleviation_config("Gelu+:1:-1");
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("memory_optimizer_test_tmp_dir")};
  PathString config_path{ConcatPathComponent(tmp_dir.Path(),
                                             ORT_TSTR("gelurecompute.json"))};
  const std::string config_path_str = ToUTF8String(config_path);
  std::ofstream outfile(config_path_str);
  outfile << "[\"" << alleviation_config << "\"]" << std::endl;
  outfile.close();

  const std::string probe_config("1:0");
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MemoryOptimizer>(config_path_str, probe_config), TransformerLevel::Level3));

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
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("memory_optimizer_test_tmp_dir")};
  PathString config_path{ConcatPathComponent(tmp_dir.Path(),
                                             ORT_TSTR("tilerecompute.json"))};
  const std::string config_path_str = ToUTF8String(config_path);
  std::ofstream outfile(config_path_str);
  outfile << "[\"" << alleviation_config << "\"]" << std::endl;
  outfile.close();

  const std::string probe_config("1:0");
  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MemoryOptimizer>(config_path_str, probe_config), TransformerLevel::Level3));

  ASSERT_STATUS_OK(graph_transformation_mgr.ApplyTransformers(graph, TransformerLevel::Level3, *logger));

  op_to_count = CountOpsInGraph(graph);
  ASSERT_EQ(op_to_count["Tile"], 2);
  ASSERT_TRUE(op_to_count["com.microsoft.YieldOp"] == 1);
  ASSERT_TRUE(op_to_count["com.microsoft.FusedMatMul"] == 3);

  Node* recompute_tile_node{nullptr};
  Node* original_tile_node{nullptr};
  for (auto& node : graph.Nodes()) {
    if (node.OpType().compare("Tile") == 0) {
      // if name ends with _recompute, it's the recomputed node
      if (node.Name().find("_recompute") != std::string::npos) {
        recompute_tile_node = &node;
      } else {
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
}

TEST(MemoryOptimizerTests, TransformerPerLayerRecompute) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "3layer_bloom_optimized_training.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();

  // Find all optimizable subgraphs
  GraphViewer graph_viewer(graph);
  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("memory_optimizer_test_tmp_dir")};
  PathString config_path1{ConcatPathComponent(tmp_dir.Path(),
                                              ORT_TSTR("layerrecompute_initial.json"))};
  const std::string config_path1_str = ToUTF8String(config_path1);
  std::ofstream conf_stream(config_path1_str);
  conf_stream << "[]" << std::endl;  // Empty config.
  conf_stream.close();

  const std::string probe_config("1:1");
  std::map<std::string, std::pair<std::string, int>>
      cluster_id_combinations_to_saved_symbolic_byte_map;
  std::string record_str =
      optimizer::memory_optimizer::GetSerializedORTModuleMemoryStat(graph_viewer,
                                                                    config_path1_str,
                                                                    probe_config,
                                                                    true, /*enable this for test converage*/
                                                                    *logger,
                                                                    cluster_id_combinations_to_saved_symbolic_byte_map,
                                                                    nullptr,
                                                                    nullptr);

  InlinedHashMap<std::string, optimizer::memory_optimizer::UserConfig> cluster_id_to_config_map;
  PathString config_path2{ConcatPathComponent(tmp_dir.Path(),
                                              ORT_TSTR("layerrecompute_2.json"))};
  const std::string config_path2_str = ToUTF8String(config_path2);
  std::ofstream conf_stream2(config_path2_str);
  conf_stream2 << "[" << std::endl;  // Empty config.
  int index = 0;
  for (auto it = cluster_id_combinations_to_saved_symbolic_byte_map.begin();
       it != cluster_id_combinations_to_saved_symbolic_byte_map.end(); ++it) {
    std::string cluster_id = it->first;
    conf_stream2 << (index == 0 ? "" : ",") << "\"" << it->first << "\"";
    index += 1;
  }
  conf_stream2 << "]" << std::endl;
  conf_stream2.close();

  ORT_ENFORCE(optimizer::memory_optimizer::ParseOptimizationConfigFromString(config_path2_str, cluster_id_to_config_map)
                  .IsOK());
  std::ostringstream oss;
  index = 0;
  oss << "[";
  for (auto it = cluster_id_to_config_map.begin(); it != cluster_id_to_config_map.end(); ++it) {
    if (it->second.type == optimizer::memory_optimizer::OptimizationType::Recompute) {
      oss << (index == 0 ? "" : ",") << "\"" << it->first << ":1:-1\"";
      ++index;
    }
  }
  oss << "]";

  // Apply the transformer
  GraphTransformerManager graph_transformation_mgr{5};
  PathString config_path{ConcatPathComponent(tmp_dir.Path(),
                                             ORT_TSTR("layerrecompute.json"))};
  const std::string config_path_str = ToUTF8String(config_path);
  std::ofstream outfile(config_path_str);
  outfile << oss.str() << std::endl;
  outfile.close();

  ASSERT_STATUS_OK(graph_transformation_mgr.Register(
      std::make_unique<MemoryOptimizer>(config_path_str, probe_config), TransformerLevel::Level3));

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
            recompute_ln_node_parent_add_or_ln_node = graph.GetProducerNode(consumer->InputDefs()[0]->Name());
            ASSERT_TRUE(recompute_ln_node_parent_add_or_ln_node != nullptr);
            ASSERT_TRUE(recompute_ln_node_parent_add_or_ln_node->Name().find("_recompute") == std::string::npos);
          } else {
            original_ln_node = consumer;
            original_ln_node_parent_add_or_ln_node = graph.GetProducerNode(consumer->InputDefs()[0]->Name());
            ASSERT_TRUE(original_ln_node_parent_add_or_ln_node);
            ASSERT_TRUE(original_ln_node_parent_add_or_ln_node->Name().find("_recompute") == std::string::npos);
          }
        } else if (consumer->OpType().compare("LayerNormalizationGrad") == 0) {
          input_layer_norm_grad_node = consumer;
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
            recompute_ln_node_parent_add_node = graph.GetProducerNode(consumer->InputDefs()[0]->Name());
            ASSERT_TRUE(recompute_ln_node_parent_add_node);
            ASSERT_EQ(recompute_ln_node_parent_add_node->OpType(), "Add");
            ASSERT_TRUE(recompute_ln_node_parent_add_node->Name().find("_recompute") != std::string::npos);
          } else {
            original_ln_node = consumer;
            original_ln_node_parent_add_node = graph.GetProducerNode(consumer->InputDefs()[0]->Name());
            ASSERT_TRUE(original_ln_node_parent_add_node);
          }
        } else if (consumer->OpType().compare("LayerNormalizationGrad") == 0) {
          ln_grad_node = consumer;
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
  const auto& node_topology_list =
      graph_viewer.GetNodesInTopologicalOrder(optimizer::memory_optimizer::TOPOLOGICAL_SORT_ALGORITHM);

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

TEST(MemoryOptimizerTests, TransformerLayerDetectionTest) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto model_uri = MODEL_FOLDER "3layer_bloom_optimized_training.onnx";
  std::shared_ptr<Model> model;
  ASSERT_STATUS_OK(Model::Load(model_uri, model, nullptr, *logger));
  Graph& graph = model->MainGraph();
  GraphViewer graph_viewer(graph);

  InlinedHashMap<NodeIndex, ptrdiff_t> node_index_to_its_order_in_topological_sort_map;
  const auto& node_ids =
      graph_viewer.GetNodesInTopologicalOrder(optimizer::memory_optimizer::TOPOLOGICAL_SORT_ALGORITHM);

  // Find boundary ops between forward and backward pass, currently, it's limited to YieldOp.
  ptrdiff_t yield_op_order_in_topological_sort = -1;
  for (size_t i = 0; i < node_ids.size(); ++i) {
    const Node* p_node = graph_viewer.GetNode(node_ids[i]);
    if (p_node == nullptr) { /* skip removed nodes*/
      continue;
    }

    if (p_node->OpType() == "YieldOp") {
      // There are multiple YieldOps in the graph。
      ASSERT_EQ(yield_op_order_in_topological_sort, -1);
      yield_op_order_in_topological_sort = static_cast<ptrdiff_t>(i);
    }

    node_index_to_its_order_in_topological_sort_map[p_node->Index()] = static_cast<ptrdiff_t>(i);
  }

  InlinedVector<const Node*> layer_boundary_ln_node;
  optimizer::memory_optimizer::FindLayerBoundaryLayerNormNodes(graph_viewer, *logger,
                                                               node_index_to_its_order_in_topological_sort_map,
                                                               yield_op_order_in_topological_sort,
                                                               layer_boundary_ln_node);

  ASSERT_EQ(layer_boundary_ln_node.size(), 3);
  ASSERT_EQ(layer_boundary_ln_node[0]->Name(), "LayerNormalization_token_0");
  ASSERT_EQ(layer_boundary_ln_node[1]->Name(), "LayerNormalization_token_6");
  ASSERT_EQ(layer_boundary_ln_node[2]->Name(), "LayerNormalization_token_12");
}

/*
Test graph looks like as below.
           graph input [1, 1, 256, 256] (float)
                 |
             PythonOp
            /    |
           /     |  256 (float)
          / ...  | /
         /      Add
         |       |
         |       |
         | ... YieldOp
         |       |
         \       |
          \   Identity
           \     |
          PythonOpGrad
                 |
          graph output

graph out [1, 1, 256, 256] (float)

Be noted:
 the Add's input initializer 256 is a scalar float;

After enabling recompute, PythonOp will be recomputed after YieldOp, and PythonOpGrad will take the output
of recomputed PythonOp as input. The graph looks as below:
           graph input [1, 1, 256, 256] (float)
                 |
             PythonOp
               / |
                 |  256 (float)
            ...  | /
                Add
                 |
                 |
           ... YieldOp
                 |
 (recomputed)    |
  PythonOp   Identity
         \       |
          PythonOpGrad
                 |
          graph output

*/
TEST(MemoryOptimizerTests, PythonOpRecompute) {
  const logging::Logger* logger = &logging::LoggingManager::DefaultLogger();
  auto pre_graph_checker = [](Graph& graph) -> Status {
    auto op_count_pre = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count_pre.size() == 5U);
    TEST_RETURN_IF_NOT(op_count_pre["com.microsoft.PythonOp"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["com.microsoft.YieldOp"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["Identity"] == 1);
    TEST_RETURN_IF_NOT(op_count_pre["com.microsoft.PythonOpGrad"] == 1);
    return Status::OK();
  };

  auto post_graph_checker = [](Graph& graph) {
    const Node* recompute_node = nullptr;
    const Node* grad_node = nullptr;
    for (auto& node : graph.Nodes()) {
      if (node.OpType().compare("PythonOp") == 0 && node.Name().find("_recompute") != std::string::npos) {
        recompute_node = &node;
      } else if (node.OpType().compare("PythonOpGrad") == 0) {
        grad_node = &node;
      }
    }

    TEST_RETURN_IF_NOT(recompute_node != nullptr);
    TEST_RETURN_IF_NOT(grad_node != nullptr);

    // The recomputed PythonOp should be consumed by PythonOpGrad
    const Node* producer_node = graph.GetProducerNode(grad_node->InputDefs()[0]->Name());
    TEST_RETURN_IF_NOT(producer_node == recompute_node);

    auto op_count = CountOpsInGraph(graph);
    TEST_RETURN_IF_NOT(op_count.size() == 5U);
    TEST_RETURN_IF_NOT(op_count["com.microsoft.PythonOp"] == 2);
    TEST_RETURN_IF_NOT(op_count["Add"] == 1);
    TEST_RETURN_IF_NOT(op_count["com.microsoft.YieldOp"] == 1);
    TEST_RETURN_IF_NOT(op_count["Identity"] == 1);
    TEST_RETURN_IF_NOT(op_count["com.microsoft.PythonOpGrad"] == 1);
    return Status::OK();
  };

  auto build_test_case = [](ModelTestBuilder& builder) {
    auto* input_arg = builder.MakeInput<float>({{1, 1, 256, 256}});
    auto* pythonop_ctx_out = builder.MakeIntermediate();
    auto* pythonop_out = builder.MakeIntermediate();

    // Set pythonop_ctx_out's shape to be []
    pythonop_ctx_out->SetShape({});
    auto* python_op_node = &builder.AddNode("PythonOp", {input_arg}, {pythonop_ctx_out, pythonop_out}, kMSDomain);
    // Add attribute func_name, give an supported function name, but it won't be aligned to the real
    // function signature.
    python_op_node->AddAttribute("func_name", "flash_attn.bert_padding.IndexFirstAxis");
    // Add attribute input_convention
    python_op_node->AddAttribute("input_convention", "d");  // d is for tensor input
    // Add attribute input_tensor_ranks as int list
    python_op_node->AddAttribute("input_tensor_ranks", std::vector<int64_t>{4});
    // Add attribute input_tensor_types as int list
    python_op_node->AddAttribute("input_tensor_types", std::vector<int64_t>{1});
    // Add attribute input_requires_grads as int list
    python_op_node->AddAttribute("input_requires_grads", std::vector<int64_t>{1});
    // Add attribute output_tensor_ranks as int list
    python_op_node->AddAttribute("output_tensor_ranks", std::vector<int64_t>{4});
    // Add attribute output_tensor_types as int list
    python_op_node->AddAttribute("output_tensor_types", std::vector<int64_t>{1});
    // Add attribute training_mode as an int
    python_op_node->AddAttribute("training_mode", static_cast<int64_t>(1));

    auto* add_out = builder.MakeIntermediate();
    builder.AddNode("Add", {pythonop_out, builder.MakeScalarInitializer<float>(256)}, {add_out});

    auto* yield_out = builder.MakeIntermediate();
    builder.AddNode("YieldOp", {add_out}, {yield_out}, kMSDomain)
        .AddAttribute("full_shape_outputs", std::vector<int64_t>{});

    auto* identity_out = builder.MakeIntermediate();
    builder.AddNode("Identity", {yield_out}, {identity_out});

    auto* pythonop_grad_out = builder.MakeOutput();
    auto* pythonop_grad_node = &builder.AddNode("PythonOpGrad",
                                                {pythonop_ctx_out, identity_out}, {pythonop_grad_out},
                                                kMSDomain);
    // Add attribute func_name, give an supported function name, but it won't be aligned to the real function signature.
    pythonop_grad_node->AddAttribute("func_name", "flash_attn.bert_padding.IndexFirstAxis");
    // Add attribute output_convention
    pythonop_grad_node->AddAttribute("output_convention", "d");  // d is for tensor input
    // Add attribute input_tensor_ranks as int list
    pythonop_grad_node->AddAttribute("input_tensor_ranks", std::vector<int64_t>{4});
    // Add attribute input_tensor_types as int list
    pythonop_grad_node->AddAttribute("input_tensor_types", std::vector<int64_t>{1});
    // Add attribute output_tensor_ranks as int list
    pythonop_grad_node->AddAttribute("output_tensor_ranks", std::vector<int64_t>{4});
    // Add attribute output_tensor_types as int list
    pythonop_grad_node->AddAttribute("output_tensor_types", std::vector<int64_t>{1});
    // Add attribute output_tensor_requires_grads as int list
    pythonop_grad_node->AddAttribute("output_tensor_requires_grads", std::vector<int64_t>{1});
  };

  onnxruntime::test::TemporaryDirectory tmp_dir{ORT_TSTR("memory_optimizer_test_tmp_dir")};
  PathString config_path{ConcatPathComponent(tmp_dir.Path(),
                                             ORT_TSTR("pythonoprecompute.json"))};
  const std::string config_file_path = ToUTF8String(config_path);
  std::ofstream outfile(config_file_path);
  outfile << "[\"PythonOp+:1:-1\"]" << std::endl;
  outfile.close();

  const std::vector<int> opsets{12, 13, 14, 15, 16, 17};  // Clip support int64_t since opset 12
  for (auto& opset_version : opsets) {
    const std::string probe_config("1:0");
    std::unique_ptr<GraphTransformer> transformer =
        std::make_unique<MemoryOptimizer>(config_file_path, probe_config);
    ASSERT_STATUS_OK(TestGraphTransformer(build_test_case, opset_version, *logger, std::move(transformer),
                                          TransformerLevel::Level1,
                                          1, pre_graph_checker, post_graph_checker));
  }
}

}  // namespace test
}  // namespace onnxruntime
