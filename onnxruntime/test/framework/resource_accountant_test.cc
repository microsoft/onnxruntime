// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/resource_accountant.h"
#include "core/framework/config_options.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

#include "gtest/gtest.h"

#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef _WIN32
#include <process.h>
#define ORT_TEST_PID _getpid()
#else
#include <unistd.h>
#define ORT_TEST_PID getpid()
#endif

namespace onnxruntime {
namespace test {

namespace {

// Helper to extract size_t from ResourceCount variant.
// Uses std::get_if so test failures produce a clear assertion rather than std::bad_variant_access.
size_t GetSizeT(const ResourceCount& rc) {
  const auto* value = std::get_if<size_t>(&rc);
  EXPECT_NE(value, nullptr) << "ResourceCount does not hold size_t";
  return value != nullptr ? *value : 0;
}

// Helper to create a real SizeBasedStatsAccountant in ad-hoc mode (no stats file) via factory.
// Must be called from within a TEST body (or via ASSERT_NO_FATAL_FAILURE) because it uses ASSERT_*.
void CreateAdHocAccountant(
    size_t limit_kb,
    const std::filesystem::path& model_path,
    std::optional<ResourceAccountantMap>& acc_map,
    IResourceAccountant*& out) {
  ConfigOptions config;
  std::string setting = std::to_string(limit_kb) + ",";
  ASSERT_STATUS_OK(config.AddConfigEntry(
      kOrtSessionOptionsResourceCudaPartitioningSettings, setting.c_str()));
  ASSERT_STATUS_OK(CreateAccountants(config, model_path, acc_map));
  ASSERT_TRUE(acc_map.has_value());
  auto it = acc_map->find(kCudaExecutionProvider);
  ASSERT_NE(it, acc_map->end());
  out = it->second.get();
}

}  // namespace

// Two Add nodes that share a single initializer weight_W.
struct SharedWeightGraph {
  std::unique_ptr<Model> model;
  Graph* graph = nullptr;
  Node* node_a = nullptr;
  Node* node_b = nullptr;

  static void Create(SharedWeightGraph& h) {
    std::unordered_map<std::string, int> dom;
    dom[kOnnxDomain] = 12;
    h.model = std::make_unique<Model>(
        "test_model", false, ModelMetaData(), PathString(),
        IOnnxRuntimeOpSchemaRegistryList(), dom,
        std::vector<ONNX_NAMESPACE::FunctionProto>(),
        DefaultLoggingManager().DefaultLogger());
    h.graph = &h.model->MainGraph();

    ONNX_NAMESPACE::TypeProto ft;
    ft.mutable_tensor_type()->set_elem_type(
        ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    ft.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(250);

    ONNX_NAMESPACE::TensorProto wp;
    wp.set_name("weight_W");
    wp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    wp.add_dims(250);
    for (int i = 0; i < 250; ++i) {
      wp.add_float_data(0.0f);
    }
    h.graph->AddInitializedTensor(wp);

    auto* ia = &h.graph->GetOrCreateNodeArg("input_a", &ft);
    auto* ib = &h.graph->GetOrCreateNodeArg("input_b", &ft);
    auto* wa = &h.graph->GetOrCreateNodeArg("weight_W", &ft);
    auto* oa = &h.graph->GetOrCreateNodeArg("out_a", &ft);
    auto* ob = &h.graph->GetOrCreateNodeArg("out_b", &ft);

    h.node_a = &h.graph->AddNode("node_A", "Add", "A", {ia, wa}, {oa});
    h.node_b = &h.graph->AddNode("node_B", "Add", "B", {ib, wa}, {ob});

    ASSERT_STATUS_OK(h.graph->Resolve());
  }
};

// Ad-hoc path expected costs for SharedWeightGraph:
//   weight_W = 250 floats = 1000 bytes, each output = 250 floats = 1000 bytes
//   node_A: (1000 init + 1000 out) * 1.5 = 3000
//   node_B: (0 deduped + 1000 out) * 1.5 = 1500

// AccountForAllNodes sums pre-stored per-node costs
// that already have correct within-pass weight deduplication.
TEST(ResourceAccountantTest, AccountForAllNodes_CorrectlyUsesPreStoredCosts) {
  SharedWeightGraph h;
  ASSERT_NO_FATAL_FAILURE(SharedWeightGraph::Create(h));
  std::optional<ResourceAccountantMap> acc_map;
  IResourceAccountant* accountant = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateAdHocAccountant(/*limit_kb=*/100, PathString(), acc_map, accountant));

  IndexedSubGraph sub_graph;
  sub_graph.nodes.push_back(h.node_a->Index());
  sub_graph.nodes.push_back(h.node_b->Index());
  sub_graph.SetAccountant(accountant);

  auto cost_a = accountant->ComputeResourceCount(*h.node_a);
  sub_graph.AppendNodeCost(cost_a);
  EXPECT_EQ(GetSizeT(cost_a), size_t{3000});

  auto cost_b = accountant->ComputeResourceCount(*h.node_b);
  sub_graph.AppendNodeCost(cost_b);
  EXPECT_EQ(GetSizeT(cost_b), size_t{1500});

  ASSERT_TRUE(sub_graph.IsAccountingEnabled());
  sub_graph.AccountForAllNodes();

  EXPECT_EQ(GetSizeT(accountant->GetConsumedAmount()), size_t{4500})
      << "AccountForAllNodes should sum pre-stored costs (3000 + 1500)";
}

// Verifies that ResetForNewPass + re-probe produces correct results.
// After probing (which only writes to pending), resetting for a new pass and
// re-probing should see the full weight cost again since nothing was committed.
TEST(ResourceAccountantTest, ComputeAndAccountForNode_CorrectAfterReset) {
  SharedWeightGraph h;
  ASSERT_NO_FATAL_FAILURE(SharedWeightGraph::Create(h));
  std::optional<ResourceAccountantMap> acc_map;
  IResourceAccountant* accountant = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateAdHocAccountant(/*limit_kb=*/100, PathString(), acc_map, accountant));

  // Probing pass populates pending weights
  auto cost_a = accountant->ComputeResourceCount(*h.node_a);
  EXPECT_EQ(GetSizeT(cost_a), size_t{3000});
  auto cost_b = accountant->ComputeResourceCount(*h.node_b);
  EXPECT_EQ(GetSizeT(cost_b), size_t{1500});

  // Discard the pass (simulating capabilities.clear() before second GetCapability)
  accountant->ResetForNewPass();

  // Re-probe: weight_W was never committed, so it should be counted again
  IndexedSubGraph sub_graph;
  sub_graph.nodes.push_back(h.node_a->Index());
  sub_graph.SetAccountant(accountant);
  auto recomputed_cost = accountant->ComputeResourceCount(*h.node_a);
  sub_graph.AccountForNode(h.node_a->Index(), recomputed_cost);

  EXPECT_EQ(GetSizeT(accountant->GetConsumedAmount()), size_t{3000})
      << "After ResetForNewPass, re-probe should see full weight cost";
}

// ResetForNewPass clears the stop flag so a second GetCapability pass
// (e.g., after layout transformation) can run from scratch.
TEST(ResourceAccountantTest, ResetForNewPass_ClearsStopFlag) {
  std::optional<ResourceAccountantMap> acc_map;
  IResourceAccountant* accountant = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateAdHocAccountant(/*limit_kb=*/1, PathString(), acc_map, accountant));

  // Simulate first pass exhausting budget and setting stop.
  accountant->SetStopAssignment();
  EXPECT_TRUE(accountant->IsStopIssued());

  // Framework discards first-pass results and resets for second pass.
  accountant->ResetForNewPass();
  EXPECT_FALSE(accountant->IsStopIssued())
      << "ResetForNewPass should clear the stop flag for the next pass";
}

// Each node has a unique initializer. AccountForAllNodes sums both.
// weight_1 = 100 floats = 400 bytes, weight_2 = 100 floats = 400 bytes, outputs = 400 bytes each
// node1: (400 init + 400 out) * 1.5 = 1200
// node2: (400 init + 400 out) * 1.5 = 1200
TEST(ResourceAccountantTest, AccountForAllNodes_NoSharedWeights) {
  std::unordered_map<std::string, int> dom;
  dom[kOnnxDomain] = 12;
  Model model("test_model", false, ModelMetaData(), PathString(),
              IOnnxRuntimeOpSchemaRegistryList(), dom,
              std::vector<ONNX_NAMESPACE::FunctionProto>(),
              DefaultLoggingManager().DefaultLogger());
  Graph& graph = model.MainGraph();

  ONNX_NAMESPACE::TypeProto ft;
  ft.mutable_tensor_type()->set_elem_type(
      ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  ft.mutable_tensor_type()->mutable_shape()->add_dim()->set_dim_value(100);

  const char* names[] = {"weight_1", "weight_2"};
  for (const char* wn : names) {
    ONNX_NAMESPACE::TensorProto tp;
    tp.set_name(wn);
    tp.set_data_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    tp.add_dims(100);
    for (int i = 0; i < 100; ++i) {
      tp.add_float_data(0.0f);
    }
    graph.AddInitializedTensor(tp);
  }

  auto* input = &graph.GetOrCreateNodeArg("input", &ft);
  auto* w1 = &graph.GetOrCreateNodeArg("weight_1", &ft);
  auto* w2 = &graph.GetOrCreateNodeArg("weight_2", &ft);
  auto* out1 = &graph.GetOrCreateNodeArg("out1", &ft);
  auto* out2 = &graph.GetOrCreateNodeArg("out2", &ft);

  auto& node1 = graph.AddNode("n1", "Add", "", {input, w1}, {out1});
  auto& node2 = graph.AddNode("n2", "Add", "", {out1, w2}, {out2});
  ASSERT_STATUS_OK(graph.Resolve());

  std::optional<ResourceAccountantMap> acc_map;
  IResourceAccountant* accountant = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateAdHocAccountant(/*limit_kb=*/100, PathString(), acc_map, accountant));

  IndexedSubGraph sub_graph;
  sub_graph.nodes.push_back(node1.Index());
  sub_graph.nodes.push_back(node2.Index());
  sub_graph.SetAccountant(accountant);

  sub_graph.AppendNodeCost(accountant->ComputeResourceCount(node1));
  sub_graph.AppendNodeCost(accountant->ComputeResourceCount(node2));

  ASSERT_TRUE(sub_graph.IsAccountingEnabled());
  sub_graph.AccountForAllNodes();

  EXPECT_EQ(GetSizeT(accountant->GetConsumedAmount()), size_t{2400})
      << "No shared weights: should sum all costs (1200 + 1200)";
}

// AccountForNode per-node and AccountForAllNodes bulk produce same result.
TEST(ResourceAccountantTest, AccountForNode_MatchesAccountForAllNodes) {
  SharedWeightGraph h;
  ASSERT_NO_FATAL_FAILURE(SharedWeightGraph::Create(h));

  // Per-node path
  std::optional<ResourceAccountantMap> acc_map1;
  IResourceAccountant* acc1 = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateAdHocAccountant(/*limit_kb=*/100, PathString(), acc_map1, acc1));
  IndexedSubGraph sub1;
  sub1.nodes.push_back(h.node_a->Index());
  sub1.nodes.push_back(h.node_b->Index());
  sub1.SetAccountant(acc1);
  sub1.AppendNodeCost(acc1->ComputeResourceCount(*h.node_a));
  sub1.AppendNodeCost(acc1->ComputeResourceCount(*h.node_b));
  sub1.AccountForNode(0);
  sub1.AccountForNode(1);
  size_t per_node = GetSizeT(acc1->GetConsumedAmount());

  // Bulk path
  std::optional<ResourceAccountantMap> acc_map2;
  IResourceAccountant* acc2 = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateAdHocAccountant(/*limit_kb=*/100, PathString(), acc_map2, acc2));
  IndexedSubGraph sub2;
  sub2.nodes.push_back(h.node_a->Index());
  sub2.nodes.push_back(h.node_b->Index());
  sub2.SetAccountant(acc2);
  sub2.AppendNodeCost(acc2->ComputeResourceCount(*h.node_a));
  sub2.AppendNodeCost(acc2->ComputeResourceCount(*h.node_b));
  sub2.AccountForAllNodes();
  size_t bulk = GetSizeT(acc2->GetConsumedAmount());

  EXPECT_EQ(per_node, bulk)
      << "Per-node and bulk should produce identical results";
  EXPECT_EQ(per_node, size_t{4500});
}

// Cross-subgraph dedup: EP1 commits node_A, EP2 probes node_B and
// correctly sees weight_W as already accounted.
// node_A cost: 3000, node_B cost after commit: (0 + 1000) * 1.5 = 1500
TEST(ResourceAccountantTest, CrossSubGraph_DedupWorks) {
  SharedWeightGraph h;
  ASSERT_NO_FATAL_FAILURE(SharedWeightGraph::Create(h));
  std::optional<ResourceAccountantMap> acc_map;
  IResourceAccountant* accountant = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateAdHocAccountant(/*limit_kb=*/100, PathString(), acc_map, accountant));

  // EP1 probes and commits node_A
  IndexedSubGraph sub1;
  sub1.nodes.push_back(h.node_a->Index());
  sub1.SetAccountant(accountant);
  sub1.AppendNodeCost(accountant->ComputeResourceCount(*h.node_a));
  sub1.AccountForNode(0);
  accountant->CommitWeightsForNode(h.node_a->Index());
  EXPECT_EQ(GetSizeT(accountant->GetConsumedAmount()), size_t{3000});

  // Reset for new pass to simulate new GetCapability pass
  accountant->ResetForNewPass();

  // EP2 probes node_B: weight_W already committed, only output counted
  auto cost_b = accountant->ComputeResourceCount(*h.node_b);
  EXPECT_EQ(GetSizeT(cost_b), size_t{1500})
      << "weight_W was committed by EP1, only output (1000 * 1.5) counted";

  // EP2 commits node_B
  IndexedSubGraph sub2;
  sub2.nodes.push_back(h.node_b->Index());
  sub2.SetAccountant(accountant);
  sub2.AppendNodeCost(cost_b);
  sub2.AccountForNode(0);

  EXPECT_EQ(GetSizeT(accountant->GetConsumedAmount()), size_t{4500})
      << "Total should be 3000 + 1500 - weight_W initializer counted once";
}

// ---------------------------------------------------------------------------
// Stats-based path and factory tests
// ---------------------------------------------------------------------------

// RAII helper to remove a temp file on scope exit, even if a test assertion fails.
struct ScopedFileRemover {
  std::filesystem::path path;
  ~ScopedFileRemover() {
    std::error_code ec;
    std::filesystem::remove(path, ec);
  }
};

// Stats-based path: cost is sum of all NodeAllocationStats fields.
TEST(RealAccountantTest, StatsPath_ComputesCostFromStatsFile) {
  SharedWeightGraph h;
  ASSERT_NO_FATAL_FAILURE(SharedWeightGraph::Create(h));

  // Write a stats file with known costs (unique per PID to avoid parallel collisions)
  std::error_code ec;
  auto stats_dir = std::filesystem::temp_directory_path(ec);
  ASSERT_FALSE(ec) << ec.message();
  std::ostringstream fname;
  fname << "test_resource_accountant_stats_" << ORT_TEST_PID << ".csv";
  auto stats_path = stats_dir / fname.str();
  ScopedFileRemover stats_cleanup{stats_path};

  // Get the unique node names the accountant will look up
  std::string name_a = IResourceAccountant::MakeUniqueNodeName(*h.node_a);
  std::string name_b = IResourceAccountant::MakeUniqueNodeName(*h.node_b);

  {
    std::ofstream ofs(stats_path);
    ASSERT_TRUE(ofs.is_open());
    ofs << "#name,input_sizes,initializers_sizes,total_dynamic_sizes,total_temp_allocations\n";
    // input_sizes=100, initializers=200, dynamic=300, temp=400 -> total=1000
    ofs << name_a << ",100,200,300,400\n";
    // input_sizes=50, initializers=0, dynamic=150, temp=0 -> total=200
    ofs << name_b << ",50,0,150,0\n";
  }

  // Factory expects stats file relative to model_path dir
  ConfigOptions config;
  std::string setting = "500," + stats_path.filename().string();
  ASSERT_STATUS_OK(config.AddConfigEntry(
      kOrtSessionOptionsResourceCudaPartitioningSettings, setting.c_str()));

  std::optional<ResourceAccountantMap> acc_map;
  ASSERT_STATUS_OK(CreateAccountants(config, stats_dir / "dummy_model.onnx", acc_map));
  ASSERT_TRUE(acc_map.has_value());
  auto* accountant = acc_map->at(kCudaExecutionProvider).get();

  auto cost_a = accountant->ComputeResourceCount(*h.node_a);
  EXPECT_EQ(std::get<size_t>(cost_a), size_t{1000});

  auto cost_b = accountant->ComputeResourceCount(*h.node_b);
  EXPECT_EQ(std::get<size_t>(cost_b), size_t{200});

  // Threshold should be 500 KB = 512000 bytes
  auto threshold = accountant->GetThreshold();
  ASSERT_TRUE(threshold.has_value());
  EXPECT_EQ(std::get<size_t>(*threshold), size_t{500 * 1024});
}

// Stats-based path returns 0 for unknown nodes.
TEST(RealAccountantTest, StatsPath_UnknownNodeReturnsZero) {
  SharedWeightGraph h;
  ASSERT_NO_FATAL_FAILURE(SharedWeightGraph::Create(h));

  std::error_code ec;
  auto stats_dir = std::filesystem::temp_directory_path(ec);
  ASSERT_FALSE(ec) << ec.message();
  std::ostringstream fname;
  fname << "test_resource_accountant_empty_stats_" << ORT_TEST_PID << ".csv";
  auto stats_path = stats_dir / fname.str();
  ScopedFileRemover stats_cleanup{stats_path};

  {
    std::ofstream ofs(stats_path);
    ASSERT_TRUE(ofs.is_open());
    ofs << "#name,input_sizes,initializers_sizes,total_dynamic_sizes,total_temp_allocations\n";
    // No entries for our nodes
  }

  ConfigOptions config;
  std::string setting = "1000," + stats_path.filename().string();
  ASSERT_STATUS_OK(config.AddConfigEntry(
      kOrtSessionOptionsResourceCudaPartitioningSettings, setting.c_str()));

  std::optional<ResourceAccountantMap> acc_map;
  ASSERT_STATUS_OK(CreateAccountants(config, stats_dir / "dummy_model.onnx", acc_map));
  auto* accountant = acc_map->at(kCudaExecutionProvider).get();

  auto cost = accountant->ComputeResourceCount(*h.node_a);
  EXPECT_EQ(std::get<size_t>(cost), size_t{0});
}

// Factory with no limit and no stats file creates accountant with no threshold.
TEST(RealAccountantTest, Factory_NoLimitNoStats) {
  ConfigOptions config;
  ASSERT_STATUS_OK(config.AddConfigEntry(
      kOrtSessionOptionsResourceCudaPartitioningSettings, ","));

  std::optional<ResourceAccountantMap> acc_map;
  ASSERT_STATUS_OK(CreateAccountants(config, PathString(), acc_map));
  ASSERT_TRUE(acc_map.has_value());
  auto* accountant = acc_map->at(kCudaExecutionProvider).get();
  EXPECT_FALSE(accountant->GetThreshold().has_value());
}

// Factory returns empty optional when no config is set.
TEST(RealAccountantTest, Factory_NoConfigReturnsEmpty) {
  ConfigOptions config;
  std::optional<ResourceAccountantMap> acc_map;
  ASSERT_STATUS_OK(CreateAccountants(config, PathString(), acc_map));
  EXPECT_FALSE(acc_map.has_value());
}

// Factory rejects malformed config (missing comma).
TEST(RealAccountantTest, Factory_MalformedConfigReturnsError) {
  ConfigOptions config;
  ASSERT_STATUS_OK(config.AddConfigEntry(
      kOrtSessionOptionsResourceCudaPartitioningSettings, "1000"));  // missing comma

  std::optional<ResourceAccountantMap> acc_map;
  auto status = CreateAccountants(config, PathString(), acc_map);
  EXPECT_FALSE(status.IsOK());
}

}  // namespace test
}  // namespace onnxruntime
