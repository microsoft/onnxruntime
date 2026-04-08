// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/resource_accountant.h"
#include "core/graph/indexed_sub_graph.h"
#include "core/graph/constants.h"
#include "core/graph/model.h"

#include "gtest/gtest.h"

#include "test/util/include/asserts.h"
#include "test/util/include/test_environment.h"

namespace onnxruntime {
namespace test {

// Test accountant mimicking SizeBasedStatsAccountant ad-hoc path:
// Uses pending/committed weight sets so that:
// - Within a GetCapability pass, shared weights are deduped
// - Across passes, only committed weights persist and pending are discarded
class TestDedupAccountant : public IResourceAccountant {
 public:
  TestDedupAccountant() = default;

  ResourceCount GetConsumedAmount() const override {
    return consumed_;
  }

  void AddConsumedAmount(const ResourceCount& amount) noexcept override {
    if (std::holds_alternative<size_t>(amount)) {
      consumed_ += std::get<size_t>(amount);
    }
  }

  void RemoveConsumedAmount(const ResourceCount& amount) noexcept override {
    if (std::holds_alternative<size_t>(amount)) {
      consumed_ -= std::get<size_t>(amount);
    }
  }

  ResourceCount ComputeResourceCount(const Node& node) override {
    const auto* graph = node.GetContainingGraph();
    if (graph == nullptr) {
      return static_cast<size_t>(0);
    }

    size_t total = 0;
    for (const auto* input_def : node.InputDefs()) {
      if (!input_def->Exists()) {
        continue;
      }
      const auto& name = input_def->Name();
      constexpr bool check_outer_scope = true;
      const auto* init = graph->GetInitializer(name, check_outer_scope);
      if (init != nullptr) {
        if (committed_weights_.count(name) > 0) {
          continue;
        }
        if (pending_weights_.count(name) > 0) {
          continue;
        }
        auto it = weight_sizes_.find(name);
        if (it != weight_sizes_.end()) {
          total += it->second;
        }
        pending_weights_.insert(name);
        pending_weights_by_node_[node.Index()].insert(name);
      }
    }
    return total;
  }

  void ResetPendingWeights() override {
    pending_weights_.clear();
    pending_weights_by_node_.clear();
  }

  void CommitWeightsForNode(NodeIndex node_index) override {
    auto it = pending_weights_by_node_.find(node_index);
    if (it != pending_weights_by_node_.end()) {
      for (const auto& name : it->second) {
        pending_weights_.erase(name);
      }
      committed_weights_.insert(it->second.begin(), it->second.end());
      pending_weights_by_node_.erase(it);
    }
  }

  void RegisterWeight(const std::string& name, size_t size) {
    weight_sizes_[name] = size;
  }

  size_t GetConsumedSizeT() const { return consumed_; }

 private:
  size_t consumed_ = 0;
  InlinedHashSet<std::string> committed_weights_;
  InlinedHashSet<std::string> pending_weights_;
  InlinedHashMap<NodeIndex, InlinedHashSet<std::string>> pending_weights_by_node_;
  InlinedHashMap<std::string, size_t> weight_sizes_;
};

// Two Add nodes that share a single initializer weight_W.
struct SharedWeightGraph {
  std::unique_ptr<Model> model;
  Graph* graph = nullptr;
  Node* node_a = nullptr;
  Node* node_b = nullptr;

  static SharedWeightGraph Create() {
    SharedWeightGraph h;
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

    auto status = h.graph->Resolve();
    ORT_ENFORCE(status.IsOK(), status.ErrorMessage());
    return h;
  }
};

// Regression: AccountForAllNodes sums pre-stored per-node costs
// that already have correct within-pass weight deduplication.
TEST(ResourceAccountantTest, AccountForAllNodes_CorrectlyUsesPreStoredCosts) {
  auto h = SharedWeightGraph::Create();
  TestDedupAccountant accountant;
  accountant.RegisterWeight("weight_W", 1000);

  IndexedSubGraph sub_graph;
  sub_graph.nodes.push_back(h.node_a->Index());
  sub_graph.nodes.push_back(h.node_b->Index());
  sub_graph.SetAccountant(&accountant);

  auto cost_a = accountant.ComputeResourceCount(*h.node_a);
  sub_graph.AppendNodeCost(cost_a);
  EXPECT_EQ(std::get<size_t>(cost_a), size_t{1000});

  auto cost_b = accountant.ComputeResourceCount(*h.node_b);
  sub_graph.AppendNodeCost(cost_b);
  EXPECT_EQ(std::get<size_t>(cost_b), size_t{0});

  ASSERT_TRUE(sub_graph.IsAccountingEnabled());
  sub_graph.AccountForAllNodes();

  EXPECT_EQ(accountant.GetConsumedSizeT(), size_t{1000})
      << "AccountForAllNodes should sum pre-stored costs (1000 + 0)";
}

// Verifies that ResetPendingWeights + re-probe produces correct results.
// After probing (which only writes to pending), resetting pending and
// re-probing should see the full weight cost again since nothing was committed.
TEST(ResourceAccountantTest, ComputeAndAccountForNode_CorrectAfterReset) {
  auto h = SharedWeightGraph::Create();
  TestDedupAccountant accountant;
  accountant.RegisterWeight("weight_W", 1000);

  // Probing pass populates pending weights
  auto cost_a = accountant.ComputeResourceCount(*h.node_a);
  EXPECT_EQ(std::get<size_t>(cost_a), size_t{1000});
  auto cost_b = accountant.ComputeResourceCount(*h.node_b);
  EXPECT_EQ(std::get<size_t>(cost_b), size_t{0});

  // Discard the pass (simulating capabilities.clear() before second GetCapability)
  accountant.ResetPendingWeights();

  // Re-probe: weight_W was never committed, so it should be counted again
  IndexedSubGraph sub_graph;
  sub_graph.nodes.push_back(h.node_a->Index());
  sub_graph.SetAccountant(&accountant);
  auto recomputed_cost = accountant.ComputeResourceCount(*h.node_a);
  sub_graph.AccountForNode(h.node_a->Index(), recomputed_cost);

  EXPECT_EQ(accountant.GetConsumedSizeT(), size_t{1000})
      << "After ResetPendingWeights, re-probe should see full weight cost";
}

// Each node has a unique initializer. AccountForAllNodes sums both.
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

  TestDedupAccountant accountant;
  accountant.RegisterWeight("weight_1", 400);
  accountant.RegisterWeight("weight_2", 600);

  IndexedSubGraph sub_graph;
  sub_graph.nodes.push_back(node1.Index());
  sub_graph.nodes.push_back(node2.Index());
  sub_graph.SetAccountant(&accountant);

  sub_graph.AppendNodeCost(accountant.ComputeResourceCount(node1));
  sub_graph.AppendNodeCost(accountant.ComputeResourceCount(node2));

  ASSERT_TRUE(sub_graph.IsAccountingEnabled());
  sub_graph.AccountForAllNodes();

  EXPECT_EQ(accountant.GetConsumedSizeT(), size_t{1000})
      << "No shared weights: should sum all costs (400 + 600)";
}

// AccountForNode per-node and AccountForAllNodes bulk produce same result.
TEST(ResourceAccountantTest, AccountForNode_MatchesAccountForAllNodes) {
  auto h = SharedWeightGraph::Create();

  // Per-node path
  TestDedupAccountant acc1;
  acc1.RegisterWeight("weight_W", 1000);
  IndexedSubGraph sub1;
  sub1.nodes.push_back(h.node_a->Index());
  sub1.nodes.push_back(h.node_b->Index());
  sub1.SetAccountant(&acc1);
  sub1.AppendNodeCost(acc1.ComputeResourceCount(*h.node_a));
  sub1.AppendNodeCost(acc1.ComputeResourceCount(*h.node_b));
  sub1.AccountForNode(0);
  sub1.AccountForNode(1);
  size_t per_node = acc1.GetConsumedSizeT();

  // Bulk path
  TestDedupAccountant acc2;
  acc2.RegisterWeight("weight_W", 1000);
  IndexedSubGraph sub2;
  sub2.nodes.push_back(h.node_a->Index());
  sub2.nodes.push_back(h.node_b->Index());
  sub2.SetAccountant(&acc2);
  sub2.AppendNodeCost(acc2.ComputeResourceCount(*h.node_a));
  sub2.AppendNodeCost(acc2.ComputeResourceCount(*h.node_b));
  sub2.AccountForAllNodes();
  size_t bulk = acc2.GetConsumedSizeT();

  EXPECT_EQ(per_node, bulk)
      << "Per-node and bulk should produce identical results";
  EXPECT_EQ(per_node, size_t{1000});
}

// Cross-subgraph dedup: EP1 commits node_A, EP2 probes node_B and
// correctly sees weight_W as already accounted.
TEST(ResourceAccountantTest, CrossSubGraph_DedupWorks) {
  auto h = SharedWeightGraph::Create();
  TestDedupAccountant accountant;
  accountant.RegisterWeight("weight_W", 1000);

  // EP1 probes and commits node_A
  IndexedSubGraph sub1;
  sub1.nodes.push_back(h.node_a->Index());
  sub1.SetAccountant(&accountant);
  sub1.AppendNodeCost(accountant.ComputeResourceCount(*h.node_a));
  sub1.AccountForNode(0);
  EXPECT_EQ(accountant.GetConsumedSizeT(), size_t{1000});

  // EP2 probes node_B: weight_W already committed
  auto cost_b = accountant.ComputeResourceCount(*h.node_b);
  EXPECT_EQ(std::get<size_t>(cost_b), size_t{0})
      << "weight_W was committed by EP1, should be deduped for EP2";

  // EP2 commits node_B with cost 0
  IndexedSubGraph sub2;
  sub2.nodes.push_back(h.node_b->Index());
  sub2.SetAccountant(&accountant);
  sub2.AppendNodeCost(cost_b);
  sub2.AccountForNode(0);

  EXPECT_EQ(accountant.GetConsumedSizeT(), size_t{1000})
      << "Total should still be 1000 - weight_W counted once across both";
}

}  // namespace test
}  // namespace onnxruntime
