// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(REDUCED_OPS_BUILD)  // may not work with excluded op kernel implementations

#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/utils.h"
#include "core/session/inference_session.h"

#include "test/unittest_util/framework_test_utils.h"
#include "test/internal_testing_ep/internal_testing_execution_provider.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_utils.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/framework/config_options.h"
#include "core/framework/execution_providers.h"
#include "core/framework/ep_context_options.h"
#include "core/framework/fuse_nodes_funcs.h"
#include "core/framework/graph_partitioner.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/framework/resource_accountant.h"
#include "core/graph/constants.h"
#include "core/optimizer/graph_optimizer_registry.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#endif  // !defined(ORT_MINIMAL_BUILD)

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <limits>
#include <queue>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

namespace onnxruntime {
namespace test {

using namespace onnxruntime::internal_testing_ep;

// tests use onnx format models currently so exclude them from a minimal build.
// it would be possible to use ORT format models but the same partitioning code would run either way
#if !defined(ORT_MINIMAL_BUILD)

// These NHWC two-pass partitioning helpers and tests only use ONNX-domain ops
// (Conv, LogSoftmax) and core framework utilities, so they are intentionally not
// guarded by DISABLE_CONTRIB_OPS and provide regression coverage in contrib-disabled builds.
namespace {

constexpr size_t kNhwcSurvivorWorkspaceBytes = 700 * 1024;
constexpr size_t kNhwcDroppedWorkspaceBytes = 50 * 1024;
constexpr size_t kNhwcPass2OnlyWorkspaceBytes = 350 * 1024;

Level1MemoryEstimate GetNhwcAccountingTestEstimate(const std::string& op_type) {
  if (op_type == "Conv") {
    return {/*runtime_workspace_bytes=*/kNhwcSurvivorWorkspaceBytes,
            /*persistent_prepack_bytes=*/102,
            /*temporary_prepack_bytes=*/103};
  }
  if (op_type == "Relu") {
    return {/*runtime_workspace_bytes=*/kNhwcPass2OnlyWorkspaceBytes,
            /*persistent_prepack_bytes=*/302,
            /*temporary_prepack_bytes=*/303};
  }
  return {/*runtime_workspace_bytes=*/kNhwcDroppedWorkspaceBytes,
          /*persistent_prepack_bytes=*/202,
          /*temporary_prepack_bytes=*/203};
}

class TwoPassNhwcTestExecutionProvider : public IExecutionProvider {
 public:
  TwoPassNhwcTestExecutionProvider() : IExecutionProvider{"TwoPassNhwcTestExecutionProvider"} {
  }

  DataLayout GetPreferredLayout() const override {
    return DataLayout::NHWC;
  }

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const GraphViewer& graph_viewer,
                const IKernelLookup&,
                const GraphOptimizerRegistry&,
                IResourceAccountant*) const override {
    // Detect second pass by checking if any node already has our EP type assigned
    // (set during the first-pass assignment). Real NHWC EPs use this pattern to
    // recognize nodes that were transformed into kMSInternalNHWCDomain.
    bool second_pass = false;
    for (const auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
      const Node* node = graph_viewer.GetNode(node_index);
      if (node != nullptr && node->GetExecutionProviderType() == Type()) {
        second_pass = true;
        break;
      }
    }

    auto generate_metadef_name = [this, &graph_viewer]() {
      HashValue model_hash;
      const int metadef_id = metadef_id_generator_.GenerateId(graph_viewer, model_hash);
      return std::string(Type()) + "_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
    };

    std::vector<std::unique_ptr<ComputeCapability>> capabilities;
    for (const auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
      const Node* node = graph_viewer.GetNode(node_index);
      if (node == nullptr) {
        continue;
      }

      const bool is_conv = node->OpType() == "Conv";
      const bool is_log_softmax = node->OpType() == "LogSoftmax";
      if (!is_conv && !is_log_softmax) {
        continue;
      }

      if (second_pass && is_log_softmax) {
        continue;
      }

      const auto& assigned_ep = node->GetExecutionProviderType();
      if (!assigned_ep.empty() && assigned_ep != Type()) {
        continue;
      }

      capabilities.push_back(utils::MakeComputeCapability(graph_viewer,
                                                          std::vector<const Node*>{node},
                                                          generate_metadef_name,
                                                          Type(),
                                                          false));
    }

    return capabilities;
  }

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override {
    for (size_t i = 0; i < fused_nodes.size(); ++i) {
      NodeComputeInfo compute_info;
      compute_info.create_state_func = [](ComputeContext*, FunctionState*) { return 0; };
      compute_info.release_state_func = [](FunctionState) {};
      compute_info.compute_func = [](FunctionState, const OrtApi*, OrtKernelContext*) {
        return Status::OK();
      };
      node_compute_funcs.push_back(std::move(compute_info));
    }

    return Status::OK();
  }

 private:
  mutable ModelMetadefIdGenerator metadef_id_generator_;
};

// Variant of the two-pass NHWC EP used to validate that the resource accountant
// is updated correctly across the NHWC two-pass partitioning flow.
//
// It reports kCudaExecutionProvider as its type so that the SizeBasedResourceAccountant
// (which CreateAccountants registers under kCudaExecutionProvider) is wired to it,
// mirroring the real in-tree CUDA EP. Like the CUDA EP, it attaches accounting costs
// only to newly claimed capabilities. Second-pass survivors are already tagged with
// this EP and carry no new cost; they rely on the partitioner's provisional reservation.
// Dropped nodes therefore never leak budget.
class AccountingNhwcTestExecutionProvider : public IExecutionProvider {
 public:
  explicit AccountingNhwcTestExecutionProvider(
      std::optional<NodeIndex> unassignable_capability_node_index = std::nullopt)
      : IExecutionProvider{kCudaExecutionProvider},
        unassignable_capability_node_index_(unassignable_capability_node_index) {
  }

  DataLayout GetPreferredLayout() const override {
    return DataLayout::NHWC;
  }

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const GraphViewer& graph_viewer,
                const IKernelLookup&,
                const GraphOptimizerRegistry&,
                IResourceAccountant* resource_accountant) const override {
    if (resource_accountant != nullptr) {
      observed_accountant_ = resource_accountant;
    }

    bool second_pass = false;
    for (const auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
      const Node* node = graph_viewer.GetNode(node_index);
      if (node != nullptr && node->GetExecutionProviderType() == Type()) {
        second_pass = true;
        break;
      }
    }

    auto generate_metadef_name = [this, &graph_viewer]() {
      HashValue model_hash;
      const int metadef_id = metadef_id_generator_.GenerateId(graph_viewer, model_hash);
      return std::string(Type()) + "_" + std::to_string(model_hash) + "_" + std::to_string(metadef_id);
    };

    std::vector<std::unique_ptr<ComputeCapability>> capabilities;
    size_t consumed_memory = 0;
    size_t memory_threshold = std::numeric_limits<size_t>::max();
    if (resource_accountant != nullptr) {
      consumed_memory = std::get<size_t>(resource_accountant->GetConsumedAmount());
      if (const auto threshold = resource_accountant->GetThreshold(); threshold.has_value()) {
        memory_threshold = std::get<size_t>(*threshold);
      }
    }

    for (const auto node_index : graph_viewer.GetNodesInTopologicalOrder()) {
      const Node* node = graph_viewer.GetNode(node_index);
      if (node == nullptr) {
        continue;
      }

      const bool is_conv = node->OpType() == "Conv";
      const bool is_log_softmax = node->OpType() == "LogSoftmax";
      const bool is_relu = node->OpType() == "Relu";
      if (!is_conv && !is_log_softmax && !is_relu) {
        continue;
      }

      // Drop LogSoftmax on the second pass to model the EP releasing a node that
      // it tentatively claimed on the first pass. Relu models a node that becomes
      // claimable only in pass 2.
      if ((second_pass && is_log_softmax) || (!second_pass && is_relu)) {
        continue;
      }

      const auto& assigned_ep = node->GetExecutionProviderType();
      if (!assigned_ep.empty() && assigned_ep != Type()) {
        continue;
      }

      const bool already_claimed = (assigned_ep == Type());

      if (unassignable_capability_node_index_.has_value() && !second_pass && is_log_softmax) {
        auto sub_graph = std::make_unique<IndexedSubGraph>();
        sub_graph->SetAccountant(resource_accountant);
        sub_graph->nodes.push_back(node->Index());
        sub_graph->nodes.push_back(*unassignable_capability_node_index_);
        sub_graph->AppendNodeCost(resource_accountant->ComputeResourceCount(
            *node, GetNhwcAccountingTestEstimate(node->OpType())));
        sub_graph->AppendNodeCost(ResourceCount{size_t{0}});
        capabilities.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
        continue;
      }

      auto capability = utils::MakeComputeCapability(graph_viewer,
                                                     std::vector<const Node*>{node},
                                                     generate_metadef_name,
                                                     Type(),
                                                     false);

      // Mirror the in-tree CUDA EP: only newly-claimed (first-pass) capabilities carry
      // accounting costs. Already-tagged second-pass survivors take the "previously
      // assigned" branch and carry no cost.
      if (resource_accountant != nullptr && !already_claimed) {
        capability->sub_graph->SetAccountant(resource_accountant);
        for (auto cost_node_index : capability->sub_graph->nodes) {
          const Node* cost_node = graph_viewer.GetNode(cost_node_index);
          const ResourceCount cost = resource_accountant->ComputeResourceCount(
              *cost_node, GetNhwcAccountingTestEstimate(cost_node->OpType()));
          const size_t cost_bytes = std::get<size_t>(cost);
          if (consumed_memory > memory_threshold ||
              cost_bytes > memory_threshold - consumed_memory) {
            resource_accountant->SetStopAssignment();
            capability.reset();
            break;
          }
          consumed_memory += cost_bytes;
          capability->sub_graph->AppendNodeCost(cost);
        }
      }

      if (capability != nullptr) {
        capabilities.push_back(std::move(capability));
      } else {
        break;
      }
    }

    return capabilities;
  }

  Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                 std::vector<NodeComputeInfo>& node_compute_funcs) override {
    for (size_t i = 0; i < fused_nodes.size(); ++i) {
      NodeComputeInfo compute_info;
      compute_info.create_state_func = [](ComputeContext*, FunctionState*) { return 0; };
      compute_info.release_state_func = [](FunctionState) {};
      compute_info.compute_func = [](FunctionState, const OrtApi*, OrtKernelContext*) {
        return Status::OK();
      };
      node_compute_funcs.push_back(std::move(compute_info));
    }

    return Status::OK();
  }

  IResourceAccountant* observed_accountant() const {
    return observed_accountant_;
  }

 private:
  mutable ModelMetadefIdGenerator metadef_id_generator_;
  mutable IResourceAccountant* observed_accountant_ = nullptr;
  std::optional<NodeIndex> unassignable_capability_node_index_;
};

}  // namespace

#define ORT_MODEL_FOLDER ORT_TSTR("testdata/")

auto RunTest(const std::string& op, const ORTCHAR_T* model_path) {
  SessionOptions so;
  auto session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

  const std::unordered_set<std::string> supported_ops{op};

  ASSERT_STATUS_OK(session->RegisterExecutionProvider(
      std::make_unique<InternalTestingExecutionProvider>(supported_ops)));

  ASSERT_STATUS_OK(session->Load(model_path));
  const auto& graph = session->GetGraph();
  GraphViewer viewer{graph};

  ASSERT_STATUS_OK(session->Initialize());

  auto& func_mgr = const_cast<SessionState&>(session->GetSessionState()).GetMutableFuncMgr();
  const NodeComputeInfo* compute_func = nullptr;

  int num_partitions{0}, num_other_nodes{0};

  for (const auto& node : graph.Nodes()) {
    EXPECT_EQ(supported_ops.count(node.OpType()), size_t(0))
        << "Nodes with supported op types should have been replaced. Node with type " << node.OpType() << " was not.";
    if (node.GetExecutionProviderType() == kInternalTestingExecutionProvider) {
      EXPECT_STATUS_OK(func_mgr.GetFuncs(node.Name(), compute_func));
      EXPECT_NE(compute_func, nullptr);
      ++num_partitions;
    } else {
      ++num_other_nodes;
    }
  }

  ASSERT_EQ(num_partitions, 1) << "Partition aware topological sort should have resulted in a single partition."
                               << " Op=" << op << " Partitions=" << num_partitions
                               << " OtherNodes=" << num_other_nodes;
};

// model has an unsupported node between the supported nodes after the initial topo sort.
// the partition aware topo sort should result in the unsupported node moving to earlier in the order,
// and allow a single partition of supported nodes to be created.
TEST(InternalTestingEP, TestSortResultsInSinglePartition) {
  // see testdata/ep_partitioning_tests.py for model description
  // There should be only one partition, regardless of whether Add or Sub is supported by the EP.
  const ORTCHAR_T* model_path = ORT_TSTR("testdata/ep_partitioning_test_1.onnx");
  RunTest("Add", model_path);
  RunTest("Sub", model_path);
}

// mode has Resize op with optional input roi which is just a placeholder.
// partition function should skip the placeholder inputs.
TEST(InternalTestingEP, TestResizeWithOptionalInput) {
  // Resize op has optional input roi which is just a placeholder
  const ORTCHAR_T* model_path = ORT_TSTR("testdata/model_resize_empty_optional_input.onnx");
  RunTest("Resize", model_path);
}

// Test that when doing the partition aware sort and selecting groups that input dependencies are correctly handled
TEST(InternalTestingEP, TestDependenciesCorrectlyHandled) {
  SessionOptions so;
  auto session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

  const std::unordered_set<std::string> supported_ops{"Add"};

  ASSERT_STATUS_OK(session->RegisterExecutionProvider(
      std::make_unique<InternalTestingExecutionProvider>(supported_ops)));

  const ORTCHAR_T* model_path = ORT_MODEL_FOLDER "ep_partitioning_test_2.onnx";
  ASSERT_STATUS_OK(session->Load(model_path));
  const auto& graph = session->GetGraph();
  GraphViewer viewer{graph};

  ASSERT_STATUS_OK(session->Initialize());  // this should fail if we don't process dependencies correctly

  auto& func_mgr = const_cast<SessionState&>(session->GetSessionState()).GetMutableFuncMgr();
  const NodeComputeInfo* compute_func = nullptr;

  int num_partitions{0};
  int num_other_nodes{0};

  for (const auto& node : graph.Nodes()) {
    EXPECT_EQ(supported_ops.count(node.OpType()), size_t(0))
        << "Nodes with supported op types should have been replaced. Node with type " << node.OpType() << " was not.";
    if (node.GetExecutionProviderType() == kInternalTestingExecutionProvider) {
      EXPECT_STATUS_OK(func_mgr.GetFuncs(node.Name(), compute_func));
      EXPECT_NE(compute_func, nullptr);
      ++num_partitions;
    } else {
      ++num_other_nodes;
    }
  }

  ASSERT_EQ(num_partitions, 2);
  ASSERT_EQ(num_other_nodes, 2);
}

TEST(InternalTestingEP, NhwcSecondPassDropFallsBackFromCpuKernelNode) {
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}, {kMSDomain, 1}};
  Model model("NhwcSecondPassDropFallsBackFromCpuKernelNode",
              false,
              ModelMetaData(),
              PathString(),
              IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version,
              {},
              DefaultLoggingManager().DefaultLogger());

  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  const std::vector<int64_t> tensor_shape{1, 1, 3, 3};
  auto* input = builder.MakeInput<float>(std::optional<std::vector<int64_t>>{tensor_shape});
  auto* weights = builder.MakeInitializer<float>(std::vector<int64_t>{1, 1, 1, 1}, std::vector<float>{1.0f});
  auto* conv_output = builder.MakeIntermediate<float>(std::optional<std::vector<int64_t>>{tensor_shape});
  auto* output = builder.MakeOutput<float>(std::optional<std::vector<int64_t>>{tensor_shape});

  builder.AddConvNode(input, weights, conv_output);
  builder.AddNode("LogSoftmax", std::vector<NodeArg*>{conv_output}, std::vector<NodeArg*>{output});
  builder.SetGraphOutputs();

  ASSERT_STATUS_OK(graph.Resolve());

  std::string model_data;
  ASSERT_TRUE(model.ToProto().SerializeToString(&model_data));

  SessionOptions so;
  auto session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());
  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::make_unique<TwoPassNhwcTestExecutionProvider>()));

  ASSERT_STATUS_OK(session->Load(model_data.data(), static_cast<int>(model_data.size())));
  ASSERT_STATUS_OK(session->Initialize());

  bool saw_log_softmax = false;
  int num_ep_nodes = 0;
  for (const auto& node : session->GetGraph().Nodes()) {
    if (node.GetExecutionProviderType() == "TwoPassNhwcTestExecutionProvider") {
      ++num_ep_nodes;
    }

    if (node.OpType() == "LogSoftmax") {
      saw_log_softmax = true;
      EXPECT_NE(node.GetExecutionProviderType(), "TwoPassNhwcTestExecutionProvider");
    }
  }

  EXPECT_GT(num_ep_nodes, 0);
  EXPECT_TRUE(saw_log_softmax);
}

// Validates that the resource accountant is updated correctly across the NHWC two-pass
// partitioning flow: a node tentatively claimed on the first pass but dropped on the
// second pass must NOT consume budget (no phantom), while a node that survives must be
// committed exactly once (no double-count). A pass-2-only node must be evaluated against
// the provisional cost of pass-1 survivors, so independently fitting passes cannot exceed
// the combined budget.
TEST(InternalTestingEP, NhwcTwoPassAccountingReservesSurvivorsDuringPass2Admission) {
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}, {kMSDomain, 1}};
  Model model("NhwcTwoPassAccountingReservesSurvivorsDuringPass2Admission",
              false,
              ModelMetaData(),
              PathString(),
              IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version,
              {},
              DefaultLoggingManager().DefaultLogger());

  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  const std::vector<int64_t> tensor_shape{1, 1, 3, 3};
  auto* input = builder.MakeInput<float>(std::optional<std::vector<int64_t>>{tensor_shape});
  auto* weights = builder.MakeInitializer<float>(std::vector<int64_t>{1, 1, 1, 1}, std::vector<float>{1.0f});
  auto* conv_output = builder.MakeIntermediate<float>(std::optional<std::vector<int64_t>>{tensor_shape});
  auto* relu_output = builder.MakeIntermediate<float>(std::optional<std::vector<int64_t>>{tensor_shape});
  auto* output = builder.MakeOutput<float>(std::optional<std::vector<int64_t>>{tensor_shape});

  builder.AddConvNode(input, weights, conv_output);
  builder.AddNode("Relu", std::vector<NodeArg*>{conv_output}, std::vector<NodeArg*>{relu_output});
  builder.AddNode("LogSoftmax", std::vector<NodeArg*>{relu_output}, std::vector<NodeArg*>{output});
  builder.SetGraphOutputs();

  ASSERT_STATUS_OK(graph.Resolve());

  // Helper to read the size_t held by a ResourceCount.
  auto get_size = [](const ResourceCount& rc) -> size_t {
    const auto* value = std::get_if<size_t>(&rc);
    EXPECT_NE(value, nullptr) << "ResourceCount does not hold size_t";
    return value != nullptr ? *value : 0;
  };

  // Build a fresh ad-hoc accountant (no stats file) via the real factory.
  auto make_accountant = [](std::optional<ResourceAccountantMap>& acc_map) -> IResourceAccountant* {
    ConfigOptions config;
    // Large memory limit so reference costs are always available; empty stats file => ad-hoc mode.
    EXPECT_STATUS_OK(config.AddConfigEntry(kOrtSessionOptionsResourceCudaPartitioningSettings, "1048576,"));
    EXPECT_STATUS_OK(CreateAccountants(config, PathString(), acc_map));
    EXPECT_TRUE(acc_map.has_value());
    auto it = acc_map->find(kCudaExecutionProvider);
    return it != acc_map->end() ? it->second.get() : nullptr;
  };

  // Reference per-node costs computed independently on fresh accountants.
  const Node* conv_node = nullptr;
  const Node* relu_node = nullptr;
  const Node* log_softmax_node = nullptr;
  for (const auto& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      conv_node = &node;
    } else if (node.OpType() == "Relu") {
      relu_node = &node;
    } else if (node.OpType() == "LogSoftmax") {
      log_softmax_node = &node;
    }
  }
  ASSERT_NE(conv_node, nullptr);
  ASSERT_NE(relu_node, nullptr);
  ASSERT_NE(log_softmax_node, nullptr);

  std::optional<ResourceAccountantMap> ref_conv_map;
  std::optional<ResourceAccountantMap> ref_relu_map;
  std::optional<ResourceAccountantMap> ref_ls_map;
  IResourceAccountant* ref_conv_acc = make_accountant(ref_conv_map);
  IResourceAccountant* ref_relu_acc = make_accountant(ref_relu_map);
  IResourceAccountant* ref_ls_acc = make_accountant(ref_ls_map);
  ASSERT_NE(ref_conv_acc, nullptr);
  ASSERT_NE(ref_relu_acc, nullptr);
  ASSERT_NE(ref_ls_acc, nullptr);
  const Level1MemoryEstimate conv_estimate = GetNhwcAccountingTestEstimate("Conv");
  const Level1MemoryEstimate relu_estimate = GetNhwcAccountingTestEstimate("Relu");
  const Level1MemoryEstimate log_softmax_estimate = GetNhwcAccountingTestEstimate("LogSoftmax");
  const size_t expected_conv_cost =
      get_size(ref_conv_acc->ComputeResourceCount(*conv_node, conv_estimate));
  const size_t expected_relu_cost =
      get_size(ref_relu_acc->ComputeResourceCount(*relu_node, relu_estimate));
  const size_t expected_log_softmax_cost =
      get_size(ref_ls_acc->ComputeResourceCount(*log_softmax_node, log_softmax_estimate));
  ASSERT_GT(expected_conv_cost, 0u);
  ASSERT_GT(expected_relu_cost, 0u);
  ASSERT_GT(expected_log_softmax_cost, 0u);
  constexpr size_t kBudgetBytes = 1000 * 1024;
  ASSERT_LT(expected_conv_cost + expected_log_softmax_cost, kBudgetBytes);
  ASSERT_LT(expected_relu_cost, kBudgetBytes);
  ASSERT_GT(expected_conv_cost + expected_relu_cost, kBudgetBytes);

  // Drive partitioning directly with the accounting-aware NHWC EP. The accountant is
  // created internally by GraphPartitioner from the config option below (keyed to
  // kCudaExecutionProvider, which matches the EP's type).
  ExecutionProviders execution_providers;
  auto& default_logger = DefaultLoggingManager().DefaultLogger();
  auto ep = std::make_unique<AccountingNhwcTestExecutionProvider>();
  auto* ep_raw = ep.get();
  ep->SetLogger(&default_logger);
  ASSERT_STATUS_OK(execution_providers.Add(kCudaExecutionProvider, std::move(ep)));

  KernelRegistryManager krm;
  ASSERT_STATUS_OK(krm.RegisterKernels(execution_providers));

  SessionOptions sess_options;
  ASSERT_STATUS_OK(sess_options.config_options.AddConfigEntry(
      kOrtSessionOptionsResourceCudaPartitioningSettings, "1000,"));

  // Capture the accountant's consumed amount when the survivor partition is assigned.
  // on_partition_assignment_fn runs after GetCapabilityForEP completes (and therefore
  // after the deferred commit), but before PlaceNode adds any further cost.
  std::optional<size_t> observed_consumed;
  std::optional<size_t> observed_workspace;
  std::optional<size_t> observed_persistent_prepack;
  std::optional<size_t> observed_temporary_prepack;
  std::optional<WorkspaceEstimateSourceCounts> observed_source_counts;
  bool pass2_only_relu_assigned = false;
  OnPartitionAssignmentFunction on_assignment =
      [&](const Graph& assignment_graph, const ComputeCapability& capability,
          const std::string& assigned_ep_type) {
        if (assigned_ep_type == kCudaExecutionProvider && ep_raw->observed_accountant() != nullptr) {
          for (NodeIndex node_index : capability.sub_graph->nodes) {
            const Node* assigned_node = assignment_graph.GetNode(node_index);
            if (assigned_node != nullptr && assigned_node->OpType() == "Relu") {
              pass2_only_relu_assigned = true;
            }
          }
          observed_consumed = get_size(ep_raw->observed_accountant()->GetConsumedAmount());
          observed_workspace = ep_raw->observed_accountant()->GetCommittedWorkspaceEstimate();
          observed_persistent_prepack =
              ep_raw->observed_accountant()->GetCommittedPersistentPrepackEstimate();
          observed_temporary_prepack =
              ep_raw->observed_accountant()->GetCommittedTemporaryPrepackEstimate();
          observed_source_counts =
              ep_raw->observed_accountant()->GetWorkspaceEstimateSourceCounts();
        }
      };

  auto graph_optimizer_registry = std::make_unique<GraphOptimizerRegistry>(
      &sess_options, nullptr /*cpu_ep*/, &default_logger);

  GraphPartitioner partitioner(krm, execution_providers, std::move(graph_optimizer_registry), []() -> bool { return false; }, on_assignment);

  layout_transformation::TransformLayoutFunction transform_layout_fn =
      [](Graph&, bool& modified, const IExecutionProvider&,
         const layout_transformation::DebugGraphFn&) -> Status {
    modified = false;
    return Status::OK();
  };
  layout_transformation::DebugGraphFn debug_graph_fn;

  FuncManager func_mgr;
  ASSERT_STATUS_OK(
      partitioner.Partition(graph, func_mgr, transform_layout_fn,
                            sess_options.config_options, default_logger, nullptr /*layering_index*/,
                            GraphPartitioner::Mode::kNormal,
                            epctx::ModelGenOptions{},
                            debug_graph_fn));

  ASSERT_TRUE(observed_consumed.has_value())
      << "Expected the surviving Conv partition to be assigned to the EP.";
  // Conv survived: committed exactly once.
  EXPECT_EQ(*observed_consumed, expected_conv_cost)
      << "Survivor Conv should be committed exactly once.";
  // LogSoftmax was dropped on the second pass: its cost must not leak (no phantom budget).
  EXPECT_NE(*observed_consumed, expected_conv_cost + expected_log_softmax_cost)
      << "Dropped LogSoftmax must not consume budget.";
  EXPECT_FALSE(pass2_only_relu_assigned)
      << "Pass-2-only Relu must be rejected because its cost plus the Conv survivor exceeds the budget.";
  ASSERT_TRUE(observed_workspace.has_value());
  ASSERT_TRUE(observed_persistent_prepack.has_value());
  ASSERT_TRUE(observed_temporary_prepack.has_value());
  ASSERT_TRUE(observed_source_counts.has_value());
  EXPECT_EQ(*observed_workspace, *conv_estimate.runtime_workspace_bytes);
  EXPECT_EQ(*observed_persistent_prepack, conv_estimate.persistent_prepack_bytes);
  EXPECT_EQ(*observed_temporary_prepack, conv_estimate.temporary_prepack_bytes);
  EXPECT_EQ(observed_source_counts->estimator, size_t{1});
  EXPECT_EQ(observed_source_counts->fallback, size_t{0});
}

TEST(InternalTestingEP, NhwcTwoPassAccountingDoesNotReserveUnassignedCapability) {
  std::unordered_map<std::string, int> domain_to_version{{kOnnxDomain, 13}, {kMSDomain, 1}};
  Model model("NhwcTwoPassAccountingDoesNotReserveUnassignedCapability",
              false,
              ModelMetaData(),
              PathString(),
              IOnnxRuntimeOpSchemaRegistryList(),
              domain_to_version,
              {},
              DefaultLoggingManager().DefaultLogger());

  Graph& graph = model.MainGraph();
  ModelTestBuilder builder(graph);

  const std::vector<int64_t> tensor_shape{1, 1, 3, 3};
  auto* input = builder.MakeInput<float>(std::optional<std::vector<int64_t>>{tensor_shape});
  auto* weights = builder.MakeInitializer<float>(std::vector<int64_t>{1, 1, 1, 1}, std::vector<float>{1.0f});
  auto* conv_output = builder.MakeIntermediate<float>(std::optional<std::vector<int64_t>>{tensor_shape});
  auto* log_softmax_output = builder.MakeIntermediate<float>(std::optional<std::vector<int64_t>>{tensor_shape});
  auto* removed_output = builder.MakeIntermediate<float>(std::optional<std::vector<int64_t>>{tensor_shape});
  auto* output = builder.MakeOutput<float>(std::optional<std::vector<int64_t>>{tensor_shape});

  builder.AddConvNode(input, weights, conv_output);
  builder.AddNode(
      "LogSoftmax", std::vector<NodeArg*>{conv_output}, std::vector<NodeArg*>{log_softmax_output});
  builder.AddNode("Relu", std::vector<NodeArg*>{log_softmax_output}, std::vector<NodeArg*>{output});
  Node& removed_node = builder.AddNode(
      "Identity", std::vector<NodeArg*>{input}, std::vector<NodeArg*>{removed_output});
  const NodeIndex removed_node_index = removed_node.Index();
  graph.RemoveNode(removed_node_index);
  builder.SetGraphOutputs();
  ASSERT_STATUS_OK(graph.Resolve());

  Node* conv_node = nullptr;
  Node* log_softmax_node = nullptr;
  Node* relu_node = nullptr;
  for (auto& node : graph.Nodes()) {
    if (node.OpType() == "Conv") {
      conv_node = &node;
    } else if (node.OpType() == "LogSoftmax") {
      log_softmax_node = &node;
    } else if (node.OpType() == "Relu") {
      relu_node = &node;
    }
  }
  ASSERT_NE(conv_node, nullptr);
  ASSERT_NE(log_softmax_node, nullptr);
  ASSERT_NE(relu_node, nullptr);

  ConfigOptions reference_config;
  ASSERT_STATUS_OK(reference_config.AddConfigEntry(
      kOrtSessionOptionsResourceCudaPartitioningSettings, "1048576,"));
  std::optional<ResourceAccountantMap> reference_accountants;
  ASSERT_STATUS_OK(CreateAccountants(reference_config, PathString(), reference_accountants));
  ASSERT_TRUE(reference_accountants.has_value());
  auto* reference_accountant = reference_accountants->at(kCudaExecutionProvider).get();
  const size_t expected_conv_cost = std::get<size_t>(reference_accountant->ComputeResourceCount(
      *conv_node, GetNhwcAccountingTestEstimate("Conv")));
  const size_t conflicting_cost = std::get<size_t>(reference_accountant->ComputeResourceCount(
      *log_softmax_node, GetNhwcAccountingTestEstimate("LogSoftmax")));
  const size_t expected_relu_cost = std::get<size_t>(reference_accountant->ComputeResourceCount(
      *relu_node, GetNhwcAccountingTestEstimate("Relu")));

  ExecutionProviders execution_providers;
  auto& default_logger = DefaultLoggingManager().DefaultLogger();
  auto ep = std::make_unique<AccountingNhwcTestExecutionProvider>(removed_node_index);
  ep->SetLogger(&default_logger);
  ASSERT_STATUS_OK(execution_providers.Add(kCudaExecutionProvider, std::move(ep)));

  KernelRegistryManager krm;
  ASSERT_STATUS_OK(krm.RegisterKernels(execution_providers));

  SessionOptions sess_options;
  const size_t memory_threshold = expected_conv_cost + expected_relu_cost;
  const std::string partitioning_settings = std::to_string(memory_threshold) + ",";
  ASSERT_STATUS_OK(sess_options.config_options.AddConfigEntry(
      kOrtSessionOptionsResourceCudaPartitioningSettings,
      partitioning_settings.c_str()));
  bool pass2_only_relu_assigned = false;
  OnPartitionAssignmentFunction on_assignment =
      [&](const Graph& assignment_graph, const ComputeCapability& capability,
          const std::string& assigned_ep_type) {
        if (assigned_ep_type != kCudaExecutionProvider) {
          return;
        }

        for (NodeIndex node_index : capability.sub_graph->nodes) {
          const Node* assigned_node = assignment_graph.GetNode(node_index);
          if (assigned_node != nullptr && assigned_node->OpType() == "Relu") {
            pass2_only_relu_assigned = true;
          }
        }
      };
  auto graph_optimizer_registry = std::make_unique<GraphOptimizerRegistry>(
      &sess_options, nullptr /*cpu_ep*/, &default_logger);
  GraphPartitioner partitioner(
      krm, execution_providers, std::move(graph_optimizer_registry),
      []() -> bool { return false; }, on_assignment);

  layout_transformation::TransformLayoutFunction transform_layout_fn =
      [](Graph&, bool& modified, const IExecutionProvider&,
         const layout_transformation::DebugGraphFn&) -> Status {
    modified = false;
    return Status::OK();
  };
  layout_transformation::DebugGraphFn debug_graph_fn;
  FuncManager func_mgr;
  ASSERT_STATUS_OK(
      partitioner.Partition(graph, func_mgr, transform_layout_fn,
                            sess_options.config_options, default_logger, nullptr /*layering_index*/,
                            GraphPartitioner::Mode::kNormal,
                            epctx::ModelGenOptions{},
                            debug_graph_fn));

  EXPECT_TRUE(pass2_only_relu_assigned)
      << "The pass-2 Relu must fit when the rejected LogSoftmax capability is not reserved.";
  EXPECT_GT(conflicting_cost, size_t{0});
}

// Infrastructure that was used to check NNAPI coverage.
// Ideally this could be updated to read the model paths, supported ops and stop ops from input files
// and provide info on the partitions so no code changes are required to investigate different scenarios.
static std::unordered_set<std::string> GetNnapiSupportedOps() {
  return std::unordered_set<std::string>{
      "Add",
      "Sub",
      "Mul",
      "Div",
      "QLinearAdd",
      "Pow",
      "Relu",
      "Transpose",
      "Reshape",
      "BatchNormalization",
      "GlobalAveragePool",
      "GlobalMaxPool",
      "AveragePool",
      "MaxPool",
      "QLinearAveragePool",
      "Conv",
      "QLinearConv",
      "Cast",
      "Softmax",
      "Identity",
      "Gemm",
      "MatMul",
      "QLinearMatMul",
      "Abs",
      "Exp",
      "Floor",
      "Log",
      "Sigmoid",
      "Neg",
      "Sin",
      "Sqrt",
      "Tanh",
      "QLinearSigmoid",
      "Concat",
      "Squeeze",
      "QuantizeLinear",
      "DequantizeLinear",
      "LRN",
      "Clip",
      "Resize",
      "Flatten",
      "Min",
      "Max"};
}

struct PartitionStats {
  int num_nodes_handled;
  int num_nodes_not_handled;
  int num_compiled_nodes;
};

static void TestNnapiPartitioning(const std::string& test_name, const std::string& model_uri,
                                  bool optimize, bool debug_output,
                                  const std::unordered_set<std::string>& stop_ops,
                                  const std::vector<std::string>& additional_supported_ops,
                                  PartitionStats& stats) {
  SessionOptions so;
  so.graph_optimization_level = optimize ? TransformerLevel::Level3 : TransformerLevel::Level1;

  auto session = std::make_unique<InferenceSessionWrapper>(so, GetEnvironment());

  // we disable NCHWc in mobile scenarios as it's not relevant to ARM
  if (optimize) {
    ASSERT_STATUS_OK(session->FilterEnabledOptimizers({"NchwcTransformer"}));
  }

  auto ops = GetNnapiSupportedOps();
  for (const auto& op_type : additional_supported_ops) {
    ops.insert(op_type);
  }

  auto ep = std::make_unique<InternalTestingExecutionProvider>(ops, stop_ops, DataLayout::NHWC);
  if (debug_output) {
    ep->SetDebugOutput(true);
  }

  ASSERT_STATUS_OK(session->RegisterExecutionProvider(std::move(ep)));

  ASSERT_STATUS_OK(session->Load(model_uri));
  const auto& graph = session->GetGraph();
  GraphViewer viewer{graph};

  // save node count before optimization/partitioning. we lose some accuracy if optimizations replace nodes
  const auto num_nodes = graph.NumberOfNodes();

  ASSERT_STATUS_OK(session->Initialize());

  // log the unsupported ops after initializer so that anything removed by constant folding etc. isn't listed.
  std::unordered_map<std::string, int> unsupported_ops;
  std::ostringstream oss;
  std::string unsupported_op_str;

  for (const Node& node : graph.Nodes()) {
    if (node.GetExecutionProviderType() != kInternalTestingExecutionProvider &&
        ops.count(node.OpType()) == 0) {
      auto entry = unsupported_ops.find(node.OpType());
      if (entry != unsupported_ops.end()) {
        ++entry->second;
      } else {
        unsupported_ops[node.OpType()] = 1;
      }
    }
  }

  if (!unsupported_ops.empty()) {
    bool add_comma = false;
    for (const auto& pair : unsupported_ops) {
      if (add_comma) {
        oss << ",";
      }

      oss << pair.first << "(" << pair.second << ")";
      add_comma = true;
    }

    unsupported_op_str = oss.str();
  }

  auto& func_mgr = const_cast<SessionState&>(session->GetSessionState()).GetMutableFuncMgr();
  const NodeComputeInfo* compute_func = nullptr;

  stats.num_nodes_not_handled = 0;
  stats.num_compiled_nodes = 0;

  // find the nodes downstream of the excluded nodes to check that they were assigned correctly
  std::unordered_set<const Node*> excluded_nodes;
  if (!stop_ops.empty()) {
    for (const auto& node : graph.Nodes()) {
      if (stop_ops.find(node.OpType()) != stop_ops.cend()) {
        excluded_nodes.insert(&node);

        // add all the downstream nodes to the excluded list
        std::queue<const Node*> nodes_to_add;
        nodes_to_add.push(&node);
        while (!nodes_to_add.empty()) {
          const Node* cur_node = nodes_to_add.front();
          nodes_to_add.pop();

          std::for_each(cur_node->OutputNodesBegin(), cur_node->OutputNodesEnd(),
                        [&nodes_to_add, &excluded_nodes](const Node& output_node) {
                          nodes_to_add.push(&output_node);
                          excluded_nodes.insert(&output_node);
                        });
        }
      }
    }
  }

  for (const auto& node : graph.Nodes()) {
    if (stop_ops.empty() || excluded_nodes.find(&node) == excluded_nodes.cend()) {
      EXPECT_EQ(ops.count(node.OpType()), size_t(0))
          << "Nodes with supported op types should have been replaced. Node with type "
          << node.OpType() << " was not.";
    } else {
      EXPECT_NE(node.GetExecutionProviderType(), kInternalTestingExecutionProvider)
          << "Node is downstream from a 'stop at' node and should not have been taken. Node:"
          << node.Name();
    }

    if (node.GetExecutionProviderType() == kInternalTestingExecutionProvider) {
      EXPECT_STATUS_OK(func_mgr.GetFuncs(node.Name(), compute_func));
      EXPECT_NE(compute_func, nullptr);
      ++stats.num_compiled_nodes;
    } else {
      ++stats.num_nodes_not_handled;
    }
  }

  stats.num_nodes_handled = num_nodes - stats.num_nodes_not_handled;

  auto pad_str = [](std::string const& str, size_t len = 10) {
    return (str.size() < len) ? str + std::string(len - str.size(), ' ') : str;
  };

  std::cout << pad_str(test_name, 25)
            << ": Compiled=" << stats.num_compiled_nodes
            << " Handled=" << stats.num_nodes_handled
            << " NotHandled=" << stats.num_nodes_not_handled
            << " UnsupportedOps=" << unsupported_op_str
            << "\n";
}

// DISABLED - manually update model_paths and enable to test coverage as needed
TEST(InternalTestingEP, DISABLED_TestNnapiPartitioningMlPerfModels) {
  const auto supported_ops = GetNnapiSupportedOps();

  // list the models you want to test here
  std::vector<std::string> model_paths = {
      "deeplabv3_mnv2_ade20k_float.onnx",
      "mobilebert.onnx",
      "mobiledet.onnx",

  };

  for (const auto& model_uri : model_paths) {
    auto run_tests = [&](bool optimize) {
      std::cout << "\n================================\n";
      std::cout << "Model: " << model_uri;
      if (optimize) {
        std::cout << " (optimized)";
      }
      std::cout << std::endl;

      constexpr bool debug_output = false;
      PartitionStats stats{}, stop_at_nms_stats{}, slice_stats{};

      // arbitrary examples of running different combinations to test what partitioning results
      TestNnapiPartitioning("Base", model_uri, optimize, debug_output,
                            {}, {}, stats);

      TestNnapiPartitioning("StopAt[NMS]", model_uri, optimize, debug_output,
                            {"NonMaxSuppression"}, {}, stop_at_nms_stats);

      TestNnapiPartitioning("ExtraOps[Slice]", model_uri, optimize, debug_output,
                            {}, {"Slice"}, slice_stats);
    };

    run_tests(false);
    // run_tests(true);  // optimized - if models have already be optimized this isn't helpful
  }
}

#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(REDUCED_OPS_BUILD)
