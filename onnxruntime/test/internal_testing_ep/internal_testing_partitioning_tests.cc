// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(REDUCED_OPS_BUILD)  // may not work with excluded op kernel implementations

#include "core/common/logging/logging.h"
#include "core/framework/compute_capability.h"
#include "core/framework/model_metadef_id_generator.h"
#include "core/framework/utils.h"
#include "core/providers/partitioning_utils.h"
#include "core/session/inference_session.h"

#include "test/unittest_util/framework_test_utils.h"
#include "test/unittest_util/graph_transform_test_builder.h"
#include "test/internal_testing_ep/internal_testing_execution_provider.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"
#include "test/util/include/test_utils.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

#include <queue>

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

namespace onnxruntime {
namespace test {

using namespace onnxruntime::internal_testing_ep;

// tests use onnx format models currently so exclude them from a minimal build.
// it would be possible to use ORT format models but the same partitioning code would run either way
#if !defined(ORT_MINIMAL_BUILD)

#if !defined(DISABLE_CONTRIB_OPS)
namespace {

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
    ++get_capability_calls_;
    const bool second_pass = get_capability_calls_ > 1;

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
  mutable size_t get_capability_calls_{0};
  mutable ModelMetadefIdGenerator metadef_id_generator_;
};

}  // namespace
#endif  // !defined(DISABLE_CONTRIB_OPS)

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

#if !defined(DISABLE_CONTRIB_OPS)
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
#endif  // !defined(DISABLE_CONTRIB_OPS)

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
