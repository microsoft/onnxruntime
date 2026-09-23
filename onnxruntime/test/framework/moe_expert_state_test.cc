// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <sstream>
#include <thread>

#include "gtest/gtest.h"
#include "core/framework/moe_expert_state.h"
#include "core/framework/session_state.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime::test {
#if !defined(ORT_MINIMAL_BUILD)
namespace {
class MoeExpertStateTest : public testing::Test {
 protected:
  void SetUp() override {
    ONNX_NAMESPACE::ModelProto model;
    model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
    model.add_opset_import()->set_version(13);
    auto& graph = *model.mutable_graph();
    graph.set_name("counter_kernel_identity");
    auto& input = *graph.add_input();
    input.set_name("input");
    auto* type = input.mutable_type()->mutable_tensor_type();
    type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
    type->mutable_shape()->add_dim()->set_dim_value(1);
    auto& output = *graph.add_output();
    output = input;
    output.set_name("output");
    const char* names[] = {"input", "first", "second", "output"};
    for (int i = 0; i < 3; ++i) {
      auto* node = graph.add_node();
      node->set_op_type("Identity");
      node->add_input(names[i]);
      node->add_output(names[i + 1]);
    }
    SessionOptions options;
    options.graph_optimization_level = TransformerLevel::Default;
    options.intra_op_param.thread_pool_size = 1;
    session_ = std::make_unique<InferenceSessionWrapper>(options, GetEnvironment());
    const auto bytes = model.SerializeAsString();
    ASSERT_STATUS_OK(session_->Load(bytes.data(), static_cast<int>(bytes.size())));
    ASSERT_STATUS_OK(session_->Initialize());
    for (size_t i = 0; i < 3; ++i) {
      kernels_[i] = session_->GetSessionState().GetKernel(i);
      ASSERT_NE(kernels_[i], nullptr);
    }
  }

  std::unique_ptr<InferenceSessionWrapper> session_;
  const OpKernel* kernels_[3]{};
};

TEST_F(MoeExpertStateTest, KernelExpertDictionarySeparatesNodesAndSubgraphs) {
  MoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 3));
  auto* usage = state.GetUsage(kernels_[0]);
  ASSERT_NE(usage, nullptr);
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[1], "main", 1, "QMoE", 2));
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[2], "main/0/4:body", 0, "MoE", 4));
  EXPECT_EQ(state.GetUsage(kernels_[0]), usage);
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  EXPECT_EQ(state.TotalExpertCount(), 9U);
  size_t expert_id = 99;
  ASSERT_STATUS_OK(state.GetExpertId(kernels_[0], 0, expert_id));
  EXPECT_EQ(expert_id, 0U);
  ASSERT_STATUS_OK(state.GetExpertId(kernels_[1], 0, expert_id));
  EXPECT_EQ(expert_id, 3U);
  ASSERT_STATUS_OK(state.GetExpertId(kernels_[2], 0, expert_id));
  EXPECT_EQ(expert_id, 5U);
  EXPECT_FALSE(state.GetExpertId(kernels_[1], 2, expert_id).IsOK());
  EXPECT_EQ(expert_id, 5U);

  const int selected[] = {2, 0, 2, 0};
  ASSERT_STATUS_OK(usage->RecordUsage(selected));
  ASSERT_STATUS_OK(usage->RecordUsage(selected));
  const auto snapshot = state.GetSnapshot();
  EXPECT_EQ(snapshot.at({"main", 0}).counters, (InlinedVector<double>{2, 0, 2}));
  EXPECT_EQ(snapshot.at({"main", 1}).counters, (InlinedVector<double>{0, 0}));
  EXPECT_EQ(snapshot.at({"main/0/4:body", 0}).counters, (InlinedVector<double>{0, 0, 0, 0}));
  ASSERT_STATUS_OK(state.GetUsage(kernels_[2])->RecordUsage(selected));
  InlinedVector<double> counters;
  ASSERT_STATUS_OK(state.GetUsage(kernels_[2])->GetCounters(counters));
  EXPECT_EQ(counters, (InlinedVector<double>{1, 0, 1, 0}));
}

TEST_F(MoeExpertStateTest, AppliesExponentialUpdateToEveryExpert) {
  MoeExpertState state;
  ASSERT_STATUS_OK(state.SetCounterParameters(0.5, 2.0));
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 3));
  std::istringstream initial(
      "moe_expert_state 1\n"
      "\"main\" 0 MoE 0 4\n"
      "\"main\" 0 MoE 1 2\n"
      "\"main\" 0 MoE 2 1\n");
  ASSERT_STATUS_OK(state.Load(initial));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  auto* usage = state.GetUsage(kernels_[0]);
  ASSERT_NE(usage, nullptr);
  const int selected[] = {2, 0, 2};
  ASSERT_STATUS_OK(usage->RecordUsage(selected));
  EXPECT_EQ(state.GetSnapshot().at({"main", 0}).counters, (InlinedVector<double>{4, 1, 2.5}));
  ASSERT_STATUS_OK(usage->RecordUsage(selected));
  EXPECT_EQ(state.GetSnapshot().at({"main", 0}).counters, (InlinedVector<double>{4, 0.5, 3.25}));
  ASSERT_STATUS_OK(usage->RecordUsage({}));
  EXPECT_EQ(state.GetSnapshot().at({"main", 0}).counters, (InlinedVector<double>{2, 0.25, 1.625}));
}

TEST_F(MoeExpertStateTest, ValidatesCounterParameters) {
  for (const auto& [alpha, beta] : {
           std::pair{-0.1, 1.0},
           std::pair{1.1, 1.0},
           std::pair{std::numeric_limits<double>::infinity(), 1.0},
           std::pair{1.0, -0.1},
           std::pair{1.0, std::numeric_limits<double>::infinity()}}) {
    MoeExpertState state;
    EXPECT_FALSE(state.SetCounterParameters(alpha, beta).IsOK());
  }
  MoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 1));
  EXPECT_FALSE(state.SetCounterParameters(0.5, 1.0).IsOK());
}

TEST_F(MoeExpertStateTest, LoadsPartialStateAndValidatesAtomically) {
  MoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 2, "QMoE", 3));
  std::istringstream valid("moe_expert_state 1\n\"main\" 2 QMoE 1 2.5\n");
  ASSERT_STATUS_OK(state.Load(valid));
  auto* usage = state.GetUsage(kernels_[0]);
  ASSERT_NE(usage, nullptr);
  InlinedVector<double> counters;
  for (const auto* invalid : {
           "moe_expert_state 2\n",
           "moe_expert_state 1\n\"main\" 2 QMoE 0 1\n\"main\" 2 QMoE 0 2\n",
           "moe_expert_state 1\n\"main\" 2 QMoE 0 1\n\"unknown\" 2 QMoE 0 1\n",
           "moe_expert_state 1\n\"main\" 3 QMoE 0 1\n",
           "moe_expert_state 1\n\"main\" 2 MoE 0 1\n",
           "moe_expert_state 1\n\"main\" 2 QMoE -1 1\n",
           "moe_expert_state 1\n\"main\" 2 QMoE 3 1\n",
           "moe_expert_state 1\n\"main\" 2 QMoE 0 -1\n",
           "moe_expert_state 1\n\"main\" 2 QMoE 0 nan\n",
           "moe_expert_state 1\n\"main\" 2 QMoE 0 inf\n",
           "moe_expert_state 1\n\"main\" 2 QMoE 0 1 extra\n",
           "moe_expert_state 1\nincomplete\n"}) {
    SCOPED_TRACE(invalid);
    std::istringstream input(invalid);
    EXPECT_FALSE(state.Load(input).IsOK());
    ASSERT_STATUS_OK(usage->GetCounters(counters));
    EXPECT_EQ(counters, (InlinedVector<double>{0, 2.5, 0}));
  }
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  const int selected[] = {1, 1};
  ASSERT_STATUS_OK(usage->RecordUsage(selected));
  ASSERT_STATUS_OK(usage->GetCounters(counters));
  EXPECT_EQ(counters, (InlinedVector<double>{0, 3.5, 0}));
}

TEST_F(MoeExpertStateTest, RejectsInvalidRegistrationAndUpdates) {
  MoeExpertState state;
  EXPECT_FALSE(state.RegisterNode(nullptr, "main", 0, "MoE", 2).IsOK());
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 2));
  EXPECT_FALSE(state.RegisterNode(kernels_[1], "main", 0, "MoE", 2).IsOK());
  EXPECT_FALSE(state.RegisterNode(kernels_[0], "main", 1, "MoE", 2).IsOK());
  EXPECT_FALSE(state.RegisterNode(kernels_[1], "main", 1, "Add", 2).IsOK());
  EXPECT_FALSE(state.RegisterNode(kernels_[1], "main", 1, "QMoE", 0).IsOK());
  EXPECT_EQ(state.TotalExpertCount(), 2U);
  auto* usage = state.GetUsage(kernels_[0]);
  ASSERT_NE(usage, nullptr);
  EXPECT_FALSE(usage->RecordUsage({}).IsOK());
  EXPECT_FALSE(state.BeginRun().IsOK());
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  EXPECT_FALSE(state.RegisterNode(kernels_[1], "main", 1, "MoE", 2).IsOK());
  std::istringstream initial("moe_expert_state 1\n");
  EXPECT_FALSE(state.Load(initial).IsOK());
  EXPECT_FALSE(state.FinalizeInitialization().IsOK());
  const int out_of_bounds[] = {0, 2};
  const int negative[] = {0, -1};
  EXPECT_FALSE(usage->RecordUsage(out_of_bounds).IsOK());
  EXPECT_FALSE(usage->RecordUsage(negative).IsOK());
  EXPECT_EQ(state.GetUsage(kernels_[1]), nullptr);
  EXPECT_EQ(state.GetUsage(nullptr), nullptr);
  EXPECT_EQ(state.GetSnapshot().at({"main", 0}).counters, (InlinedVector<double>{0, 0}));
}

TEST_F(MoeExpertStateTest, OverflowDoesNotPartiallyUpdateCounters) {
  MoeExpertState state;
  ASSERT_STATUS_OK(state.SetCounterParameters(1, std::numeric_limits<double>::max()));
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 2));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  auto* usage = state.GetUsage(kernels_[0]);
  ASSERT_NE(usage, nullptr);
  const int first[] = {0};
  ASSERT_STATUS_OK(usage->RecordUsage(first));
  const int both[] = {1, 0};
  EXPECT_FALSE(usage->RecordUsage(both).IsOK());
  EXPECT_EQ(state.GetSnapshot().at({"main", 0}).counters,
            (InlinedVector<double>{std::numeric_limits<double>::max(), 0}));
}

TEST_F(MoeExpertStateTest, DistinctKernelsCanUpdateIndependently) {
  MoeExpertState state, other;
  for (size_t i = 0; i < 3; ++i) {
    ASSERT_STATUS_OK(state.RegisterNode(kernels_[i], "main", i, "MoE", 2));
  }
  ASSERT_STATUS_OK(other.RegisterNode(kernels_[0], "main", 0, "MoE", 2));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  ASSERT_STATUS_OK(other.FinalizeInitialization());
  InlinedVector<std::thread> workers;
  for (const auto* kernel : kernels_) {
    auto* usage = state.GetUsage(kernel);
    workers.emplace_back([usage]() {
      const int selected[] = {1, 1};
      for (int j = 0; j < 100; ++j) {
        ASSERT_STATUS_OK(usage->RecordUsage(selected));
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (const auto& [key, node] : state.GetSnapshot()) {
    EXPECT_EQ(node.counters, (InlinedVector<double>{0, 100}));
  }
  EXPECT_EQ(other.GetSnapshot().at({"main", 0}).counters, (InlinedVector<double>{0, 0}));
}

TEST_F(MoeExpertStateTest, RejectsOverlappingRunsButAllowsIndependentSessions) {
  MoeExpertState state, other;
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  ASSERT_STATUS_OK(other.FinalizeInitialization());
  ASSERT_STATUS_OK(state.BeginRun());
  EXPECT_FALSE(state.BeginRun().IsOK());
  ASSERT_STATUS_OK(other.BeginRun());
  other.EndRun();
  state.EndRun();
  ASSERT_STATUS_OK(state.BeginRun());
  state.EndRun();
}
}  // namespace
#endif
}  // namespace onnxruntime::test
