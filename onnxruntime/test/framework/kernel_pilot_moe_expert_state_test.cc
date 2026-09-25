// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <thread>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "core/common/logging/logging.h"
#include "core/framework/kernel_pilot_moe_expert_state.h"
#include "core/framework/session_state.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/capturing_sink.h"
#include "test/util/include/inference_session_wrapper.h"

namespace onnxruntime::test {
#if !defined(ORT_MINIMAL_BUILD)
namespace {
class KernelPilotMoeExpertStateTest : public testing::Test {
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

  Status CollectAndRecord(KernelPilotMoeExpertState& state, const OpKernel* kernel, gsl::span<const int> ids) {
    auto* pilot = state.GetKernelPilot(kernel);
    ORT_RETURN_IF_NOT(pilot, "Missing test kernel collector.");
    ORT_RETURN_IF_ERROR(pilot->Moe().BeginInvocation(pilot->Moe().ExpertCount()));
    ORT_RETURN_IF_ERROR(pilot->Moe().Collect(ids));
    return pilot->RecordUsage();
  }

  static InlinedVector<double> Counters(const KernelPilotMoeExpertState& state, const OpKernel* kernel) {
    InlinedVector<double> counters;
    ORT_THROW_IF_ERROR(state.GetCounters(kernel, counters));
    return counters;
  }
};

TEST_F(KernelPilotMoeExpertStateTest, KernelExpertDictionarySeparatesNodesAndSubgraphs) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 3));
  auto* pilot = state.GetKernelPilot(kernels_[0]);
  ASSERT_NE(pilot, nullptr);
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[1], "main", 1, "QMoE", 2));
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[2], "main/0/4:body", 0, "MoE", 4));
  EXPECT_EQ(state.GetKernelPilot(kernels_[0]), pilot);
  EXPECT_NE(state.GetKernelPilot(kernels_[1]), pilot);
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
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  InlinedVector<double> first;
  ASSERT_STATUS_OK(state.GetCounters(kernels_[0], first));
  ASSERT_EQ(first.size(), 3U);
  EXPECT_DOUBLE_EQ(first[0], 0.19);
  EXPECT_DOUBLE_EQ(first[1], 0);
  EXPECT_DOUBLE_EQ(first[2], 0.19);
  InlinedVector<double> second;
  ASSERT_STATUS_OK(state.GetCounters(kernels_[1], second));
  EXPECT_EQ(second, (InlinedVector<double>{0, 0}));
  InlinedVector<double> third;
  ASSERT_STATUS_OK(state.GetCounters(kernels_[2], third));
  EXPECT_EQ(third, (InlinedVector<double>{0, 0, 0, 0}));
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[2], selected));
  InlinedVector<double> counters;
  ASSERT_STATUS_OK(state.GetCounters(kernels_[2], counters));
  EXPECT_EQ(counters, (InlinedVector<double>{0.1, 0, 0.1, 0}));

  const auto stats = state.GetExpertStats();
  ASSERT_EQ(stats.size(), 9U);
  for (const auto& stat : stats) {
    if (stat.kernel == kernels_[0]) {
      EXPECT_LT(stat.expert_id, 3U);
      EXPECT_DOUBLE_EQ(stat.popularity, stat.expert_id == 1 ? 0.0 : 0.19);
    } else if (stat.kernel == kernels_[1]) {
      EXPECT_LT(stat.expert_id, 2U);
      EXPECT_DOUBLE_EQ(stat.popularity, 0.0);
    } else {
      ASSERT_EQ(stat.kernel, kernels_[2]);
      EXPECT_LT(stat.expert_id, 4U);
      EXPECT_DOUBLE_EQ(stat.popularity, (stat.expert_id == 0 || stat.expert_id == 2) ? 0.1 : 0.0);
    }
  }
}

TEST_F(KernelPilotMoeExpertStateTest, LogsCounterUpdateAsStructuredJson) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 7, "MoE", 3));
  ASSERT_STATUS_OK(state.FinalizeInitialization());

  auto capturing_sink = std::make_unique<CapturingSink>();
  auto* capturing_sink_ptr = capturing_sink.get();
  logging::LoggingManager logging_manager(
      std::move(capturing_sink), logging::Severity::kINFO, false,
      logging::LoggingManager::InstanceType::Temporal);
  auto logger = logging_manager.CreateLogger("moe_counter_update");
  ASSERT_STATUS_OK(state.BeginRun("request \"one\"", logger.get()));

  const int selected[] = {2, 0, 2};
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  ASSERT_STATUS_OK(state.EndRun());

  ASSERT_EQ(capturing_sink_ptr->Messages().size(), 1U);
  const std::string& message = capturing_sink_ptr->Messages()[0];
  constexpr std::string_view marker{"moe_expert_counters "};
  const size_t marker_position = message.find(marker);
  ASSERT_NE(marker_position, std::string::npos);
  const auto event = nlohmann::json::parse(message.substr(marker_position + marker.size()));
  EXPECT_EQ(event["request_id"], "request \"one\"");
  EXPECT_EQ(event["graph_scope"], "main");
  EXPECT_EQ(event["node_index"], 7);
  EXPECT_EQ(event["node_type"], "Identity");
  EXPECT_EQ(event["selected_experts"], nlohmann::json({2, 0}));
  EXPECT_EQ(event["counters"], nlohmann::json({0.1, 0.0, 0.1}));
}

TEST_F(KernelPilotMoeExpertStateTest, LogsGraphScopeForSubgraphNode) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main/4/11:then_branch", 0, "MoE", 3));
  ASSERT_STATUS_OK(state.FinalizeInitialization());

  auto capturing_sink = std::make_unique<CapturingSink>();
  auto* capturing_sink_ptr = capturing_sink.get();
  logging::LoggingManager logging_manager(
      std::move(capturing_sink), logging::Severity::kINFO, false,
      logging::LoggingManager::InstanceType::Temporal);
  auto logger = logging_manager.CreateLogger("moe_subgraph_counter_update");
  ASSERT_STATUS_OK(state.BeginRun("request", logger.get()));

  const int selected[] = {1};
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  ASSERT_STATUS_OK(state.EndRun());

  ASSERT_EQ(capturing_sink_ptr->Messages().size(), 1U);
  const std::string& message = capturing_sink_ptr->Messages()[0];
  constexpr std::string_view marker{"moe_expert_counters "};
  const size_t marker_position = message.find(marker);
  ASSERT_NE(marker_position, std::string::npos);
  const auto event = nlohmann::json::parse(message.substr(marker_position + marker.size()));
  EXPECT_EQ(event["graph_scope"], "main/4/11:then_branch");
  EXPECT_EQ(event["node_index"], 0);
}

TEST_F(KernelPilotMoeExpertStateTest, AppliesExponentialUpdateToEveryExpert) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.SetCounterParameters(0.5, 0.25));
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 3));
  std::istringstream initial(
      "moe_expert_state 1\n"
      "\"main\" 0 Identity 0 4\n"
      "\"main\" 0 Identity 1 2\n"
      "\"main\" 0 Identity 2 1\n");
  ASSERT_STATUS_OK(state.Load(initial));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  const int selected[] = {2, 0, 2};
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{2.25, 1, 0.75}));
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{1.375, 0.5, 0.625}));
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], {}));
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{0.6875, 0.25, 0.3125}));
}

TEST_F(KernelPilotMoeExpertStateTest, ValidatesCounterParameters) {
  for (const auto& [alpha, beta] : {
           std::pair{-0.1, 1.0},
           std::pair{1.1, 1.0},
           std::pair{0.0, 1.1},
           std::pair{0.9, 0.2},
           std::pair{1.0, std::numeric_limits<double>::epsilon()},
           std::pair{std::numeric_limits<double>::infinity(), 1.0},
           std::pair{std::numeric_limits<double>::quiet_NaN(), 0.0},
           std::pair{1.0, -0.1},
           std::pair{1.0, std::numeric_limits<double>::infinity()},
           std::pair{0.0, std::numeric_limits<double>::quiet_NaN()}}) {
    SCOPED_TRACE(MakeString("alpha=", alpha, ", beta=", beta));
    KernelPilotMoeExpertState state;
    EXPECT_FALSE(state.SetCounterParameters(alpha, beta).IsOK());
  }
  for (const auto& [alpha, beta] : {
           std::pair{0.0, 0.0}, std::pair{1.0, 0.0}, std::pair{0.0, 1.0},
           std::pair{0.9, 0.1}, std::pair{0.5, 0.5}, std::pair{0.5, 0.25}}) {
    SCOPED_TRACE(MakeString("alpha=", alpha, ", beta=", beta));
    KernelPilotMoeExpertState state;
    EXPECT_STATUS_OK(state.SetCounterParameters(alpha, beta));
  }
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 1));
  EXPECT_FALSE(state.SetCounterParameters(0.5, 0.25).IsOK());
}

TEST_F(KernelPilotMoeExpertStateTest, LoadsPartialStateAndValidatesAtomically) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 2, "QMoE", 3));
  std::istringstream valid("moe_expert_state 1\n\"main\" 2 Identity 1 2.5\n");
  ASSERT_STATUS_OK(state.Load(valid));
  InlinedVector<double> counters;
  for (const auto* invalid : {
           "moe_expert_state 2\n",
           "moe_expert_state 1\n\"main\" 2 Identity 0 1\n\"main\" 2 Identity 0 2\n",
           "moe_expert_state 1\n\"main\" 2 Identity 0 1\n\"unknown\" 2 Identity 0 1\n",
           "moe_expert_state 1\n\"main\" 3 Identity 0 1\n",
           "moe_expert_state 1\n\"main\" 2 MoE 0 1\n",
           "moe_expert_state 1\n\"main\" 2 Identity -1 1\n",
           "moe_expert_state 1\n\"main\" 2 Identity 3 1\n",
           "moe_expert_state 1\n\"main\" 2 Identity 0 -1\n",
           "moe_expert_state 1\n\"main\" 2 Identity 0 nan\n",
           "moe_expert_state 1\n\"main\" 2 Identity 0 inf\n",
           "moe_expert_state 1\n\"main\" 2 Identity 0 1 extra\n",
           "moe_expert_state 1\nincomplete\n"}) {
    SCOPED_TRACE(invalid);
    std::istringstream input(invalid);
    EXPECT_FALSE(state.Load(input).IsOK());
    ASSERT_STATUS_OK(state.GetCounters(kernels_[0], counters));
    EXPECT_EQ(counters, (InlinedVector<double>{0, 2.5, 0}));
  }
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  const int selected[] = {1, 1};
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  ASSERT_STATUS_OK(state.GetCounters(kernels_[0], counters));
  ASSERT_EQ(counters.size(), 3U);
  EXPECT_DOUBLE_EQ(counters[0], 0);
  EXPECT_DOUBLE_EQ(counters[1], 2.35);
  EXPECT_DOUBLE_EQ(counters[2], 0);
}

TEST_F(KernelPilotMoeExpertStateTest, RejectsInvalidRegistrationAndUpdates) {
  KernelPilotMoeExpertState state;
  EXPECT_FALSE(state.RegisterNode(nullptr, "main", 0, "MoE", 2).IsOK());
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 2));
  EXPECT_FALSE(state.RegisterNode(kernels_[1], "main", 0, "MoE", 2).IsOK());
  EXPECT_FALSE(state.RegisterNode(kernels_[0], "main", 1, "MoE", 2).IsOK());
  EXPECT_FALSE(state.RegisterNode(kernels_[1], "main", 1, "Add", 2).IsOK());
  EXPECT_FALSE(state.RegisterNode(kernels_[1], "main", 1, "QMoE", 0).IsOK());
  EXPECT_EQ(state.TotalExpertCount(), 2U);
  ASSERT_STATUS_OK(state.GetKernelPilot(kernels_[0])->Moe().BeginInvocation(2));
  EXPECT_FALSE(state.GetKernelPilot(kernels_[0])->RecordUsage().IsOK());
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  EXPECT_FALSE(state.RegisterNode(kernels_[1], "main", 1, "MoE", 2).IsOK());
  std::istringstream initial("moe_expert_state 1\n");
  EXPECT_FALSE(state.Load(initial).IsOK());
  EXPECT_FALSE(state.FinalizeInitialization().IsOK());
  const int out_of_bounds[] = {0, 2};
  const int negative[] = {0, -1};
  EXPECT_FALSE(CollectAndRecord(state, kernels_[0], out_of_bounds).IsOK());
  EXPECT_FALSE(CollectAndRecord(state, kernels_[0], negative).IsOK());
  EXPECT_EQ(state.GetKernelPilot(kernels_[1]), nullptr);
  EXPECT_EQ(state.GetKernelPilot(nullptr), nullptr);
  ASSERT_STATUS_OK(state.GetKernelPilot(kernels_[0])->Moe().BeginInvocation(3));
  EXPECT_FALSE(state.GetKernelPilot(kernels_[0])->RecordUsage().IsOK());
  ASSERT_STATUS_OK(state.GetKernelPilot(kernels_[0])->Moe().BeginInvocation(2));
  InlinedVector<double> counters{7};
  EXPECT_FALSE(state.GetCounters(kernels_[1], counters).IsOK());
  EXPECT_FALSE(state.GetCounters(nullptr, counters).IsOK());
  EXPECT_EQ(counters, (InlinedVector<double>{7}));
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{0, 0}));
  const int selected[] = {1};
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{0, 0.1}));
}

TEST_F(KernelPilotMoeExpertStateTest, ValidCoefficientsKeepCountersBounded) {
  for (const auto& [alpha, beta] : {
           std::pair{0.0, 0.0}, std::pair{1.0, 0.0}, std::pair{0.0, 1.0},
           std::pair{0.9, 0.1}, std::pair{0.5, 0.25}}) {
    for (const double initial_value : {0.0, std::numeric_limits<double>::max()}) {
      SCOPED_TRACE(MakeString("alpha=", alpha, ", beta=", beta, ", initial=", initial_value));
      KernelPilotMoeExpertState state;
      ASSERT_STATUS_OK(state.SetCounterParameters(alpha, beta));
      ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 2));
      std::stringstream initial;
      initial << std::setprecision(std::numeric_limits<double>::max_digits10)
              << "moe_expert_state 1\n\"main\" 0 Identity 0 " << initial_value
              << "\n\"main\" 0 Identity 1 " << initial_value << "\n";
      ASSERT_STATUS_OK(state.Load(initial));
      ASSERT_STATUS_OK(state.FinalizeInitialization());
      const int selected[] = {0, 0};
      for (int invocation = 0; invocation < 1024; ++invocation) {
        ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
        InlinedVector<double> counters;
        ASSERT_STATUS_OK(state.GetCounters(kernels_[0], counters));
        for (double counter : counters) {
          EXPECT_TRUE(std::isfinite(counter));
          EXPECT_GE(counter, 0);
          EXPECT_LE(counter, std::max(initial_value, 1.0));
        }
      }
    }
  }
}

TEST_F(KernelPilotMoeExpertStateTest, DistinctKernelsCanUpdateIndependently) {
  KernelPilotMoeExpertState state, other;
  for (size_t i = 0; i < 3; ++i) {
    ASSERT_STATUS_OK(state.RegisterNode(kernels_[i], "main", i, "MoE", 2));
  }
  ASSERT_STATUS_OK(other.RegisterNode(kernels_[0], "main", 0, "MoE", 2));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  ASSERT_STATUS_OK(other.FinalizeInitialization());
  InlinedVector<std::thread> workers;
  for (const auto* kernel : kernels_) {
    workers.emplace_back([this, &state, kernel]() {
      const int selected[] = {1, 1};
      for (int j = 0; j < 100; ++j) {
        ASSERT_STATUS_OK(CollectAndRecord(state, kernel, selected));
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  for (const auto* kernel : kernels_) {
    const auto counters = Counters(state, kernel);
    ASSERT_EQ(counters.size(), 2U);
    EXPECT_DOUBLE_EQ(counters[0], 0);
    EXPECT_NEAR(counters[1], 1.0 - std::pow(0.9, 100), 1e-14);
  }
  EXPECT_EQ(Counters(other, kernels_[0]), (InlinedVector<double>{0, 0}));
}

}  // namespace
#endif
}  // namespace onnxruntime::test
