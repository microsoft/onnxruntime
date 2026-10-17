// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
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
constexpr double kDefaultCounterAlpha = 0.75;

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

  Status CollectWeightedAndRecord(KernelPilotMoeExpertState& state, const OpKernel* kernel,
                                  gsl::span<const int> ids, gsl::span<const float> probabilities) {
    auto* pilot = state.GetKernelPilot(kernel);
    ORT_RETURN_IF_NOT(pilot, "Missing test kernel collector.");
    ORT_RETURN_IF_ERROR(pilot->Moe().BeginInvocation(pilot->Moe().ExpertCount()));
    ORT_RETURN_IF_ERROR(pilot->Moe().CollectWeighted(ids, probabilities));
    return pilot->RecordUsage();
  }

  static InlinedVector<double> Counters(const KernelPilotMoeExpertState& state, const OpKernel* kernel) {
    InlinedVector<double> counters;
    ORT_THROW_IF_ERROR(state.GetCounters(kernel, counters));
    return counters;
  }
};

class FakeMoeExpertCache final : public IKernelPilotMoeExpertCache {
 public:
  explicit FakeMoeExpertCache(int device_id = 0) : device_id_(device_id) {}

  int DeviceId() const noexcept override { return device_id_; }
  bool HasPendingSwap() const noexcept override { return pending_; }
  Status ReclaimCompletedSwap() override {
    if (completed_) {
      pending_ = false;
      completed_ = false;
    }
    return Status::OK();
  }
  Status StartSwap(int cuda_expert_id, int cpu_expert_id) override {
    ORT_RETURN_IF(pending_, "Test cache already has a pending swap.");
    pending_ = true;
    swaps_.emplace_back(cuda_expert_id, cpu_expert_id);
    return Status::OK();
  }

  void Complete() noexcept { completed_ = true; }
  const InlinedVector<std::pair<int, int>>& Swaps() const noexcept { return swaps_; }

 private:
  int device_id_;
  bool pending_{false};
  bool completed_{false};
  InlinedVector<std::pair<int, int>> swaps_;
};

class DeferredMoeUsage final : public IKernelPilotMoeExpertUsage {
 public:
  bool HasPendingUsage() const noexcept override { return pending; }
  Status CompleteUsage(KernelPilotMoeExpertUsageSnapshot& snapshot) override {
    pending = false;
    snapshot.counters = counters;
    snapshot.selected_probabilities = probabilities;
    snapshot.first_selected_routes = first_routes;
    return completion_status;
  }

  bool pending{true};
  InlinedVector<double> counters{0.25, 0.0, 0.25};
  InlinedVector<float> probabilities{0.5f, -1.0f, 0.75f};
  InlinedVector<int64_t> first_routes{1, -1, 0};
  Status completion_status;
};

TEST_F(KernelPilotMoeExpertStateTest, DefersGpuUsageUntilEndRunAndPreservesSelectionOrder) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 3));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  auto* pilot = state.GetKernelPilot(kernels_[0]);
  DeferredMoeUsage usage;
  ASSERT_STATUS_OK(pilot->AttachMoeExpertUsage(&usage));
  ASSERT_STATUS_OK(state.BeginRun("deferred", nullptr));
  ASSERT_STATUS_OK(pilot->Moe().BeginInvocation(3));
  ASSERT_STATUS_OK(pilot->RecordUsage());
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{0.0, 0.0, 0.0}));
  ASSERT_STATUS_OK(state.EndRun());
  EXPECT_FALSE(usage.pending);
  EXPECT_EQ(Counters(state, kernels_[0]), usage.counters);
  gsl::span<const int> selected;
  ASSERT_STATUS_OK(pilot->Moe().GetSelectedExperts(selected));
  EXPECT_EQ((InlinedVector<int>{selected.begin(), selected.end()}), (InlinedVector<int>{2, 0}));
  EXPECT_EQ(pilot->MoeExpertCounters()[0], 0.25);
}

TEST_F(KernelPilotMoeExpertStateTest, ValidatesAndFreezesCpuOffloadStrategy) {
  KernelPilotMoeExpertState state;
  EXPECT_FALSE(state.SetCpuOffloadStrategy("unknown").IsOK());
  ASSERT_STATUS_OK(state.SetCpuOffloadStrategy("adaptive"));
  ASSERT_STATUS_OK(state.SetCpuOffloadStrategy("basic"));
  ASSERT_STATUS_OK(state.SetCpuOffloadStrategy("layered"));
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "QMoE", 3));
  EXPECT_FALSE(state.SetCpuOffloadStrategy("adaptive").IsOK());
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  EXPECT_FALSE(state.SetCpuOffloadStrategy("basic").IsOK());
}

TEST_F(KernelPilotMoeExpertStateTest, ValidatesAndFreezesCpuOffloadGpuRatio) {
  KernelPilotMoeExpertState state;
  for (double ratio : {0.0, -0.1, 1.1, std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_FALSE(state.SetCpuOffloadGpuRatio(ratio).IsOK());
  }
  for (double ratio : {0.01, 0.8, 1.0}) {
    ASSERT_STATUS_OK(state.SetCpuOffloadGpuRatio(ratio));
  }
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "QMoE", 3));
  EXPECT_FALSE(state.SetCpuOffloadGpuRatio(0.8).IsOK());
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  EXPECT_FALSE(state.SetCpuOffloadGpuRatio(0.8).IsOK());
}

TEST_F(KernelPilotMoeExpertStateTest, ValidatesAndFreezesCpuOffloadLayerOrder) {
  KernelPilotMoeExpertState state;
  EXPECT_FALSE(state.SetCpuOffloadLayerOrder("unknown").IsOK());
  for (const char* order : {"first", "last", "middle"}) {
    ASSERT_STATUS_OK(state.SetCpuOffloadLayerOrder(order));
  }
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "QMoE", 3));
  EXPECT_FALSE(state.SetCpuOffloadLayerOrder("first").IsOK());
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  EXPECT_FALSE(state.SetCpuOffloadLayerOrder("last").IsOK());
}

TEST_F(KernelPilotMoeExpertStateTest, DiscardsDeferredUsageForFailedKernel) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 3));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  auto* pilot = state.GetKernelPilot(kernels_[0]);
  DeferredMoeUsage usage;
  ASSERT_STATUS_OK(pilot->AttachMoeExpertUsage(&usage));
  ASSERT_STATUS_OK(state.BeginRun("failed kernel", nullptr));
  ASSERT_STATUS_OK(pilot->Moe().BeginInvocation(3));
  ASSERT_STATUS_OK(state.EndRun(false));
  EXPECT_FALSE(usage.pending);
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{0.0, 0.0, 0.0}));
}

TEST_F(KernelPilotMoeExpertStateTest, CommitsSuccessfulDeferredKernelWhenLaterRunWorkFails) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 3));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  auto* pilot = state.GetKernelPilot(kernels_[0]);
  DeferredMoeUsage usage;
  ASSERT_STATUS_OK(pilot->AttachMoeExpertUsage(&usage));
  ASSERT_STATUS_OK(state.BeginRun("failed later work", nullptr));
  ASSERT_STATUS_OK(pilot->Moe().BeginInvocation(3));
  ASSERT_STATUS_OK(pilot->RecordUsage());
  ASSERT_STATUS_OK(state.EndRun(false));
  EXPECT_EQ(Counters(state, kernels_[0]), usage.counters);
}

TEST_F(KernelPilotMoeExpertStateTest, PropagatesAsyncUsageFailureAndReleasesRun) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 3));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  auto* pilot = state.GetKernelPilot(kernels_[0]);
  DeferredMoeUsage usage;
  usage.completion_status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "test CPU worker failure");
  ASSERT_STATUS_OK(pilot->AttachMoeExpertUsage(&usage));
  ASSERT_STATUS_OK(state.BeginRun("async failure", nullptr));
  ASSERT_STATUS_OK(pilot->Moe().BeginInvocation(3));
  ASSERT_STATUS_OK(pilot->RecordUsage());
  const Status status = state.EndRun();
  EXPECT_FALSE(status.IsOK());
  EXPECT_EQ(status.ErrorMessage(), "test CPU worker failure");
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{0.0, 0.0, 0.0}));
  ASSERT_STATUS_OK(state.BeginRun("next run", nullptr));
  ASSERT_STATUS_OK(state.EndRun());
}

void MakeEligibleForCudaFp16MoePlacement(const OpKernel* kernel) {
  auto& node = const_cast<Node&>(kernel->Node());
  const_cast<std::string&>(node.OpType()) = "MoE";
  const_cast<std::string&>(node.Domain()) = kMSDomain;
  node.SetExecutionProviderType(kCudaExecutionProvider);
  auto* input_type = const_cast<ONNX_NAMESPACE::TypeProto*>(node.InputDefs()[0]->TypeAsProto());
  input_type->mutable_tensor_type()->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
}

void MakeEligibleForCudaQMoEPlacement(const OpKernel* kernel) {
  MakeEligibleForCudaFp16MoePlacement(kernel);
  auto& node = const_cast<Node&>(kernel->Node());
  const_cast<std::string&>(node.OpType()) = "QMoE";
  node.AddAttribute("expert_weight_bits", int64_t{4});
  node.AddAttribute("weights_prepacked", int64_t{0});
}

TEST_F(KernelPilotMoeExpertStateTest, LayeredPlacementPreservesExactBudgetAndTopologicalOrder) {
  const std::array<size_t, 3> expert_counts{3, 5, 4};
  const std::array<std::array<size_t, 3>, 12> expected_counts{{{3, 5, 3}, {3, 4, 3}, {2, 4, 3}, {2, 4, 2}, {2, 4, 1}, {2, 4, 0}, {2, 3, 0}, {2, 2, 0}, {2, 1, 0}, {2, 0, 0}, {1, 0, 0}, {0, 0, 0}}};
  for (size_t cpu_count = 1; cpu_count <= 12; ++cpu_count) {
    SCOPED_TRACE(cpu_count);
    KernelPilotMoeExpertState state;
    ASSERT_STATUS_OK(state.SetCpuOffloadStrategy("layered"));
    ASSERT_STATUS_OK(state.SetCpuOffloadExpertCount(cpu_count));
    for (size_t i = 0; i < expert_counts.size(); ++i) {
      MakeEligibleForCudaQMoEPlacement(kernels_[i]);
      ASSERT_STATUS_OK(state.RegisterNode(kernels_[i], "main", 20 - i * 10, "QMoE", expert_counts[i], i));
    }
    ASSERT_STATUS_OK(state.FinalizeInitialization());
    for (size_t i = 0; i < expert_counts.size(); ++i) {
      gsl::span<const int> experts;
      ASSERT_STATUS_OK(state.GetKernelPilot(kernels_[i])->GetMoeCudaExperts(experts));
      ASSERT_EQ(experts.size(), expected_counts[cpu_count - 1][i]);
      for (size_t expert = 0; expert < experts.size(); ++expert) {
        EXPECT_EQ(experts[expert], static_cast<int>(expert));
      }
    }
  }
}

TEST_F(KernelPilotMoeExpertStateTest, LayeredPlacementUsesConfiguredRatioAndGraphScopeOrder) {
  for (double ratio : {0.01, 0.5, 1.0}) {
    SCOPED_TRACE(ratio);
    KernelPilotMoeExpertState state;
    ASSERT_STATUS_OK(state.SetCpuOffloadStrategy("layered"));
    ASSERT_STATUS_OK(state.SetCpuOffloadGpuRatio(ratio));
    ASSERT_STATUS_OK(state.SetCpuOffloadExpertCount(7));
    for (size_t i = 0; i < 3; ++i) {
      MakeEligibleForCudaQMoEPlacement(kernels_[i]);
      ASSERT_STATUS_OK(state.RegisterNode(kernels_[i], i == 0 ? "main" : "main/0/body", i, "QMoE", 4, 2 - i));
    }
    ASSERT_STATUS_OK(state.FinalizeInitialization());
    // Main graph precedes subgraphs; subgraph nodes follow topological, not index order.
    const std::array<size_t, 3> expected = ratio == 0.5 ? std::array<size_t, 3>{2, 1, 2}
                                                        : std::array<size_t, 3>{4, 0, 1};
    for (size_t i = 0; i < 3; ++i) {
      gsl::span<const int> experts;
      ASSERT_STATUS_OK(state.GetKernelPilot(kernels_[i])->GetMoeCudaExperts(experts));
      EXPECT_EQ(experts.size(), expected[i]);
    }
  }
}

TEST_F(KernelPilotMoeExpertStateTest, LayerOrderControlsLayeredOverflowAndBoundary) {
  const std::array<size_t, 3> expert_counts{3, 5, 4};
  const std::array<size_t, 7> cpu_counts{1, 2, 5, 6, 8, 10, 12};
  const std::array<std::array<size_t, 3>, 7> last_counts{{{2, 5, 4}, {2, 4, 4}, {0, 4, 3}, {0, 3, 3}, {0, 1, 3}, {0, 0, 2}, {0, 0, 0}}};
  const std::array<std::array<size_t, 3>, 7> middle_counts{{{2, 5, 4}, {2, 5, 3}, {0, 4, 3}, {0, 4, 2}, {0, 4, 0}, {0, 2, 0}, {0, 0, 0}}};
  for (const char* order : {"last", "middle"}) {
    for (size_t budget = 0; budget < cpu_counts.size(); ++budget) {
      SCOPED_TRACE(MakeString("order=", order, ", cpu_count=", cpu_counts[budget]));
      KernelPilotMoeExpertState state;
      ASSERT_STATUS_OK(state.SetCpuOffloadStrategy("layered"));
      ASSERT_STATUS_OK(state.SetCpuOffloadLayerOrder(order));
      ASSERT_STATUS_OK(state.SetCpuOffloadExpertCount(cpu_counts[budget]));
      for (size_t i = 0; i < expert_counts.size(); ++i) {
        MakeEligibleForCudaQMoEPlacement(kernels_[i]);
        ASSERT_STATUS_OK(state.RegisterNode(kernels_[i], "main", 20 - i * 10, "QMoE", expert_counts[i], i));
      }
      ASSERT_STATUS_OK(state.FinalizeInitialization());
      const auto& expected = std::string_view(order) == "last" ? last_counts[budget] : middle_counts[budget];
      for (size_t i = 0; i < expert_counts.size(); ++i) {
        gsl::span<const int> experts;
        ASSERT_STATUS_OK(state.GetKernelPilot(kernels_[i])->GetMoeCudaExperts(experts));
        EXPECT_EQ(experts.size(), expected[i]);
      }
    }
  }
}

TEST_F(KernelPilotMoeExpertStateTest, LayeredPopularityAndSwapsStayWithinGpuLayersIncludingBoundary) {
  for (const char* order : {"first", "last", "middle"}) {
    for (size_t swap_limit : {0U, 1U, 5U}) {
      SCOPED_TRACE(swap_limit);
      SCOPED_TRACE(order);
      KernelPilotMoeExpertState state;
      ASSERT_STATUS_OK(state.SetCpuOffloadStrategy("layered"));
      ASSERT_STATUS_OK(state.SetCpuOffloadLayerOrder(order));
      ASSERT_STATUS_OK(state.SetCpuOffloadExpertCount(7));
      ASSERT_STATUS_OK(state.SetMaxInFlightSwapsPerDevice(swap_limit));
      for (size_t i = 0; i < 3; ++i) {
        MakeEligibleForCudaQMoEPlacement(kernels_[i]);
        ASSERT_STATUS_OK(state.RegisterNode(kernels_[i], "main", i, "QMoE", 4, i));
      }
      std::istringstream initial(
          R"({"alpha":0,"epsilon":0,"experts":[)"
          R"({"moe":0,"expert":3,"popularity":100},)"
          R"({"moe":1,"expert":3,"popularity":100},)"
          R"({"moe":2,"expert":3,"popularity":10000}]})");
      ASSERT_STATUS_OK(state.Load(initial));
      ASSERT_STATUS_OK(state.FinalizeInitialization());
      const std::array<InlinedVector<int>, 3> expected =
          std::string_view(order) == "first"  ? std::array<InlinedVector<int>, 3>{{{0, 1, 3}, {0, 3}, {}}}
          : std::string_view(order) == "last" ? std::array<InlinedVector<int>, 3>{{{}, {0, 3}, {0, 1, 3}}}
                                              : std::array<InlinedVector<int>, 3>{{{}, {0, 1, 3}, {0, 3}}};
      std::array<FakeMoeExpertCache, 3> caches;
      for (size_t i = 0; i < 3; ++i) {
        auto* pilot = state.GetKernelPilot(kernels_[i]);
        gsl::span<const int> experts;
        ASSERT_STATUS_OK(pilot->GetMoeCudaExperts(experts));
        EXPECT_EQ((InlinedVector<int>{experts.begin(), experts.end()}), expected[i]);
        ASSERT_STATUS_OK(pilot->AttachMoeExpertCache(&caches[i]));
      }
      ASSERT_STATUS_OK(state.BeginRun("", nullptr));
      const int selected[] = {2};
      for (const auto* kernel : kernels_) {
        ASSERT_STATUS_OK(CollectAndRecord(state, kernel, selected));
      }
      ASSERT_STATUS_OK(state.EndRun());
      size_t swaps = 0;
      for (size_t i = 0; i < caches.size(); ++i) {
        if (!expected[i].empty() && swaps < swap_limit) {
          EXPECT_EQ(caches[i].Swaps(), (InlinedVector<std::pair<int, int>>{{0, 2}}));
          ++swaps;
        } else {
          EXPECT_TRUE(caches[i].Swaps().empty());
        }
      }
    }
  }
}

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
  EXPECT_DOUBLE_EQ(first[0], 1.0 - std::pow(kDefaultCounterAlpha, 2));
  EXPECT_DOUBLE_EQ(first[1], 0);
  EXPECT_DOUBLE_EQ(first[2], 1.0 - std::pow(kDefaultCounterAlpha, 2));
  InlinedVector<double> second;
  ASSERT_STATUS_OK(state.GetCounters(kernels_[1], second));
  EXPECT_EQ(second, (InlinedVector<double>{0, 0}));
  InlinedVector<double> third;
  ASSERT_STATUS_OK(state.GetCounters(kernels_[2], third));
  EXPECT_EQ(third, (InlinedVector<double>{0, 0, 0, 0}));
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[2], selected));
  InlinedVector<double> counters;
  ASSERT_STATUS_OK(state.GetCounters(kernels_[2], counters));
  EXPECT_EQ(counters, (InlinedVector<double>{1.0 - kDefaultCounterAlpha, 0,
                                             1.0 - kDefaultCounterAlpha, 0}));

  const auto stats = state.GetExpertStats();
  ASSERT_EQ(stats.size(), 9U);
  for (const auto& stat : stats) {
    if (stat.kernel == kernels_[0]) {
      EXPECT_LT(stat.expert_id, 3U);
      EXPECT_DOUBLE_EQ(stat.popularity,
                       stat.expert_id == 1 ? 0.0 : 1.0 - std::pow(kDefaultCounterAlpha, 2));
    } else if (stat.kernel == kernels_[1]) {
      EXPECT_LT(stat.expert_id, 2U);
      EXPECT_DOUBLE_EQ(stat.popularity, 0.0);
    } else {
      ASSERT_EQ(stat.kernel, kernels_[2]);
      EXPECT_LT(stat.expert_id, 4U);
      EXPECT_DOUBLE_EQ(stat.popularity,
                       (stat.expert_id == 0 || stat.expert_id == 2) ? 1.0 - kDefaultCounterAlpha : 0.0);
    }
  }
}

TEST_F(KernelPilotMoeExpertStateTest, LogsCounterUpdateAsStructuredJson) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.SetCpuOffloadExpertCount(2));
  MakeEligibleForCudaFp16MoePlacement(kernels_[0]);
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 7, "MoE", 3));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  FakeMoeExpertCache cache;
  auto* pilot = state.GetKernelPilot(kernels_[0]);
  ASSERT_NE(pilot, nullptr);
  ASSERT_STATUS_OK(pilot->AttachMoeExpertCache(&cache));
  ASSERT_STATUS_OK(pilot->PublishMoeExpertSwap(0, 2));
  const int initially_selected[] = {2};
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], initially_selected));

  auto capturing_sink = std::make_unique<CapturingSink>();
  auto* capturing_sink_ptr = capturing_sink.get();
  logging::LoggingManager logging_manager(
      std::move(capturing_sink), logging::Severity::kINFO, false,
      logging::LoggingManager::InstanceType::Temporal);
  auto logger = logging_manager.CreateLogger("moe_counter_update");
  ASSERT_STATUS_OK(state.BeginRun("request \"one\"", logger.get()));

  const int selected[] = {2, 0, 1, 2};
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  ASSERT_STATUS_OK(state.EndRun());

  ASSERT_EQ(capturing_sink_ptr->Messages().size(), 2U);
  const std::string& message = capturing_sink_ptr->Messages()[0];
  constexpr std::string_view marker{"moe_expert_counters "};
  const size_t marker_position = message.find(marker);
  ASSERT_NE(marker_position, std::string::npos);
  const auto event = nlohmann::json::parse(message.substr(marker_position + marker.size()));
  EXPECT_EQ(event["request_id"], "request \"one\"");
  EXPECT_EQ(event["graph_scope"], "main");
  EXPECT_EQ(event["node_index"], 7);
  EXPECT_EQ(event["node_type"], "MoE");
  EXPECT_EQ(event["moe_count"], 1);
  EXPECT_EQ(event["total_expert_count"], 3);
  EXPECT_EQ(event["expert_offset"], 0);
  EXPECT_EQ(event["selected_experts"],
            nlohmann::json({
                {
                    {"expert_id", 2},
                    {"score", 1.0 - kDefaultCounterAlpha},
                    {"selected_rank", 0},
                    {"node_rank", 0},
                    {"global_position", 0},
                    {"device", "CUDA"},
                },
                {
                    {"expert_id", 0},
                    {"score", 0.0},
                    {"selected_rank", 1},
                    {"node_rank", 1},
                    {"global_position", 1},
                    {"device", "CPU"},
                },
                {
                    {"expert_id", 1},
                    {"score", 0.0},
                    {"selected_rank", 2},
                    {"node_rank", 2},
                    {"global_position", 2},
                    {"device", "CPU"},
                },
            }));
  EXPECT_NE(capturing_sink_ptr->Messages()[1].find("moe_expert_swaps "), std::string::npos);
  EXPECT_FALSE(event.contains("counters"));
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

TEST_F(KernelPilotMoeExpertStateTest, GlobalPositionUsesPreUpdateScoresAndGlobalIdTies) {
  KernelPilotMoeExpertState state;
  for (size_t node_index = 0; node_index < 3; ++node_index) {
    ASSERT_STATUS_OK(state.RegisterNode(kernels_[node_index], "main", node_index, "MoE", 2));
  }
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  const int selected[] = {0};
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));

  auto capturing_sink = std::make_unique<CapturingSink>();
  auto* capturing_sink_ptr = capturing_sink.get();
  logging::LoggingManager logging_manager(
      std::move(capturing_sink), logging::Severity::kINFO, false,
      logging::LoggingManager::InstanceType::Temporal);
  auto logger = logging_manager.CreateLogger("moe_counter_update");
  ASSERT_STATUS_OK(state.BeginRun("", logger.get()));
  for (const auto* kernel : kernels_) {
    ASSERT_STATUS_OK(CollectAndRecord(state, kernel, selected));
  }
  ASSERT_STATUS_OK(state.EndRun());

  ASSERT_EQ(capturing_sink_ptr->Messages().size(), 3U);
  constexpr std::string_view marker{"moe_expert_counters "};
  const std::string& message = capturing_sink_ptr->Messages().back();
  const size_t marker_position = message.find(marker);
  ASSERT_NE(marker_position, std::string::npos);
  const auto event = nlohmann::json::parse(message.substr(marker_position + marker.size()));
  EXPECT_EQ(event["moe_count"], 3);
  EXPECT_EQ(event["total_expert_count"], 6);
  ASSERT_EQ(event["selected_experts"].size(), 1U);
  EXPECT_EQ(event["selected_experts"][0]["global_position"], 4);
}

TEST_F(KernelPilotMoeExpertStateTest, LimitsLogsWithoutLimitingCountersAndResetsBudgetPerRun) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.SetCounterAlpha(0));
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 2));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  auto sink = std::make_unique<CapturingSink>();
  auto* sink_ptr = sink.get();
  logging::LoggingManager manager(std::move(sink), logging::Severity::kINFO, false,
                                  logging::LoggingManager::InstanceType::Temporal);
  auto logger = manager.CreateLogger("moe_log_budget");
  constexpr size_t limit = KernelPilotMoeExpertState::kMaxCounterLogRecordsPerRun;
  for (int run = 0; run < 2; ++run) {
    const std::string request_id = MakeString("request \"", run, "\"");
    ASSERT_STATUS_OK(state.BeginRun(request_id, logger.get()));
    const size_t start = sink_ptr->Messages().size();
    const int selected[] = {0};
    for (size_t i = 0; i < limit; ++i) {
      ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
    }
    ASSERT_EQ(sink_ptr->Messages().size(), start + limit);
    for (size_t i = start; i < start + limit; ++i) {
      EXPECT_NE(sink_ptr->Messages()[i].find("moe_expert_counters "), std::string::npos);
    }
    const int suppressed[] = {1};
    for (int i = 0; i < 3; ++i) {
      ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], suppressed));
    }
    ASSERT_STATUS_OK(state.EndRun());
    EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{0, 1}));
    ASSERT_EQ(sink_ptr->Messages().size(), start + limit + 1);
    const auto& message = sink_ptr->Messages().back();
    constexpr std::string_view marker{"moe_expert_counters_truncated "};
    const auto position = message.find(marker);
    ASSERT_NE(position, std::string::npos);
    const auto summary = nlohmann::json::parse(message.substr(position + marker.size()));
    EXPECT_EQ(summary["request_id"], request_id);
    EXPECT_EQ(summary["max_records"], limit);
  }
}

TEST_F(KernelPilotMoeExpertStateTest, DisabledInfoDoesNotConsumeLogBudget) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.SetCounterAlpha(0));
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 2));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  auto sink = std::make_unique<CapturingSink>();
  auto* sink_ptr = sink.get();
  logging::LoggingManager manager(std::move(sink), logging::Severity::kWARNING, false,
                                  logging::LoggingManager::InstanceType::Temporal);
  auto logger = manager.CreateLogger("moe_disabled_info");
  ASSERT_STATUS_OK(state.BeginRun("request", logger.get()));
  const int selected[] = {1};
  for (size_t i = 0; i <= KernelPilotMoeExpertState::kMaxCounterLogRecordsPerRun; ++i) {
    ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  }
  EXPECT_TRUE(sink_ptr->Messages().empty());
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{0, 1}));
  logger->SetSeverity(logging::Severity::kINFO);
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  ASSERT_STATUS_OK(state.EndRun());
  ASSERT_EQ(sink_ptr->Messages().size(), 1U);
  EXPECT_NE(sink_ptr->Messages()[0].find("moe_expert_counters "), std::string::npos);
}

TEST_F(KernelPilotMoeExpertStateTest, ParallelNodesShareLogBudget) {
  class SynchronizedCapturingSink : public CapturingSink {
   public:
    void SendImpl(const logging::Timestamp& timestamp, const std::string& logger_id,
                  const logging::Capture& message) override {
      std::lock_guard<std::mutex> lock(mutex_);
      CapturingSink::SendImpl(timestamp, logger_id, message);
    }

   private:
    std::mutex mutex_;
  };
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.SetCounterAlpha(0.5));
  for (size_t i = 0; i < 3; ++i) {
    ASSERT_STATUS_OK(state.RegisterNode(kernels_[i], MakeString("main/", i, "/4:body"), 0, "MoE", 2));
  }
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  auto sink = std::make_unique<SynchronizedCapturingSink>();
  auto* sink_ptr = sink.get();
  logging::LoggingManager manager(std::move(sink), logging::Severity::kINFO, false,
                                  logging::LoggingManager::InstanceType::Temporal);
  auto logger = manager.CreateLogger("moe_parallel_log_budget");
  ASSERT_STATUS_OK(state.BeginRun("request", logger.get()));
  InlinedVector<std::thread> workers;
  for (const auto* kernel : kernels_) {
    workers.emplace_back([this, &state, kernel]() {
      const int selected[] = {1};
      for (size_t i = 0; i < KernelPilotMoeExpertState::kMaxCounterLogRecordsPerRun; ++i) {
        ASSERT_STATUS_OK(CollectAndRecord(state, kernel, selected));
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  ASSERT_STATUS_OK(state.EndRun());
  size_t records = 0, summaries = 0;
  for (const auto& message : sink_ptr->Messages()) {
    records += message.find("moe_expert_counters ") != std::string::npos;
    summaries += message.find("moe_expert_counters_truncated ") != std::string::npos;
  }
  EXPECT_EQ(records, KernelPilotMoeExpertState::kMaxCounterLogRecordsPerRun);
  EXPECT_EQ(summaries, 1U);
  for (const auto* kernel : kernels_) {
    EXPECT_EQ(Counters(state, kernel), (InlinedVector<double>{0, 1}));
  }
}

TEST_F(KernelPilotMoeExpertStateTest, AppliesExponentialUpdateToEveryExpert) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.SetCounterAlpha(0.5));
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 3));
  std::istringstream initial(
      R"({"alpha":0.5,"epsilon":0.01,"experts":[)"
      R"({"moe":0,"expert":0,"popularity":4},)"
      R"({"moe":0,"expert":1,"popularity":2},)"
      R"({"moe":0,"expert":2,"popularity":1}]})");
  ASSERT_STATUS_OK(state.Load(initial));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  const int selected[] = {2, 0, 2};
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{2.5, 1, 1}));
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{1.75, 0.5, 1}));
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], {}));
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{0.875, 0.25, 0.5}));
}

TEST_F(KernelPilotMoeExpertStateTest, BlendsBinarySelectionWithRoutingProbability) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.SetCounterAlpha(0.5));
  ASSERT_STATUS_OK(state.SetCounterProbabilityWeight(0.25));
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 3));
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  const int selected[] = {0, 2};
  const float probabilities[] = {0.8f, 0.2f};
  ASSERT_STATUS_OK(CollectWeightedAndRecord(state, kernels_[0], selected, probabilities));
  const auto counters = Counters(state, kernels_[0]);
  ASSERT_EQ(counters.size(), 3U);
  EXPECT_NEAR(counters[0], 0.475, 1e-7);
  EXPECT_EQ(counters[1], 0);
  EXPECT_NEAR(counters[2], 0.4, 1e-7);
}

TEST_F(KernelPilotMoeExpertStateTest, AdaptiveSwapUsesStrictEpsilonAndDeterministicTies) {
  for (double epsilon : {1.0, 0.99}) {
    SCOPED_TRACE(epsilon);
    KernelPilotMoeExpertState state;
    ASSERT_STATUS_OK(state.SetCounterAlpha(0.5));
    ASSERT_STATUS_OK(state.SetSwapEpsilon(epsilon));
    ASSERT_STATUS_OK(state.SetCpuOffloadExpertCount(2));
    MakeEligibleForCudaFp16MoePlacement(kernels_[0]);
    ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 4));
    ASSERT_STATUS_OK(state.FinalizeInitialization());
    FakeMoeExpertCache cache;
    auto* pilot = state.GetKernelPilot(kernels_[0]);
    ASSERT_NE(pilot, nullptr);
    ASSERT_STATUS_OK(pilot->AttachMoeExpertCache(&cache));

    ASSERT_STATUS_OK(state.BeginRun("", nullptr));
    const int cuda_selected[] = {0, 1};
    ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], cuda_selected));
    ASSERT_STATUS_OK(state.EndRun());
    EXPECT_TRUE(cache.Swaps().empty());

    ASSERT_STATUS_OK(state.BeginRun("", nullptr));
    const int cpu_selected[] = {2, 3};
    ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], cpu_selected));
    ASSERT_STATUS_OK(state.EndRun());
    if (epsilon == 1.0) {
      EXPECT_TRUE(cache.Swaps().empty());
    } else {
      EXPECT_EQ(cache.Swaps(), (InlinedVector<std::pair<int, int>>{{0, 2}}));
    }
  }
}

TEST_F(KernelPilotMoeExpertStateTest, AdaptiveSwapHonorsPerDeviceLimit) {
  for (const size_t max_in_flight_swaps : {0U, 2U, 4U}) {
    SCOPED_TRACE(max_in_flight_swaps);
    KernelPilotMoeExpertState state;
    ASSERT_STATUS_OK(state.SetCounterAlpha(0));
    ASSERT_STATUS_OK(state.SetMaxInFlightSwapsPerDevice(max_in_flight_swaps));
    ASSERT_STATUS_OK(state.SetCpuOffloadExpertCount(3));
    for (size_t i = 0; i < 3; ++i) {
      MakeEligibleForCudaFp16MoePlacement(kernels_[i]);
      ASSERT_STATUS_OK(state.RegisterNode(kernels_[i], "main", i, "MoE", 2));
    }
    ASSERT_STATUS_OK(state.FinalizeInitialization());
    std::array<FakeMoeExpertCache, 3> caches;
    for (size_t i = 0; i < caches.size(); ++i) {
      ASSERT_STATUS_OK(state.GetKernelPilot(kernels_[i])->AttachMoeExpertCache(&caches[i]));
    }

    auto sink = std::make_unique<CapturingSink>();
    auto* sink_ptr = sink.get();
    logging::LoggingManager manager(std::move(sink), logging::Severity::kINFO, false,
                                    logging::LoggingManager::InstanceType::Temporal);
    auto logger = manager.CreateLogger("moe_swap_queue");
    ASSERT_STATUS_OK(state.BeginRun("request", logger.get()));
    const int selected[] = {1};
    for (const auto* kernel : kernels_) {
      ASSERT_STATUS_OK(CollectAndRecord(state, kernel, selected));
    }
    ASSERT_STATUS_OK(state.EndRun());
    const size_t expected_swap_count = std::min(max_in_flight_swaps, caches.size());
    for (size_t i = 0; i < caches.size(); ++i) {
      if (i < expected_swap_count) {
        EXPECT_EQ(caches[i].Swaps(), (InlinedVector<std::pair<int, int>>{{0, 1}}));
      } else {
        EXPECT_TRUE(caches[i].Swaps().empty());
      }
    }

    constexpr std::string_view marker{"moe_expert_swaps "};
    nlohmann::json event;
    bool found_swap_event = false;
    for (const auto& message : sink_ptr->Messages()) {
      const size_t marker_position = message.find(marker);
      if (marker_position != std::string::npos) {
        event = nlohmann::json::parse(message.substr(marker_position + marker.size()));
        found_swap_event = true;
        break;
      }
    }
    ASSERT_TRUE(found_swap_event);
    EXPECT_EQ(event["request_id"], "request");
    EXPECT_EQ(event["total_in_flight"], expected_swap_count);
    EXPECT_EQ(event["total_blocked_candidates"], caches.size() - expected_swap_count);
    ASSERT_EQ(event["scheduled_swaps"].size(), expected_swap_count);
    for (size_t i = 0; i < expected_swap_count; ++i) {
      EXPECT_EQ(event["scheduled_swaps"][i],
                nlohmann::json({
                    {"graph_scope", "main"},
                    {"node_index", i},
                    {"device_id", 0},
                    {"outgoing_cuda_expert", 0},
                    {"incoming_cpu_expert", 1},
                    {"outgoing_score", 0.0},
                    {"incoming_score", 1.0},
                }));
    }
    ASSERT_EQ(event["devices"].size(), 1U);
    EXPECT_EQ(event["devices"][0],
              nlohmann::json({
                  {"device_id", 0},
                  {"in_flight", expected_swap_count},
                  {"blocked_candidates", caches.size() - expected_swap_count},
              }));
  }
}

TEST_F(KernelPilotMoeExpertStateTest, ValidatesCounterParameters) {
  for (const double alpha : {
           -0.1,
           1.1,
           std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN()}) {
    SCOPED_TRACE(MakeString("alpha=", alpha));
    KernelPilotMoeExpertState state;
    EXPECT_FALSE(state.SetCounterAlpha(alpha).IsOK());
  }
  for (const double alpha : {0.0, 0.5, 0.9, 1.0}) {
    SCOPED_TRACE(MakeString("alpha=", alpha));
    KernelPilotMoeExpertState state;
    EXPECT_STATUS_OK(state.SetCounterAlpha(alpha));
  }
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 1));
  EXPECT_FALSE(state.SetCounterAlpha(0.5).IsOK());
}

TEST_F(KernelPilotMoeExpertStateTest, ValidatesCounterProbabilityWeight) {
  for (const double probability_weight : {
           -0.1,
           1.1,
           std::numeric_limits<double>::infinity(),
           std::numeric_limits<double>::quiet_NaN()}) {
    KernelPilotMoeExpertState state;
    EXPECT_FALSE(state.SetCounterProbabilityWeight(probability_weight).IsOK());
  }
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.SetCounterProbabilityWeight(0));
  ASSERT_STATUS_OK(state.SetCounterProbabilityWeight(0.5));
  ASSERT_STATUS_OK(state.SetCounterProbabilityWeight(1));
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 1));
  EXPECT_FALSE(state.SetCounterProbabilityWeight(0.5).IsOK());
}

TEST_F(KernelPilotMoeExpertStateTest, ValidatesSwapEpsilon) {
  for (double epsilon : {-0.1, std::numeric_limits<double>::infinity(),
                         std::numeric_limits<double>::quiet_NaN()}) {
    KernelPilotMoeExpertState state;
    EXPECT_FALSE(state.SetSwapEpsilon(epsilon).IsOK());
  }
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.SetSwapEpsilon(0));
  ASSERT_STATUS_OK(state.SetSwapEpsilon(1.5));
  ASSERT_STATUS_OK(state.SetMaxInFlightSwapsPerDevice(0));
  ASSERT_STATUS_OK(state.SetMaxInFlightSwapsPerDevice(8));
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 1));
  EXPECT_FALSE(state.SetSwapEpsilon(0.5).IsOK());
  EXPECT_FALSE(state.SetMaxInFlightSwapsPerDevice(2).IsOK());
}

TEST_F(KernelPilotMoeExpertStateTest, WritesFinalPopularityJsonOnDestruction) {
  const std::filesystem::path output_path{"moe_expert_popularity_test.json"};
  auto cleanup = gsl::finally([&output_path]() { std::filesystem::remove(output_path); });
  {
    KernelPilotMoeExpertState state;
    ASSERT_STATUS_OK(state.SetCounterAlpha(0.5));
    ASSERT_STATUS_OK(state.SetSwapEpsilon(0.125));
    ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 20, "MoE", 2));
    ASSERT_STATUS_OK(state.RegisterNode(kernels_[1], "main", 10, "MoE", 2));
    ASSERT_STATUS_OK(state.FinalizeInitialization());
    ASSERT_STATUS_OK(state.SetPopularityOutputFile(output_path.string()));
    const int selected[] = {1};
    ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  }

  std::ifstream input{output_path};
  ASSERT_TRUE(input.is_open());
  const auto output = nlohmann::json::parse(input);
  EXPECT_DOUBLE_EQ(output["alpha"], 0.5);
  EXPECT_DOUBLE_EQ(output["epsilon"], 0.125);
  ASSERT_EQ(output["experts"].size(), 4U);
  EXPECT_EQ(output["experts"][0], nlohmann::json({{"moe", 0}, {"expert", 0}, {"popularity", 0.0}}));
  EXPECT_EQ(output["experts"][1], nlohmann::json({{"moe", 0}, {"expert", 1}, {"popularity", 0.0}}));
  EXPECT_EQ(output["experts"][2], nlohmann::json({{"moe", 1}, {"expert", 0}, {"popularity", 0.0}}));
  EXPECT_EQ(output["experts"][3], nlohmann::json({{"moe", 1}, {"expert", 1}, {"popularity", 0.5}}));
}

TEST_F(KernelPilotMoeExpertStateTest, LoadedJsonParametersOverrideConfiguredValues) {
  const std::filesystem::path output_path{"moe_expert_loaded_parameters_test.json"};
  auto cleanup = gsl::finally([&output_path]() { std::filesystem::remove(output_path); });
  {
    KernelPilotMoeExpertState state;
    ASSERT_STATUS_OK(state.SetCounterAlpha(0.9));
    ASSERT_STATUS_OK(state.SetSwapEpsilon(0.9));
    ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 1));
    std::istringstream input(
        R"({"alpha":0.4,"epsilon":0.2,"experts":[{"moe":0,"expert":0,"popularity":0.75}]})");
    ASSERT_STATUS_OK(state.Load(input));
    ASSERT_STATUS_OK(state.FinalizeInitialization());
    EXPECT_DOUBLE_EQ(state.GetKernelPilot(kernels_[0])->MoeExpertCounterAlpha(), 0.4);
    ASSERT_STATUS_OK(state.SetPopularityOutputFile(output_path.string()));
  }

  std::ifstream input{output_path};
  ASSERT_TRUE(input.is_open());
  const auto output = nlohmann::json::parse(input);
  EXPECT_DOUBLE_EQ(output["alpha"], 0.4);
  EXPECT_DOUBLE_EQ(output["epsilon"], 0.2);
  ASSERT_EQ(output["experts"].size(), 1U);
  EXPECT_DOUBLE_EQ(output["experts"][0]["popularity"], 0.75);
}

TEST_F(KernelPilotMoeExpertStateTest, LoadsJsonStateAndValidatesAtomically) {
  KernelPilotMoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 2, "QMoE", 3));
  std::istringstream valid(
      R"({"alpha":0.5,"epsilon":0.2,"experts":[{"moe":0,"expert":1,"popularity":2.5}]})");
  ASSERT_STATUS_OK(state.Load(valid));
  InlinedVector<double> counters;
  for (const auto* invalid : {
           "",
           R"({"alpha":-1,"epsilon":0.2,"experts":[]})",
           R"({"alpha":0.5,"epsilon":-1,"experts":[]})",
           R"({"alpha":0.5,"epsilon":0.2,"experts":[{"moe":1,"expert":0,"popularity":1}]})",
           R"({"alpha":0.5,"epsilon":0.2,"experts":[{"moe":0,"expert":3,"popularity":1}]})",
           R"({"alpha":0.5,"epsilon":0.2,"experts":[{"moe":0,"expert":0,"popularity":-1}]})",
           R"({"alpha":0.5,"epsilon":0.2,"experts":[{"moe":0,"expert":0,"popularity":1},{"moe":0,"expert":0,"popularity":2}]})",
           R"({"alpha":0.5,"experts":[]})"}) {
    SCOPED_TRACE(invalid);
    std::istringstream input(invalid);
    EXPECT_FALSE(state.Load(input).IsOK());
    ASSERT_STATUS_OK(state.GetCounters(kernels_[0], counters));
    EXPECT_EQ(counters, (InlinedVector<double>{0, 2.5, 0}));
  }
  ASSERT_STATUS_OK(state.FinalizeInitialization());
  EXPECT_DOUBLE_EQ(state.GetKernelPilot(kernels_[0])->MoeExpertCounterAlpha(), 0.5);
  const int selected[] = {1, 1};
  ASSERT_STATUS_OK(CollectAndRecord(state, kernels_[0], selected));
  ASSERT_STATUS_OK(state.GetCounters(kernels_[0], counters));
  ASSERT_EQ(counters.size(), 3U);
  EXPECT_DOUBLE_EQ(counters[0], 0);
  EXPECT_DOUBLE_EQ(counters[1], 0.5 * 2.5 + 0.5);
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
  std::istringstream initial(R"({"alpha":0.5,"epsilon":0.1,"experts":[]})");
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
  EXPECT_EQ(Counters(state, kernels_[0]), (InlinedVector<double>{0, 1.0 - kDefaultCounterAlpha}));
}

TEST_F(KernelPilotMoeExpertStateTest, ValidCoefficientsKeepCountersBounded) {
  for (const double alpha : {0.0, 0.5, 0.9, 1.0}) {
    for (const double initial_value : {0.0, std::numeric_limits<double>::max()}) {
      SCOPED_TRACE(MakeString("alpha=", alpha, ", initial=", initial_value));
      KernelPilotMoeExpertState state;
      ASSERT_STATUS_OK(state.SetCounterAlpha(alpha));
      ASSERT_STATUS_OK(state.RegisterNode(kernels_[0], "main", 0, "MoE", 2));
      nlohmann::json initial_json{
          {"alpha", alpha},
          {"epsilon", 0.01},
          {"experts",
           {{{"moe", 0}, {"expert", 0}, {"popularity", initial_value}},
            {{"moe", 0}, {"expert", 1}, {"popularity", initial_value}}}}};
      std::istringstream initial(initial_json.dump());
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
    EXPECT_NEAR(counters[1], 1.0 - std::pow(kDefaultCounterAlpha, 100), 1e-14);
  }
  EXPECT_EQ(Counters(other, kernels_[0]), (InlinedVector<double>{0, 0}));
}

}  // namespace
#endif
}  // namespace onnxruntime::test
