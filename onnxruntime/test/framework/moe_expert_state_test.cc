// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <limits>
#include <sstream>
#include <thread>

#include "gtest/gtest.h"
#include "core/framework/moe_expert_state.h"
#include "test/util/include/asserts.h"

namespace onnxruntime::test {

TEST(MoeExpertStateTest, CountsOncePerInvocationAndSeparatesNodes) {
  MoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode("main", 0, "MoE", 3));
  ASSERT_STATUS_OK(state.RegisterNode("main", 1, "QMoE", 3));
  ASSERT_STATUS_OK(state.RegisterNode("main/0/4:body", 0, "MoE", 3));
  EXPECT_EQ(state.TotalExpertCount(), 9U);
  const int selected[] = {2, 0, 2, 0};
  ASSERT_STATUS_OK(state.RecordUsage("main", 0, selected));
  ASSERT_STATUS_OK(state.RecordUsage("main", 0, selected));
  auto snapshot = state.GetSnapshot();
  EXPECT_EQ(snapshot.at({"main", 0}).counters, (InlinedVector<double>{2, 0, 2}));
  EXPECT_EQ(snapshot.at({"main", 1}).counters, (InlinedVector<double>{0, 0, 0}));
  EXPECT_EQ(snapshot.at({"main/0/4:body", 0}).counters, (InlinedVector<double>{0, 0, 0}));
  ASSERT_STATUS_OK(state.RecordUsage("main/0/4:body", 0, selected));
  EXPECT_EQ(snapshot.at({"main/0/4:body", 0}).counters, (InlinedVector<double>{0, 0, 0}));
  InlinedVector<double> counters;
  ASSERT_STATUS_OK(state.GetCounters("main/0/4:body", 0, counters));
  EXPECT_EQ(counters, (InlinedVector<double>{1, 0, 1}));
}

TEST(MoeExpertStateTest, AppliesExponentialUpdateToEveryExpert) {
  MoeExpertState state;
  ASSERT_STATUS_OK(state.SetCounterParameters(0.5, 2.0));
  ASSERT_STATUS_OK(state.RegisterNode("main", 0, "MoE", 3));
  std::istringstream initial(
      "moe_expert_state 1\n"
      "\"main\" 0 MoE 0 4\n"
      "\"main\" 0 MoE 1 2\n"
      "\"main\" 0 MoE 2 1\n");
  ASSERT_STATUS_OK(state.Load(initial));
  const int selected[] = {2, 0, 2};
  ASSERT_STATUS_OK(state.RecordUsage("main", 0, selected));
  EXPECT_EQ(state.GetSnapshot().at({"main", 0}).counters, (InlinedVector<double>{4, 1, 2.5}));
  ASSERT_STATUS_OK(state.RecordUsage("main", 0, selected));
  EXPECT_EQ(state.GetSnapshot().at({"main", 0}).counters, (InlinedVector<double>{4, 0.5, 3.25}));
}

TEST(MoeExpertStateTest, ValidatesCounterParameters) {
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
  ASSERT_STATUS_OK(state.RegisterNode("main", 0, "MoE", 1));
  EXPECT_FALSE(state.SetCounterParameters(0.5, 1.0).IsOK());
}

TEST(MoeExpertStateTest, LoadsPartialStateAndValidatesAtomically) {
  MoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode("main", 2, "QMoE", 3));
  std::istringstream valid("moe_expert_state 1\n\"main\" 2 QMoE 1 2.5\n");
  ASSERT_STATUS_OK(state.Load(valid));
  InlinedVector<double> counters;
  ASSERT_STATUS_OK(state.GetCounters("main", 2, counters));
  EXPECT_EQ(counters, (InlinedVector<double>{0, 2.5, 0}));
  const int selected[] = {1, 1};
  ASSERT_STATUS_OK(state.RecordUsage("main", 2, selected));
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
    ASSERT_STATUS_OK(state.GetCounters("main", 2, counters));
    EXPECT_EQ(counters, (InlinedVector<double>{0, 3.5, 0}));
  }
}

TEST(MoeExpertStateTest, RejectsInvalidUpdatesWithoutPartialCounts) {
  MoeExpertState state;
  ASSERT_STATUS_OK(state.RegisterNode("main", 0, "MoE", 2));
  EXPECT_FALSE(state.RegisterNode("main", 0, "MoE", 2).IsOK());
  EXPECT_FALSE(state.RegisterNode("main", 1, "Add", 2).IsOK());
  EXPECT_FALSE(state.RegisterNode("main", 1, "QMoE", 0).IsOK());
  const int out_of_bounds[] = {0, 2};
  const int negative[] = {0, -1};
  EXPECT_FALSE(state.RecordUsage("main", 0, out_of_bounds).IsOK());
  EXPECT_FALSE(state.RecordUsage("main", 0, negative).IsOK());
  EXPECT_FALSE(state.RecordUsage("main", 1, {}).IsOK());
  EXPECT_EQ(state.GetSnapshot().at({"main", 0}).counters, (InlinedVector<double>{0, 0}));
}

TEST(MoeExpertStateTest, ConcurrentUpdatesAndSessionIsolation) {
  MoeExpertState state, other;
  ASSERT_STATUS_OK(state.RegisterNode("main", 0, "MoE", 2));
  ASSERT_STATUS_OK(other.RegisterNode("main", 0, "MoE", 2));
  InlinedVector<std::thread> workers;
  for (int i = 0; i < 4; ++i) {
    workers.emplace_back([&state]() {
      const int selected[] = {1, 1};
      for (int j = 0; j < 100; ++j) {
        ASSERT_STATUS_OK(state.RecordUsage("main", 0, selected));
        const auto snapshot = state.GetSnapshot();
        EXPECT_EQ(snapshot.at({"main", 0}).counters[0], 0);
      }
    });
  }
  for (auto& worker : workers) {
    worker.join();
  }
  EXPECT_EQ(state.GetSnapshot().at({"main", 0}).counters, (InlinedVector<double>{0, 400}));
  EXPECT_EQ(other.GetSnapshot().at({"main", 0}).counters, (InlinedVector<double>{0, 0}));
}

}  // namespace onnxruntime::test
