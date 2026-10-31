// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Run bookkeeping of QMoESessionController. An invocation record stands for one logical execution of an offload node.
// These tests drive the records directly, so they need neither a model nor a stream nor a GPU.

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include "gtest/gtest.h"

#include "core/common/status.h"
#include "core/providers/cuda/qmoe_session_controller.h"

namespace onnxruntime {
namespace cuda {
namespace test {
namespace {

std::shared_ptr<QMoEInvocationRecord> MakeRecord(const std::string& node_name) {
  auto record = std::make_shared<QMoEInvocationRecord>();
  record->node_name = node_name;
  return record;
}

void StageObservation(QMoEInvocationRecord& record, int* commits) {
  std::lock_guard<std::mutex> lock(record.mutex);
  record.commit_observation = [commits]() { ++*commits; };
}

class QMoESessionControllerRunTest : public ::testing::Test {
 protected:
  void SetUp() override { controller_.RegisterOffloadNode(); }

  void StartRun() {
    const common::Status status = controller_.OnRunStart();
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  }

  common::Status EndRun(bool run_succeeded = true) {
    return controller_.OnRunEnd(/*sync_stream=*/true, run_succeeded);
  }

  QMoESessionController controller_{/*device_id=*/0};
};

}  // namespace

// A run that captures a CUDA graph registers the record while capturing and again when the captured graph is replayed
// right away. No controller run starts in between, and the replay is what produces the observation.
TEST_F(QMoESessionControllerRunTest, CaptureAndImmediateReplayCommitOneObservation) {
  StartRun();
  auto record = MakeRecord("moe_0");
  controller_.RegisterInvocation(record);  // while capturing
  controller_.RegisterInvocation(record);  // when the captured graph is replayed

  int commits = 0;
  StageObservation(*record, &commits);  // the CPU side records it while the replay runs

  const common::Status status = EndRun();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(commits, 1);
}

TEST_F(QMoESessionControllerRunTest, CaptureAndImmediateReplayDrainTheDeviceOnce) {
  StartRun();
  auto record = MakeRecord("moe_0");
  int drains = 0;
  record->drain_device = [&drains]() {
    ++drains;
    return common::Status::OK();
  };
  controller_.RegisterInvocation(record);
  controller_.RegisterInvocation(record);

  const common::Status status = EndRun();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(drains, 1);
}

TEST_F(QMoESessionControllerRunTest, ReRegistrationKeepsTheFirstInvocationId) {
  StartRun();
  auto record = MakeRecord("moe_0");
  controller_.RegisterInvocation(record);
  const uint64_t invocation_id = record->invocation_id;
  const uint64_t run_id = record->run_id;
  EXPECT_NE(invocation_id, 0u);

  controller_.RegisterInvocation(record);
  EXPECT_EQ(record->invocation_id, invocation_id);
  EXPECT_EQ(record->run_id, run_id);
  ASSERT_TRUE(EndRun().IsOK());
}

TEST_F(QMoESessionControllerRunTest, DistinctInvocationsEachCommitTheirOwnObservation) {
  StartRun();
  auto first = MakeRecord("moe_0");
  auto second = MakeRecord("moe_1");
  int first_commits = 0;
  int second_commits = 0;
  controller_.RegisterInvocation(first);
  controller_.RegisterInvocation(second);
  controller_.RegisterInvocation(first);
  controller_.RegisterInvocation(second);
  StageObservation(*first, &first_commits);
  StageObservation(*second, &second_commits);
  EXPECT_NE(first->invocation_id, second->invocation_id);

  const common::Status status = EndRun();
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  EXPECT_EQ(first_commits, 1);
  EXPECT_EQ(second_commits, 1);
}

// Replaying a captured graph reuses one record per node, so each run has to register it again.
TEST_F(QMoESessionControllerRunTest, RecordReusedAcrossRunsContributesOncePerRun) {
  auto record = MakeRecord("moe_0");
  int commits = 0;
  uint64_t previous_run_id = 0;
  for (int run = 1; run <= 3; ++run) {
    StartRun();
    controller_.RegisterInvocation(record);
    StageObservation(*record, &commits);
    EXPECT_GT(record->run_id, previous_run_id);
    previous_run_id = record->run_id;

    const common::Status status = EndRun();
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    EXPECT_EQ(commits, run);
  }
}

TEST_F(QMoESessionControllerRunTest, ObservationOfAnUnsuccessfulRunIsDiscardedAndNotCarriedOver) {
  auto record = MakeRecord("moe_0");
  int commits = 0;

  StartRun();
  controller_.RegisterInvocation(record);
  StageObservation(*record, &commits);
  ASSERT_TRUE(EndRun(/*run_succeeded=*/false).IsOK());
  EXPECT_EQ(commits, 0);

  // The next run registers the same record without staging anything. It must not inherit the discarded observation.
  StartRun();
  controller_.RegisterInvocation(record);
  ASSERT_TRUE(EndRun().IsOK());
  EXPECT_EQ(commits, 0);
}

TEST_F(QMoESessionControllerRunTest, FailedInvocationDiscardsTheObservationsOfTheWholeRun) {
  StartRun();
  auto healthy = MakeRecord("moe_0");
  auto failed = MakeRecord("moe_1");
  int healthy_commits = 0;
  int failed_commits = 0;
  controller_.RegisterInvocation(healthy);
  controller_.RegisterInvocation(failed);
  StageObservation(*healthy, &healthy_commits);
  StageObservation(*failed, &failed_commits);
  failed->MarkFailed("injected failure");

  const common::Status status = EndRun();
  EXPECT_FALSE(status.IsOK());
  EXPECT_NE(status.ErrorMessage().find("injected failure"), std::string::npos) << status.ErrorMessage();
  EXPECT_EQ(healthy_commits, 0);
  EXPECT_EQ(failed_commits, 0);
}

namespace {

// The lookup a Run's config entries provide.
auto RunEntries(std::map<std::string, std::string> entries) {
  return [entries = std::move(entries)](const char* key, std::string& value) {
    const auto it = entries.find(key);
    if (it == entries.end()) return false;
    value = it->second;
    return true;
  };
}

}  // namespace

TEST_F(QMoESessionControllerRunTest, RunControlsAreScopedToTheRunThatCarriesThem) {
  StartRun();
  ASSERT_TRUE(controller_.ApplyRunControls(RunEntries({{kQMoERunInWarmupConfig, "1"},
                                                       {kQMoERunPrefillNewTurnConfig, "1"},
                                                       {kQMoERunPrefillTotalTokensConfig, "512"}}))
                  .IsOK());
  EXPECT_TRUE(controller_.RunControls().InWarmup());
  EXPECT_EQ(controller_.RunControls().PrefillTotalTokens(0), 512);

  ASSERT_TRUE(EndRun().IsOK());
  EXPECT_FALSE(controller_.RunControls().InWarmup());
  EXPECT_FALSE(controller_.RunControls().ConsumePrefillNewTurn());
  EXPECT_EQ(controller_.RunControls().PrefillTotalTokens(0), 512);  // the total persists on the session
}

TEST_F(QMoESessionControllerRunTest, RunControlsAreClearedWhenTheRunFails) {
  StartRun();
  ASSERT_TRUE(controller_.ApplyRunControls(RunEntries({{kQMoERunInWarmupConfig, "1"}})).IsOK());
  ASSERT_TRUE(EndRun(/*run_succeeded=*/false).IsOK());
  EXPECT_FALSE(controller_.RunControls().InWarmup());
}

TEST_F(QMoESessionControllerRunTest, MalformedRunControlFailsTheRunAndLeavesTheSessionUsable) {
  StartRun();
  const common::Status status =
      controller_.ApplyRunControls(RunEntries({{kQMoERunPrefillTotalTokensConfig, "many"}, {kQMoERunInWarmupConfig, "1"}}));
  ASSERT_FALSE(status.IsOK());
  EXPECT_EQ(status.Code(), common::INVALID_ARGUMENT);
  EXPECT_NE(status.ErrorMessage().find(kQMoERunPrefillTotalTokensConfig), std::string::npos) << status.ErrorMessage();
  EXPECT_FALSE(controller_.RunControls().InWarmup()) << "a rejected set of entries applies nothing";

  // The execution provider ends the controller run when OnRunStart fails after the controller started it.
  ASSERT_TRUE(EndRun(/*run_succeeded=*/false).IsOK());
  StartRun();
  ASSERT_TRUE(EndRun().IsOK());
}

TEST_F(QMoESessionControllerRunTest, HandoffRequestIsPendingOnlyAfterTheRunCompletes) {
  StartRun();
  ASSERT_TRUE(controller_.ApplyRunControls(RunEntries({{kQMoERunPrefillHandoffDoneConfig, "1"}})).IsOK());
  EXPECT_FALSE(controller_.RunControls().HandoffDonePending());

  ASSERT_TRUE(EndRun().IsOK());
  EXPECT_TRUE(controller_.RunControls().HandoffDonePending());
  EXPECT_TRUE(controller_.RunControls().ConsumeHandoffDone());
  EXPECT_FALSE(controller_.RunControls().HandoffDonePending());
}

TEST_F(QMoESessionControllerRunTest, HandoffRequestIsDroppedWhenTheRunFails) {
  StartRun();
  ASSERT_TRUE(controller_.ApplyRunControls(RunEntries({{kQMoERunPrefillHandoffDoneConfig, "1"}})).IsOK());
  ASSERT_TRUE(EndRun(/*run_succeeded=*/false).IsOK());
  EXPECT_FALSE(controller_.RunControls().HandoffDonePending());
}

TEST_F(QMoESessionControllerRunTest, HandoffRequestIsDroppedWhenAnInvocationFails) {
  StartRun();
  auto record = MakeRecord("moe_0");
  controller_.RegisterInvocation(record);
  ASSERT_TRUE(controller_.ApplyRunControls(RunEntries({{kQMoERunPrefillHandoffDoneConfig, "1"}})).IsOK());
  record->MarkFailed("injected failure");

  EXPECT_FALSE(EndRun().IsOK());
  EXPECT_FALSE(controller_.RunControls().HandoffDonePending());
}

TEST_F(QMoESessionControllerRunTest, RunControlsOfOneSessionDoNotReachAnotherSession) {
  QMoESessionController other{/*device_id=*/0};
  other.RegisterOffloadNode();

  StartRun();
  ASSERT_TRUE(controller_.ApplyRunControls(RunEntries({{kQMoERunInWarmupConfig, "1"},
                                                       {kQMoERunPrefillNewTurnConfig, "1"},
                                                       {kQMoERunPrefillTotalTokensConfig, "64"},
                                                       {kQMoERunPrefillHandoffDoneConfig, "1"}}))
                  .IsOK());
  EXPECT_FALSE(other.RunControls().InWarmup());
  EXPECT_FALSE(other.RunControls().ConsumePrefillNewTurn());
  EXPECT_EQ(other.RunControls().PrefillTotalTokens(0), 0);

  ASSERT_TRUE(EndRun().IsOK());
  EXPECT_FALSE(other.RunControls().ConsumeHandoffDone());
  EXPECT_TRUE(controller_.RunControls().ConsumeHandoffDone());
}

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
