// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <climits>
#include <map>
#include <string>

#include "core/providers/cuda/qmoe_run_controls.h"
#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

namespace {

using RunEntries = std::map<std::string, std::string>;

bool Apply(cuda::QMoERunControls& controls, const RunEntries& entries, std::string* error = nullptr) {
  std::string local_error;
  const bool ok = controls.Apply(
      [&entries](const char* key, std::string& value) {
        const auto it = entries.find(key);
        if (it == entries.end()) return false;
        value = it->second;
        return true;
      },
      error != nullptr ? *error : local_error);
  return ok;
}

}  // namespace

TEST(QMoERunControlsTest, DefaultsAreInactive) {
  cuda::QMoERunControls controls;
  EXPECT_FALSE(controls.InWarmup());
  EXPECT_FALSE(controls.ConsumePrefillNewTurn());
  EXPECT_FALSE(controls.HandoffDonePending());
  EXPECT_FALSE(controls.ConsumeHandoffDone());
  EXPECT_EQ(controls.PrefillTotalTokens(/*default_value=*/7), 7);
}

TEST(QMoERunControlsTest, RunWithoutEntriesChangesNothing) {
  cuda::QMoERunControls controls;
  ASSERT_TRUE(Apply(controls, {}));
  EXPECT_FALSE(controls.InWarmup());
  EXPECT_FALSE(controls.ConsumePrefillNewTurn());
  EXPECT_EQ(controls.PrefillTotalTokens(5), 5);
  controls.EndRun(/*run_completed=*/true);
  EXPECT_FALSE(controls.HandoffDonePending());
}

TEST(QMoERunControlsTest, WarmupAppliesToThatRunOnly) {
  cuda::QMoERunControls controls;
  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunInWarmupConfig, "1"}}));
  EXPECT_TRUE(controls.InWarmup());
  controls.EndRun(true);
  EXPECT_FALSE(controls.InWarmup());

  // A Run that does not repeat the entry is not a warm-up Run, even when the previous Run never reached EndRun.
  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunInWarmupConfig, "1"}}));
  ASSERT_TRUE(Apply(controls, {}));
  EXPECT_FALSE(controls.InWarmup());
}

TEST(QMoERunControlsTest, WarmupIsClearedWhenTheRunFails) {
  cuda::QMoERunControls controls;
  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunInWarmupConfig, "1"}}));
  controls.EndRun(/*run_completed=*/false);
  EXPECT_FALSE(controls.InWarmup());
}

TEST(QMoERunControlsTest, NewTurnIsConsumedExactlyOnce) {
  cuda::QMoERunControls controls;
  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunPrefillNewTurnConfig, "1"}}));
  EXPECT_TRUE(controls.ConsumePrefillNewTurn());
  EXPECT_FALSE(controls.ConsumePrefillNewTurn());
}

TEST(QMoERunControlsTest, UnconsumedNewTurnDoesNotLeakIntoTheNextRun) {
  cuda::QMoERunControls controls;
  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunPrefillNewTurnConfig, "1"}}));
  controls.EndRun(true);
  EXPECT_FALSE(controls.ConsumePrefillNewTurn());

  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunPrefillNewTurnConfig, "1"}}));
  ASSERT_TRUE(Apply(controls, {}));
  EXPECT_FALSE(controls.ConsumePrefillNewTurn());
}

TEST(QMoERunControlsTest, TotalTokensPersistUntilReplaced) {
  cuda::QMoERunControls controls;
  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunPrefillTotalTokensConfig, "4096"}}));
  EXPECT_EQ(controls.PrefillTotalTokens(0), 4096);
  controls.EndRun(true);
  ASSERT_TRUE(Apply(controls, {}));
  EXPECT_EQ(controls.PrefillTotalTokens(0), 4096);
  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunPrefillTotalTokensConfig, "0"}}));
  EXPECT_EQ(controls.PrefillTotalTokens(99), 0);
}

TEST(QMoERunControlsTest, TotalTokensAcceptsTheInt32Range) {
  cuda::QMoERunControls controls;
  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunPrefillTotalTokensConfig, std::to_string(INT_MAX)}}));
  EXPECT_EQ(controls.PrefillTotalTokens(0), INT_MAX);
}

TEST(QMoERunControlsTest, MalformedTotalTokensAreRejected) {
  const char* const bad_values[] = {"", "-1", "abc", "12x", " 12", "12 ", "+5", "1.5", "2147483648", "99999999999999999999"};
  for (const char* bad : bad_values) {
    cuda::QMoERunControls controls;
    std::string error;
    EXPECT_FALSE(Apply(controls, {{cuda::kQMoERunPrefillTotalTokensConfig, bad}}, &error)) << "value '" << bad << "'";
    EXPECT_NE(error.find(cuda::kQMoERunPrefillTotalTokensConfig), std::string::npos) << error;
    EXPECT_EQ(controls.PrefillTotalTokens(-5), -5) << "a rejected value must not be stored: '" << bad << "'";
  }
}

TEST(QMoERunControlsTest, FlagsAcceptOnlyZeroAndOne) {
  const char* const flag_keys[] = {cuda::kQMoERunInWarmupConfig, cuda::kQMoERunPrefillNewTurnConfig,
                                   cuda::kQMoERunPrefillHandoffDoneConfig};
  const char* const bad_values[] = {"", "2", "-1", "true", "false", "yes", "01", "1 "};
  for (const char* key : flag_keys) {
    for (const char* bad : bad_values) {
      cuda::QMoERunControls controls;
      std::string error;
      EXPECT_FALSE(Apply(controls, {{key, bad}}, &error)) << key << "='" << bad << "'";
      EXPECT_NE(error.find(key), std::string::npos) << error;
    }
    cuda::QMoERunControls controls;
    EXPECT_TRUE(Apply(controls, {{key, "0"}})) << key;
    EXPECT_TRUE(Apply(controls, {{key, "1"}})) << key;
  }
}

TEST(QMoERunControlsTest, MalformedEntryAppliesNothing) {
  cuda::QMoERunControls controls;
  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunPrefillTotalTokensConfig, "100"}}));

  const RunEntries mixed = {
      {cuda::kQMoERunInWarmupConfig, "1"},
      {cuda::kQMoERunPrefillNewTurnConfig, "1"},
      {cuda::kQMoERunPrefillHandoffDoneConfig, "1"},
      {cuda::kQMoERunPrefillTotalTokensConfig, "not-a-number"},
  };
  EXPECT_FALSE(Apply(controls, mixed));
  EXPECT_FALSE(controls.InWarmup());
  EXPECT_FALSE(controls.ConsumePrefillNewTurn());
  controls.EndRun(true);
  EXPECT_FALSE(controls.HandoffDonePending());
  EXPECT_EQ(controls.PrefillTotalTokens(0), 100);
}

TEST(QMoERunControlsTest, HandoffBecomesPendingOnlyWhenTheRunCompletes) {
  cuda::QMoERunControls controls;
  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunPrefillHandoffDoneConfig, "1"}}));
  EXPECT_FALSE(controls.HandoffDonePending()) << "the hand-off is requested by this Run, not yet pending";
  controls.EndRun(/*run_completed=*/true);
  EXPECT_TRUE(controls.HandoffDonePending());
  EXPECT_TRUE(controls.ConsumeHandoffDone());
  EXPECT_FALSE(controls.ConsumeHandoffDone());
  EXPECT_FALSE(controls.HandoffDonePending());
}

TEST(QMoERunControlsTest, FailedRunDoesNotArmTheHandoff) {
  cuda::QMoERunControls controls;
  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunPrefillHandoffDoneConfig, "1"}}));
  controls.EndRun(/*run_completed=*/false);
  EXPECT_FALSE(controls.HandoffDonePending());

  // The failed Run's request is gone; a later successful Run without the entry does not arm it either.
  ASSERT_TRUE(Apply(controls, {}));
  controls.EndRun(true);
  EXPECT_FALSE(controls.HandoffDonePending());
}

TEST(QMoERunControlsTest, PendingHandoffSurvivesLaterRunsUntilConsumed) {
  cuda::QMoERunControls controls;
  ASSERT_TRUE(Apply(controls, {{cuda::kQMoERunPrefillHandoffDoneConfig, "1"}}));
  controls.EndRun(true);
  ASSERT_TRUE(Apply(controls, {}));
  controls.EndRun(true);
  EXPECT_TRUE(controls.HandoffDonePending());
  EXPECT_TRUE(controls.ConsumeHandoffDone());
}

TEST(QMoERunControlsTest, ControlsOfOneSessionAreInvisibleToAnother) {
  cuda::QMoERunControls session_a;
  cuda::QMoERunControls session_b;
  ASSERT_TRUE(Apply(session_a, {{cuda::kQMoERunInWarmupConfig, "1"},
                                {cuda::kQMoERunPrefillNewTurnConfig, "1"},
                                {cuda::kQMoERunPrefillTotalTokensConfig, "1234"},
                                {cuda::kQMoERunPrefillHandoffDoneConfig, "1"}}));
  session_a.EndRun(true);

  EXPECT_FALSE(session_b.InWarmup());
  EXPECT_FALSE(session_b.ConsumePrefillNewTurn());
  EXPECT_FALSE(session_b.ConsumeHandoffDone());
  EXPECT_EQ(session_b.PrefillTotalTokens(0), 0);
  EXPECT_TRUE(session_a.ConsumeHandoffDone());
}

}  // namespace test
}  // namespace onnxruntime
