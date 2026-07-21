// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <limits>

#include "core/platform/posix/telemetry_sampling.h"

namespace onnxruntime::test {

TEST(TelemetrySamplingTest, HandlesRateBoundaries) {
  EXPECT_FALSE(telemetry_internal::ShouldSampleSession("process-a", 1, -1.0));
  EXPECT_FALSE(telemetry_internal::ShouldSampleSession("process-a", 1, 0.0));
  EXPECT_FALSE(telemetry_internal::ShouldSampleSession(
      "process-a", 1, std::numeric_limits<double>::quiet_NaN()));
  EXPECT_TRUE(telemetry_internal::ShouldSampleSession("process-a", 1, 100.0));
  EXPECT_TRUE(telemetry_internal::ShouldSampleSession("process-a", 1, 101.0));
}

TEST(TelemetrySamplingTest, DecisionIsStableForCorrelatedEvents) {
  const bool session_start = telemetry_internal::ShouldSampleSession("process-a", 42, 10.0);
  const bool model_load = telemetry_internal::ShouldSampleSession("process-a", 42, 10.0);
  const bool evaluation_start = telemetry_internal::ShouldSampleSession("process-a", 42, 10.0);
  const bool evaluation_stop = telemetry_internal::ShouldSampleSession("process-a", 42, 10.0);

  EXPECT_EQ(session_start, model_load);
  EXPECT_EQ(session_start, evaluation_start);
  EXPECT_EQ(session_start, evaluation_stop);
}

TEST(TelemetrySamplingTest, TenPercentRateHasExpectedDistribution) {
  int sampled = 0;
  constexpr int kSessionCount = 100000;
  for (uint32_t session_id = 0; session_id < kSessionCount; ++session_id) {
    if (telemetry_internal::ShouldSampleSession("process-a", session_id, 10.0)) {
      ++sampled;
    }
  }

  EXPECT_GT(sampled, 9700);
  EXPECT_LT(sampled, 10300);
}

TEST(TelemetrySamplingTest, ProcessGuidAffectsDecision) {
  bool found_difference = false;
  for (uint32_t session_id = 0; session_id < 1000; ++session_id) {
    if (telemetry_internal::ShouldSampleSession("process-a", session_id, 10.0) !=
        telemetry_internal::ShouldSampleSession("process-b", session_id, 10.0)) {
      found_difference = true;
      break;
    }
  }

  EXPECT_TRUE(found_difference);
}

}  // namespace onnxruntime::test
