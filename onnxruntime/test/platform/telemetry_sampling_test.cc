// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/telemetry_sampling.h"

#include "gtest/gtest.h"

namespace onnxruntime::test {
namespace {

TEST(TelemetrySamplingTest, SessionDecisionIsStable) {
  constexpr std::string_view app_session_guid = "00000000-0000-0000-0000-000000000001";
  constexpr uint32_t session_id = 42;

  const bool decision = telemetry_internal::ShouldSampleSession(
      app_session_guid, session_id, telemetry_internal::kHighVolumeEventSampleRatePercent);
  for (int i = 0; i < 100; ++i) {
    EXPECT_EQ(telemetry_internal::ShouldSampleSession(
                  app_session_guid, session_id,
                  telemetry_internal::kHighVolumeEventSampleRatePercent),
              decision);
  }
}

TEST(TelemetrySamplingTest, HonorsBoundaryRates) {
  EXPECT_FALSE(telemetry_internal::ShouldSampleSession("guid", 1, 0.0));
  EXPECT_TRUE(telemetry_internal::ShouldSampleSession("guid", 1, 100.0));
}

TEST(TelemetrySamplingTest, UsesOnePercentRates) {
  EXPECT_EQ(telemetry_internal::kModelSessionSampleRatePercent, 1.0);
  EXPECT_EQ(telemetry_internal::kHighVolumeEventSampleRatePercent, 1.0);
  EXPECT_EQ(telemetry_internal::kProcessEventSampleRatePercent, 1.0);
}

TEST(TelemetrySamplingTest, HighVolumeRateSamplesExpectedFraction) {
  constexpr uint32_t session_count = 100000;
  uint32_t sampled_count = 0;
  for (uint32_t session_id = 0; session_id < session_count; ++session_id) {
    sampled_count += telemetry_internal::ShouldSampleSession(
                         "00000000-0000-0000-0000-000000000001", session_id,
                         telemetry_internal::kHighVolumeEventSampleRatePercent)
                         ? 1
                         : 0;
  }

  EXPECT_GT(sampled_count, 900u);
  EXPECT_LT(sampled_count, 1100u);
}

}  // namespace
}  // namespace onnxruntime::test
