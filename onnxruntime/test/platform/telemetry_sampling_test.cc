// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry_sampling.h"

#include <array>
#include <cmath>

#include "gtest/gtest.h"

namespace onnxruntime::test {
namespace {

TEST(TelemetrySamplingTest, HonorsBoundaryRates) {
  EXPECT_FALSE(telemetry_internal::ShouldSampleSession("guid", 1, 0.0));
  EXPECT_TRUE(telemetry_internal::ShouldSampleSession("guid", 1, 100.0));
}

TEST(TelemetrySamplingTest, SessionDecisionsAreStableAndSampleExpectedFraction) {
  constexpr uint32_t session_count = 100000;
  constexpr std::string_view guid = "00000000-0000-0000-0000-000000000001";
  constexpr std::array rates{
      telemetry_internal::kModelSessionSampleRatePercent,
      telemetry_internal::kHighVolumeEventSampleRatePercent,
      telemetry_internal::kOtherProcessEventSampleRatePercent,
  };
  for (const double rate : rates) {
    SCOPED_TRACE(rate);
    uint32_t sampled_count = 0;
    for (uint32_t session_id = 0; session_id < session_count; ++session_id) {
      const bool sampled = telemetry_internal::ShouldSampleSession(guid, session_id, rate);
      ASSERT_EQ(telemetry_internal::ShouldSampleSession(guid, session_id, rate), sampled) << session_id;
      ASSERT_EQ(telemetry_internal::ShouldSampleSession(guid, session_id),
                telemetry_internal::ShouldSampleSession(
                    guid, session_id, telemetry_internal::kModelSessionSampleRatePercent))
          << session_id;
      sampled_count += sampled ? 1 : 0;
    }

    const double probability = rate / 100.0;
    const double expected_count = session_count * probability;
    const double tolerance = 6.0 * std::sqrt(expected_count * (1.0 - probability));
    EXPECT_NEAR(sampled_count, expected_count, tolerance);
  }
}

}  // namespace
}  // namespace onnxruntime::test
