// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_census.h"
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

TEST(DeviceCensusTest, SerializesSortedVersions) {
  telemetry_internal::DeviceCensusState state{
      telemetry_internal::kDeviceCensusSchemaVersion,
      "00000000-0000-0000-0000-000000000001",
      20700,
      false,
      {}};
  ASSERT_TRUE(telemetry_internal::AddDeviceCensusVersion(state, "1.25.0"));
  ASSERT_TRUE(telemetry_internal::AddDeviceCensusVersion(state, "1.24.0"));
  EXPECT_FALSE(telemetry_internal::AddDeviceCensusVersion(state, "1.24.0"));

  const std::string serialized =
      telemetry_internal::SerializeDeviceCensusState(state);
  const auto parsed = telemetry_internal::ParseDeviceCensusState(serialized);
  ASSERT_TRUE(parsed);
  EXPECT_EQ(parsed->schema_version,
            telemetry_internal::kDeviceCensusSchemaVersion);
  EXPECT_EQ(parsed->identity, state.identity);
  EXPECT_EQ(parsed->utc_day, 20700);
  EXPECT_FALSE(parsed->emitted);
  EXPECT_EQ(parsed->versions,
            (std::vector<std::string>{"1.24.0", "1.25.0"}));
}

TEST(DeviceCensusTest, PreservesUnknownSchemaVersion) {
  const auto parsed =
      telemetry_internal::ParseDeviceCensusState(
          "3\n00000000-0000-0000-0000-000000000001\n20700\n0\n1.24.0\n");
  ASSERT_TRUE(parsed);
  EXPECT_EQ(parsed->schema_version, 3);
}

TEST(DeviceCensusTest, RejectsInvalidVersion) {
  EXPECT_FALSE(telemetry_internal::IsValidDeviceCensusVersion("1.0.0\ninvalid"));
  EXPECT_TRUE(telemetry_internal::IsValidDeviceCensusVersion("1.0.0-rc.1+build"));
}

TEST(DeviceCensusTest, ResetsStateWhenIdentityChanges) {
  std::optional<telemetry_internal::DeviceCensusState> persisted;
  int emission_count = 0;
  ASSERT_TRUE(telemetry_internal::UpdateDeviceCensusState(
      {telemetry_internal::DeviceCensusState{
          telemetry_internal::kDeviceCensusSchemaVersion,
          "old-device",
          20700,
          false,
          {"1.0.0"}}},
      "new-device", 20700, "2.0.0", true,
      [&](const telemetry_internal::DeviceCensusState& state) {
        persisted = state;
        return true;
      },
      [&](int64_t, const std::vector<std::string>&) { ++emission_count; }));

  ASSERT_TRUE(persisted);
  EXPECT_EQ(persisted->identity, "new-device");
  EXPECT_EQ(persisted->versions, std::vector<std::string>{"2.0.0"});
  EXPECT_TRUE(persisted->emitted);
  EXPECT_EQ(emission_count, 1);
}

TEST(DeviceCensusTest, TreatsDuplicateSameDayActivityAsSuccess) {
  const telemetry_internal::DeviceCensusState existing{
      telemetry_internal::kDeviceCensusSchemaVersion,
      "device",
      20700,
      false,
      {"1.0.0"}};
  int persistence_count = 0;
  int emission_count = 0;
  EXPECT_TRUE(telemetry_internal::UpdateDeviceCensusState(
      existing, "device", 20700, "1.0.0", false,
      [&](const telemetry_internal::DeviceCensusState&) {
        ++persistence_count;
        return true;
      },
      [&](int64_t, const std::vector<std::string>&) { ++emission_count; }));
  EXPECT_EQ(persistence_count, 0);
  EXPECT_EQ(emission_count, 0);
}

TEST(DeviceCensusTest, PersistsRolloverBeforeEmittingCompletedDay) {
  const telemetry_internal::DeviceCensusState existing{
      telemetry_internal::kDeviceCensusSchemaVersion,
      "device",
      20700,
      false,
      {"1.0.0"}};
  std::optional<telemetry_internal::DeviceCensusState> persisted;
  bool observed_persisted_rollover = false;
  ASSERT_TRUE(telemetry_internal::UpdateDeviceCensusState(
      existing, "device", 20701, "2.0.0", false,
      [&](const telemetry_internal::DeviceCensusState& state) {
        persisted = state;
        return true;
      },
      [&](int64_t day, const std::vector<std::string>& versions) {
        observed_persisted_rollover =
            day == 20700 && versions == std::vector<std::string>{"1.0.0"} &&
            persisted && persisted->utc_day == 20701 &&
            persisted->versions == std::vector<std::string>{"2.0.0"};
      }));

  EXPECT_TRUE(observed_persisted_rollover);
}

}  // namespace
}  // namespace onnxruntime::test
