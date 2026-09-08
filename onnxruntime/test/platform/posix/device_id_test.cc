// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/device_id.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>

#include "gtest/gtest.h"

#include "core/platform/device_census.h"
#include "core/platform/telemetry_guid.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime::test {
namespace {

namespace fs = std::filesystem;

class ScopedTestDirectory {
 public:
  explicit ScopedTestDirectory(std::string_view name)
      : path_(fs::temp_directory_path() /
              (std::string{"ort_device_id_"} + std::string{name} + "_" + GenerateGuidV4())) {
    fs::create_directories(path_);
  }

  ~ScopedTestDirectory() {
    std::error_code error;
    fs::remove_all(path_, error);
  }

  const fs::path& Path() const { return path_; }

 private:
  fs::path path_;
};

#if !defined(__APPLE__)

TEST(DeviceIdTest, UsesAbsoluteXdgCacheHomeWithoutHome) {
  ScopedTestDirectory test_dir{"absolute_xdg"};
  const fs::path cache_home = test_dir.Path() / "cache";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"HOME", nullopt}, {"XDG_CACHE_HOME", cache_home.string()}}};

  EXPECT_EQ(fs::path(DeviceId::GetStorageDirectory()),
            cache_home / "Microsoft" / "DeveloperTools" / ".onnxruntime");
}

TEST(DeviceIdTest, IgnoresRelativeXdgCacheHome) {
  ScopedTestDirectory test_dir{"relative_xdg"};
  const fs::path home = test_dir.Path() / "home";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"HOME", home.string()}, {"XDG_CACHE_HOME", "relative-cache"}}};

  EXPECT_EQ(fs::path(DeviceId::GetStorageDirectory()),
            home / ".cache" / "Microsoft" / "DeveloperTools" / ".onnxruntime");
}

#endif

TEST(DeviceIdDeathTest, RejectsSymlinkedOwnedDirectoryBeforeReading) {
  ScopedTestDirectory test_dir{"symlink_leaf"};
  const fs::path home = test_dir.Path() / "home";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"HOME", home.string()}, {"XDG_CACHE_HOME", nullopt}}};

  const fs::path storage_dir = DeviceId::GetStorageDirectory();
  const fs::path redirected_dir = test_dir.Path() / "redirected";
  fs::create_directories(storage_dir.parent_path());
  fs::create_directories(redirected_dir);

  constexpr std::string_view kRedirectedId = "11111111-2222-4333-8444-555555555555";
  std::ofstream(redirected_dir / "deviceid") << kRedirectedId;
  fs::create_directory_symlink(redirected_dir, storage_dir);

  EXPECT_EXIT(
      {
        const DeviceIdStatus status = DeviceId::Instance().GetStatus();
        const std::string value = DeviceId::Instance().GetValue();
        std::_Exit(status == DeviceIdStatus::Failed && value != kRedirectedId
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(DeviceIdDeathTest, RepairsCorruptedFile) {
  ScopedTestDirectory test_dir{"corrupted"};
  const fs::path home = test_dir.Path() / "home";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"HOME", home.string()}, {"XDG_CACHE_HOME", nullopt}}};

  const fs::path storage_dir = DeviceId::GetStorageDirectory();
  fs::create_directories(storage_dir);
  std::ofstream(storage_dir / "deviceid") << "corrupted";

  EXPECT_EXIT(
      {
        const std::string value = DeviceId::Instance().GetValue();
        const DeviceIdStatus status = DeviceId::Instance().GetStatus();
        std::ifstream input(storage_dir / "deviceid");
        std::string persisted;
        input >> persisted;
        std::_Exit(status == DeviceIdStatus::Corrupted && value == persisted
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");

  std::ifstream input(storage_dir / "deviceid");
  std::string persisted;
  input >> persisted;
  EXPECT_EQ(persisted.size(), 36u);
  EXPECT_NE(persisted, "corrupted");
}

TEST(DeviceIdDeathTest, AccumulatesAndEmitsCompletedCensusDay) {
  ScopedTestDirectory test_dir{"device_census"};
  const fs::path home = test_dir.Path() / "home";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"HOME", home.string()}, {"XDG_CACHE_HOME", nullopt}}};
  const fs::path census_state =
      fs::path(DeviceId::GetStorageDirectory()) / "devicecensus.state";

  EXPECT_EXIT(
      {
        int emission_count = 0;
        bool emitted_expected_day = false;
        bool emitted_expected_entries = false;
        auto emit = [&](int64_t census_day,
                        const std::vector<std::string>& versions) {
          std::ifstream state_file(census_state);
          std::string persisted((std::istreambuf_iterator<char>(state_file)),
                                std::istreambuf_iterator<char>());
          const auto persisted_state =
              telemetry_internal::ParseDeviceCensusState(persisted);
          if (!persisted_state || persisted_state->utc_day != 20707) {
            std::_Exit(EXIT_FAILURE);
          }
          ++emission_count;
          emitted_expected_day = census_day == 20700;
          emitted_expected_entries =
              (versions == std::vector<std::string>{"1.24.0", "1.25.0"});
        };
        auto& device_id = DeviceId::Instance();

        const bool first = device_id.RecordCensusActivity(20700, "1.24.0", false, emit);
        const bool duplicate = device_id.RecordCensusActivity(20700, "1.24.0", false, emit);
        const bool new_version = device_id.RecordCensusActivity(20700, "1.25.0", false, emit);
        const bool next_active_day = device_id.RecordCensusActivity(20707, "1.24.0", false, emit);

        std::_Exit(first && duplicate && new_version && next_active_day &&
                           emission_count == 1 &&
                           emitted_expected_day && emitted_expected_entries
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(DeviceIdDeathTest, EmitsImmediatelyForNewDeviceIdWithoutRolloverDuplicate) {
  ScopedTestDirectory test_dir{"device_census_first_use"};
  const fs::path home = test_dir.Path() / "home";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"HOME", home.string()}, {"XDG_CACHE_HOME", nullopt}}};
  const fs::path census_state =
      fs::path(DeviceId::GetStorageDirectory()) / "devicecensus.state";

  EXPECT_EXIT(
      {
        int emission_count = 0;
        int64_t emitted_day = -1;
        auto emit = [&](int64_t census_day,
                        const std::vector<std::string>& versions) {
          std::ifstream state_file(census_state);
          std::string persisted((std::istreambuf_iterator<char>(state_file)),
                                std::istreambuf_iterator<char>());
          const auto persisted_state =
              telemetry_internal::ParseDeviceCensusState(persisted);
          if (!persisted_state || !persisted_state->emitted) {
            std::_Exit(EXIT_FAILURE);
          }
          ++emission_count;
          emitted_day = census_day;
          if (versions != std::vector<std::string>{"1.24.0"}) {
            std::_Exit(EXIT_FAILURE);
          }
        };
        auto& device_id = DeviceId::Instance();

        const bool first = device_id.RecordCensusActivity(20700, "1.24.0", true, emit);
        const bool next_active_day = device_id.RecordCensusActivity(20707, "1.24.0", false, emit);

        std::_Exit(first && next_active_day && emission_count == 1 &&
                           emitted_day == 20700
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(DeviceIdDeathTest, ResetsCensusForDifferentDeviceId) {
  ScopedTestDirectory test_dir{"device_census_identity"};
  const fs::path home = test_dir.Path() / "home";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"HOME", home.string()}, {"XDG_CACHE_HOME", nullopt}}};
  const fs::path census_state =
      fs::path(DeviceId::GetStorageDirectory()) / "devicecensus.state";

  EXPECT_EXIT(
      {
        auto& device_id = DeviceId::Instance();
        const std::string current_id = device_id.GetValue();
        fs::create_directories(census_state.parent_path());
        std::ofstream state_file(census_state);
        state_file << "2\n00000000-0000-0000-0000-000000000001\n20700\n0\n1.23.0\n";
        state_file.close();

        int emission_count = 0;
        const bool recorded = device_id.RecordCensusActivity(
            20700, "1.24.0", false,
            [&](int64_t, const std::vector<std::string>&) {
              ++emission_count;
            });

        std::ifstream persisted_file(census_state);
        std::string persisted((std::istreambuf_iterator<char>(persisted_file)),
                              std::istreambuf_iterator<char>());
        const auto persisted_state =
            telemetry_internal::ParseDeviceCensusState(persisted);
        std::_Exit(recorded && persisted_state &&
                           persisted_state->identity == current_id &&
                           persisted_state->versions ==
                               std::vector<std::string>{"1.24.0"} &&
                           emission_count == 0
                       ? EXIT_SUCCESS
                       : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(DeviceIdTest, RecordsExternalCensusIdentity) {
  ScopedTestDirectory test_dir{"external_census_identity"};
  constexpr std::string_view identity =
      "6a09e667bb67ae853c6ef372a54ff53a";
  bool observed_persisted_state = false;

  ASSERT_TRUE(DeviceId::RecordCensusActivity(
      identity, test_dir.Path().string(), 20700, "1.24.0", true,
      [&](int64_t census_day, const std::vector<std::string>& versions) {
        std::ifstream state_file(test_dir.Path() / "devicecensus.state");
        std::string persisted((std::istreambuf_iterator<char>(state_file)),
                              std::istreambuf_iterator<char>());
        const auto persisted_state =
            telemetry_internal::ParseDeviceCensusState(persisted);
        observed_persisted_state =
            persisted_state && persisted_state->identity == identity &&
            persisted_state->utc_day == census_day &&
            persisted_state->emitted &&
            versions == std::vector<std::string>{"1.24.0"};
      }));
  EXPECT_TRUE(observed_persisted_state);
}

}  // namespace
}  // namespace onnxruntime::test
