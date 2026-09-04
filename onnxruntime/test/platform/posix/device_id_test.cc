// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/device_id.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

#include "gtest/gtest.h"

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

}  // namespace
}  // namespace onnxruntime::test
