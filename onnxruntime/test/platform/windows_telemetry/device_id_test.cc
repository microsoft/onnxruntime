// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/device_id.h"

#include <filesystem>

#include "gtest/gtest.h"

#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime::test {
namespace {

namespace fs = std::filesystem;

TEST(DeviceIdWindowsTest, FallsBackToAbsoluteAppData) {
  const fs::path app_data = fs::temp_directory_path() / "ort_device_id_app_data";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"LOCALAPPDATA", nullopt},
                {"APPDATA", app_data.string()},
                {"HOME", nullopt},
                {"USERPROFILE", nullopt}}};

  EXPECT_EQ(fs::path(DeviceId::GetStorageDirectory()),
            app_data / "Microsoft" / "DeveloperTools" / ".onnxruntime");
}

TEST(DeviceIdWindowsTest, FallsBackToUserProfileWhenAppDataIsUnavailable) {
  const fs::path user_profile = fs::temp_directory_path() / "ort_device_id_user_profile";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"LOCALAPPDATA", "relative-local"},
                {"APPDATA", "relative-roaming"},
                {"HOME", "relative-home"},
                {"USERPROFILE", user_profile.string()}}};

  EXPECT_EQ(fs::path(DeviceId::GetStorageDirectory()),
            user_profile / "AppData" / "Local" / "Microsoft" / "DeveloperTools" / ".onnxruntime");
}

}  // namespace
}  // namespace onnxruntime::test
