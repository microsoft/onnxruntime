// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/device_id.h"

#include <filesystem>

#include "gtest/gtest.h"

#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime::test {
namespace {

namespace fs = std::filesystem;

constexpr const char* kDeviceIdDirectory = "Microsoft/DeveloperTools/.onnxruntime";

TEST(DeviceIdWindowsTest, UsesAbsoluteLocalAppDataFirst) {
  const fs::path local_app_data = fs::temp_directory_path() / "ort_device_id_local_app_data";
  const fs::path app_data = fs::temp_directory_path() / "ort_device_id_app_data";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"LOCALAPPDATA", local_app_data.string()},
                {"APPDATA", app_data.string()},
                {"HOME", nullopt},
                {"USERPROFILE", nullopt},
                {"HOMEDRIVE", nullopt},
                {"HOMEPATH", nullopt}}};

  EXPECT_EQ(fs::path(DeviceId::GetStorageDirectory()), local_app_data / kDeviceIdDirectory);
}

TEST(DeviceIdWindowsTest, FallsBackToAbsoluteAppData) {
  const fs::path app_data = fs::temp_directory_path() / "ort_device_id_app_data";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"LOCALAPPDATA", nullopt},
                {"APPDATA", app_data.string()},
                {"HOME", nullopt},
                {"USERPROFILE", nullopt},
                {"HOMEDRIVE", nullopt},
                {"HOMEPATH", nullopt}}};

  EXPECT_EQ(fs::path(DeviceId::GetStorageDirectory()), app_data / kDeviceIdDirectory);
}

TEST(DeviceIdWindowsTest, FallsBackToAbsoluteHome) {
  const fs::path home = fs::temp_directory_path() / "ort_device_id_home";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"LOCALAPPDATA", nullopt},
                {"APPDATA", nullopt},
                {"HOME", home.string()},
                {"USERPROFILE", nullopt},
                {"HOMEDRIVE", nullopt},
                {"HOMEPATH", nullopt}}};

  EXPECT_EQ(fs::path(DeviceId::GetStorageDirectory()),
            home / "AppData" / "Local" / kDeviceIdDirectory);
}

TEST(DeviceIdWindowsTest, FallsBackToUserProfileWhenAppDataIsUnavailable) {
  const fs::path user_profile = fs::temp_directory_path() / "ort_device_id_user_profile";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"LOCALAPPDATA", "relative-local"},
                {"APPDATA", "relative-roaming"},
                {"HOME", "relative-home"},
                {"USERPROFILE", user_profile.string()},
                {"HOMEDRIVE", nullopt},
                {"HOMEPATH", nullopt}}};

  EXPECT_EQ(fs::path(DeviceId::GetStorageDirectory()),
            user_profile / "AppData" / "Local" / kDeviceIdDirectory);
}

TEST(DeviceIdWindowsTest, FallsBackToHomeDriveAndPath) {
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"LOCALAPPDATA", nullopt},
                {"APPDATA", nullopt},
                {"HOME", nullopt},
                {"USERPROFILE", nullopt},
                {"HOMEDRIVE", "C:"},
                {"HOMEPATH", "\\Users\\ort-device-id"}}};

  EXPECT_EQ(fs::path(DeviceId::GetStorageDirectory()),
            fs::path("C:\\Users\\ort-device-id\\AppData\\Local") / kDeviceIdDirectory);
}

TEST(DeviceIdWindowsTest, RejectsRelativeFallbackPaths) {
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"LOCALAPPDATA", "relative-local"},
                {"APPDATA", "relative-roaming"},
                {"HOME", "relative-home"},
                {"USERPROFILE", "relative-profile"},
                {"HOMEDRIVE", "relative-drive"},
                {"HOMEPATH", "relative-path"}}};

  EXPECT_TRUE(DeviceId::GetStorageDirectory().empty());
}

}  // namespace
}  // namespace onnxruntime::test
