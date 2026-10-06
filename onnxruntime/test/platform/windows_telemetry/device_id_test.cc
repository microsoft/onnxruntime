// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_id.h"

#include <Windows.h>

#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "core/common/common.h"
#include "core/platform/telemetry_guid.h"
#include "core/platform/telemetry_strings.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime::test {
namespace {

namespace fs = std::filesystem;

constexpr char kDeviceIdRegistryKey[] = "SOFTWARE\\Microsoft\\DeveloperTools\\.onnxruntime";
constexpr char kDeviceIdRegistryValue[] = "deviceid";

fs::path Utf8Path(const std::string& value) {
  return fs::path(std::u8string(value.begin(), value.end()));
}

class ScopedRegistryOverride {
 public:
  ScopedRegistryOverride()
      : path_("SOFTWARE\\Microsoft\\onnxruntime-tests\\device-id-" +
              std::to_string(::GetCurrentProcessId())) {
    HKEY writable_key{};
    ORT_ENFORCE(::RegCreateKeyExA(HKEY_CURRENT_USER, path_.c_str(), 0, nullptr, REG_OPTION_NON_VOLATILE,
                                  KEY_ALL_ACCESS, nullptr, &writable_key, nullptr) == ERROR_SUCCESS);
    ORT_ENFORCE(::RegCloseKey(writable_key) == ERROR_SUCCESS);
    ORT_ENFORCE(::RegOpenKeyExA(HKEY_CURRENT_USER, path_.c_str(), 0, KEY_ALL_ACCESS, &key_) == ERROR_SUCCESS);
    ORT_ENFORCE(::RegOverridePredefKey(HKEY_CURRENT_USER, key_) == ERROR_SUCCESS);
    overridden_ = true;
  }

  ~ScopedRegistryOverride() {
    Reset();
  }

  ScopedRegistryOverride(const ScopedRegistryOverride&) = delete;
  ScopedRegistryOverride& operator=(const ScopedRegistryOverride&) = delete;

  void WriteValue(DWORD type, const void* data, DWORD size) {
    HKEY device_id_key{};
    ASSERT_EQ(::RegCreateKeyExA(HKEY_CURRENT_USER, kDeviceIdRegistryKey, 0, nullptr, REG_OPTION_NON_VOLATILE,
                                KEY_ALL_ACCESS, nullptr, &device_id_key, nullptr),
              ERROR_SUCCESS);
    ASSERT_EQ(::RegSetValueExA(device_id_key, kDeviceIdRegistryValue, 0, type,
                               static_cast<const BYTE*>(data), size),
              ERROR_SUCCESS);
    ASSERT_EQ(::RegCloseKey(device_id_key), ERROR_SUCCESS);
  }

  std::string ReadValue() {
    HKEY device_id_key{};
    if (::RegOpenKeyExA(HKEY_CURRENT_USER, kDeviceIdRegistryKey, 0, KEY_READ, &device_id_key) != ERROR_SUCCESS) {
      return {};
    }

    std::vector<char> value(512);
    DWORD type = 0;
    DWORD size = static_cast<DWORD>(value.size());
    const LSTATUS status = ::RegQueryValueExA(
        device_id_key, kDeviceIdRegistryValue, nullptr, &type,
        reinterpret_cast<BYTE*>(value.data()), &size);
    ::RegCloseKey(device_id_key);
    if (status != ERROR_SUCCESS || type != REG_SZ || size == 0) {
      return {};
    }

    value.back() = '\0';
    return value.data();
  }

  void Reset() {
    if (overridden_) {
      EXPECT_EQ(::RegOverridePredefKey(HKEY_CURRENT_USER, nullptr), ERROR_SUCCESS);
      overridden_ = false;
    }
    if (key_ != nullptr) {
      EXPECT_EQ(::RegCloseKey(key_), ERROR_SUCCESS);
      key_ = nullptr;
    }
    const LSTATUS delete_status = ::RegDeleteTreeA(HKEY_CURRENT_USER, path_.c_str());
    EXPECT_TRUE(delete_status == ERROR_SUCCESS || delete_status == ERROR_FILE_NOT_FOUND);
  }

 private:
  std::string path_;
  HKEY key_{};
  bool overridden_{};
};

class ScopedWideEnvironmentVariable {
 public:
  ScopedWideEnvironmentVariable(const wchar_t* name, const std::wstring& value) : name_(name) {
    const DWORD required_size = ::GetEnvironmentVariableW(name_.c_str(), nullptr, 0);
    if (required_size != 0) {
      original_value_.resize(required_size, L'\0');
      const DWORD value_size =
          ::GetEnvironmentVariableW(name_.c_str(), original_value_.data(), required_size);
      ORT_ENFORCE(value_size != 0 && value_size < required_size);
      original_value_.resize(value_size);
      was_defined_ = true;
    }
    ORT_ENFORCE(::SetEnvironmentVariableW(name_.c_str(), value.c_str()));
  }

  ~ScopedWideEnvironmentVariable() {
    EXPECT_TRUE(::SetEnvironmentVariableW(name_.c_str(),
                                          was_defined_ ? original_value_.c_str() : nullptr));
  }

  ScopedWideEnvironmentVariable(const ScopedWideEnvironmentVariable&) = delete;
  ScopedWideEnvironmentVariable& operator=(const ScopedWideEnvironmentVariable&) = delete;

 private:
  std::wstring name_;
  std::wstring original_value_;
  bool was_defined_{};
};

TEST(DeviceIdWindowsTest, FallsBackToAbsoluteAppData) {
  const fs::path app_data = fs::temp_directory_path() / "ort_device_id_app_data";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"LOCALAPPDATA", nullopt},
                {"APPDATA", app_data.string()},
                {"HOME", nullopt},
                {"USERPROFILE", nullopt}}};

  EXPECT_EQ(Utf8Path(DeviceId::GetStorageDirectory()),
            app_data / "Microsoft" / "DeveloperTools" / ".onnxruntime");
}

TEST(DeviceIdWindowsTest, FallsBackToUserProfileWhenAppDataIsUnavailable) {
  const fs::path user_profile = fs::temp_directory_path() / "ort_device_id_user_profile";
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"LOCALAPPDATA", "relative-local"},
                {"APPDATA", "relative-roaming"},
                {"HOME", "relative-home"},
                {"USERPROFILE", user_profile.string()}}};

  EXPECT_EQ(Utf8Path(DeviceId::GetStorageDirectory()),
            user_profile / "AppData" / "Local" / "Microsoft" / "DeveloperTools" / ".onnxruntime");
}

TEST(DeviceIdWindowsTest, FallsBackToHomeDriveAndPath) {
  const fs::path home = fs::temp_directory_path() / "ort_device_id_home";
  const std::wstring home_native = home.native();
  const std::wstring drive = home.root_name().native();
  ScopedEnvironmentVariables environment{
      EnvVarMap{{"LOCALAPPDATA", nullopt},
                {"APPDATA", nullopt},
                {"HOME", nullopt},
                {"USERPROFILE", nullopt}}};
  ScopedWideEnvironmentVariable home_drive(L"HOMEDRIVE", drive);
  ScopedWideEnvironmentVariable home_path(L"HOMEPATH", home_native.substr(drive.size()));

  EXPECT_EQ(Utf8Path(DeviceId::GetStorageDirectory()),
            home / "AppData" / "Local" / "Microsoft" / "DeveloperTools" / ".onnxruntime");
}

TEST(DeviceIdWindowsTest, CreatesUnicodeStorageDirectory) {
  const fs::path app_data = fs::temp_directory_path() / L"ort_device_id_\u6d4b\u8bd5";
  ScopedWideEnvironmentVariable local_app_data(L"LOCALAPPDATA", app_data.native());

  const fs::path storage_dir = Utf8Path(DeviceId::EnsureStorageDirectory());
  EXPECT_EQ(storage_dir, app_data / "Microsoft" / "DeveloperTools" / ".onnxruntime");
  EXPECT_TRUE(fs::is_directory(storage_dir));

  std::error_code error;
  fs::remove_all(app_data, error);
  EXPECT_FALSE(error);
}

TEST(DeviceIdWindowsTest, RejectsOversizedStoragePathsWithoutTruncation) {
  const std::wstring drive = fs::temp_directory_path().root_name().native() + L"\\";
  const std::wstring paths[] = {
      drive + std::wstring(telemetry_detail::kMaxTelemetryPathBytes, L'a'),
      drive + std::wstring(telemetry_detail::kMaxTelemetryPathBytes / 3, L'\u20ac'),
  };
  for (const auto& app_data : paths) {
    ScopedWideEnvironmentVariable local_app_data(L"LOCALAPPDATA", app_data);
    ScopedEnvironmentVariables environment{
        EnvVarMap{{"APPDATA", nullopt}, {"HOME", nullopt}, {"USERPROFILE", nullopt}, {"HOMEDRIVE", nullopt}, {"HOMEPATH", nullopt}}};
    EXPECT_TRUE(DeviceId::GetStorageDirectory().empty());
  }
}

TEST(DeviceIdWindowsTest, IncludesStorageSuffixInPathBudget) {
  const std::wstring drive = fs::temp_directory_path().root_name().native() + L"\\";
  const std::wstring suffix = L"\\Microsoft\\DeveloperTools\\.onnxruntime";
  const std::wstring app_data = drive + std::wstring(
                                            telemetry_detail::kMaxTelemetryPathBytes - drive.size() - suffix.size(), L'a');
  for (const auto& extra : {std::wstring{}, std::wstring{L"a"}}) {
    ScopedWideEnvironmentVariable local_app_data(L"LOCALAPPDATA", app_data + extra);
    const std::string storage = DeviceId::GetStorageDirectory();
    if (extra.empty()) {
      EXPECT_EQ(storage.size(), telemetry_detail::kMaxTelemetryPathBytes);
      EXPECT_EQ(Utf8Path(storage), fs::path(app_data + suffix));
    } else {
      EXPECT_TRUE(storage.empty());
    }
  }
}

TEST(DeviceIdWindowsDeathTest, CreatesMissingRegistryValue) {
  EXPECT_EXIT(
      {
        ScopedRegistryOverride registry;
        const std::string value = DeviceId::Instance().GetValue();
        const bool passed = DeviceId::Instance().GetStatus() == DeviceIdStatus::New &&
                            IsValidGuid(value) &&
                            registry.ReadValue() == value;
        registry.Reset();
        std::_Exit(passed ? EXIT_SUCCESS : EXIT_FAILURE);
      },
      ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST(DeviceIdWindowsDeathTest, LoadsExistingRegistryValue) {
  constexpr char kExistingId[] = "11111111-2222-4333-8444-555555555555";
  for (const auto& stored : {std::string(kExistingId), std::string(" \t") + kExistingId + "\r\n"}) {
    SCOPED_TRACE(stored.size());
    EXPECT_EXIT(
        {
          ScopedRegistryOverride registry;
          registry.WriteValue(REG_SZ, stored.c_str(), static_cast<DWORD>(stored.size() + 1));
          const std::string value = DeviceId::Instance().GetValue();
          const bool passed = DeviceId::Instance().GetStatus() == DeviceIdStatus::Existing &&
                              value == kExistingId &&
                              registry.ReadValue() == stored;
          registry.Reset();
          std::_Exit(passed ? EXIT_SUCCESS : EXIT_FAILURE);
        },
        ::testing::ExitedWithCode(EXIT_SUCCESS), "");
  }
}

TEST(DeviceIdWindowsDeathTest, RepairsMalformedRegistryValues) {
  const std::string malformed_values[] = {
      "corrupted",
      std::string(512, 'a'),
      std::string("11111111-2222-4333-8444-555555555555\0junk", 41),
  };
  for (const auto& malformed : malformed_values) {
    SCOPED_TRACE(malformed.size());
    EXPECT_EXIT(
        {
          ScopedRegistryOverride registry;
          registry.WriteValue(REG_SZ, malformed.c_str(), static_cast<DWORD>(malformed.size() + 1));
          const std::string value = DeviceId::Instance().GetValue();
          const bool passed = DeviceId::Instance().GetStatus() == DeviceIdStatus::Corrupted &&
                              IsValidGuid(value) &&
                              registry.ReadValue() == value;
          registry.Reset();
          std::_Exit(passed ? EXIT_SUCCESS : EXIT_FAILURE);
        },
        ::testing::ExitedWithCode(EXIT_SUCCESS), "");
  }
}

}  // namespace
}  // namespace onnxruntime::test
