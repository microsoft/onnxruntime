// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/posix/device_id.h"

#include <Windows.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "core/platform/telemetry_guid.h"

namespace onnxruntime {
namespace {

constexpr char kDeviceIdRegistryKey[] = "SOFTWARE\\Microsoft\\DeveloperTools\\.onnxruntime";
constexpr char kDeviceIdRegistryValue[] = "deviceid";
constexpr size_t kMaxDeviceIdSize = 256;

enum class RegistryReadResult {
  Missing,
  Valid,
  Invalid,
  Failed,
};

struct RegistryRead {
  RegistryReadResult result;
  std::string value;
};

class ScopedWinHandle {
 public:
  explicit ScopedWinHandle(HANDLE handle = nullptr) : handle_(handle) {}
  ~ScopedWinHandle() {
    if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) {
      ::CloseHandle(handle_);
    }
  }

  ScopedWinHandle(const ScopedWinHandle&) = delete;
  ScopedWinHandle& operator=(const ScopedWinHandle&) = delete;

  HANDLE Get() const { return handle_; }

 private:
  HANDLE handle_;
};

class ScopedDeviceIdMutex {
 public:
  ScopedDeviceIdMutex() {
    HANDLE token{};
    if (!::OpenProcessToken(::GetCurrentProcess(), TOKEN_QUERY, &token)) {
      return;
    }
    ScopedWinHandle token_handle(token);

    DWORD size = 0;
    ::GetTokenInformation(token_handle.Get(), TokenUser, nullptr, 0, &size);
    if (size == 0) {
      return;
    }

    std::vector<unsigned char> token_info(size);
    if (!::GetTokenInformation(token_handle.Get(), TokenUser, token_info.data(), size, &size)) {
      return;
    }

    const auto* token_user = reinterpret_cast<const TOKEN_USER*>(token_info.data());
    if (!::IsValidSid(token_user->User.Sid)) {
      return;
    }

    const DWORD sid_size = ::GetLengthSid(token_user->User.Sid);
    const auto* sid_bytes = static_cast<const unsigned char*>(token_user->User.Sid);
    uint64_t sid_hash = 14695981039346656037ULL;
    for (DWORD i = 0; i < sid_size; ++i) {
      sid_hash ^= sid_bytes[i];
      sid_hash *= 1099511628211ULL;
    }

    std::array<wchar_t, 96> mutex_name{};
    _snwprintf_s(mutex_name.data(), mutex_name.size(), _TRUNCATE,
                 L"Global\\Microsoft.DeveloperTools.OnnxRuntime.DeviceId.%016llx",
                 static_cast<unsigned long long>(sid_hash));
    handle_ = ::CreateMutexW(nullptr, FALSE, mutex_name.data());
    if (handle_ == nullptr) {
      return;
    }

    const DWORD wait_result = ::WaitForSingleObject(handle_, 1000);
    acquired_ = wait_result == WAIT_OBJECT_0 || wait_result == WAIT_ABANDONED;
  }

  ~ScopedDeviceIdMutex() {
    if (acquired_) {
      ::ReleaseMutex(handle_);
    }
    if (handle_ != nullptr) {
      ::CloseHandle(handle_);
    }
  }

  ScopedDeviceIdMutex(const ScopedDeviceIdMutex&) = delete;
  ScopedDeviceIdMutex& operator=(const ScopedDeviceIdMutex&) = delete;

  explicit operator bool() const { return acquired_; }

 private:
  HANDLE handle_{};
  bool acquired_{};
};

void TrimAsciiWhitespace(std::string& value) {
  while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
    value.pop_back();
  }
  const auto first = std::find_if_not(
      value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); });
  value.erase(value.begin(), first);
}

bool IsValidGuid(const std::string& value) {
  if (value.size() != 36) {
    return false;
  }
  for (size_t i = 0; i < value.size(); ++i) {
    const bool separator = i == 8 || i == 13 || i == 18 || i == 23;
    if ((separator && value[i] != '-') ||
        (!separator && !std::isxdigit(static_cast<unsigned char>(value[i])))) {
      return false;
    }
  }
  return true;
}

RegistryRead ReadDeviceIdRegistryValue() {
  HKEY key{};
  const LSTATUS open_status =
      ::RegOpenKeyExA(HKEY_CURRENT_USER, kDeviceIdRegistryKey, 0, KEY_READ | KEY_WOW64_64KEY, &key);
  if (open_status == ERROR_FILE_NOT_FOUND) {
    return {RegistryReadResult::Missing, {}};
  }
  if (open_status != ERROR_SUCCESS) {
    return {RegistryReadResult::Failed, {}};
  }

  std::array<char, kMaxDeviceIdSize> buffer{};
  DWORD type = 0;
  DWORD size = static_cast<DWORD>(buffer.size());
  const LSTATUS query_status = ::RegQueryValueExA(
      key, kDeviceIdRegistryValue, nullptr, &type, reinterpret_cast<LPBYTE>(buffer.data()), &size);
  ::RegCloseKey(key);

  if (query_status == ERROR_FILE_NOT_FOUND) {
    return {RegistryReadResult::Missing, {}};
  }
  if (query_status != ERROR_SUCCESS) {
    return {RegistryReadResult::Failed, {}};
  }
  if (type != REG_SZ || size == 0 || size > buffer.size()) {
    return {RegistryReadResult::Invalid, {}};
  }

  buffer[buffer.size() - 1] = '\0';
  std::string value(buffer.data());
  TrimAsciiWhitespace(value);
  return {IsValidGuid(value) ? RegistryReadResult::Valid : RegistryReadResult::Invalid,
          std::move(value)};
}

bool WriteDeviceIdRegistryValue(const std::string& value) {
  HKEY key{};
  if (::RegCreateKeyExA(HKEY_CURRENT_USER, kDeviceIdRegistryKey, 0, nullptr, REG_OPTION_NON_VOLATILE,
                        KEY_WRITE | KEY_WOW64_64KEY, nullptr, &key, nullptr) != ERROR_SUCCESS) {
    return false;
  }

  const LSTATUS status =
      ::RegSetValueExA(key, kDeviceIdRegistryValue, 0, REG_SZ,
                       reinterpret_cast<const BYTE*>(value.c_str()),
                       static_cast<DWORD>(value.size() + 1));
  ::RegCloseKey(key);
  return status == ERROR_SUCCESS;
}

}  // namespace

DeviceId& DeviceId::Instance() {
  static DeviceId instance;
  return instance;
}

std::string DeviceId::GetValue() {
  std::lock_guard<std::mutex> lock(mutex_);
  InitializeInternal();
  return device_id_;
}

DeviceIdStatus DeviceId::GetStatus() {
  std::lock_guard<std::mutex> lock(mutex_);
  InitializeInternal();
  return status_;
}

std::string DeviceId::GetStatusString() {
  switch (GetStatus()) {
    case DeviceIdStatus::New:
      return "New";
    case DeviceIdStatus::Existing:
      return "Existing";
    case DeviceIdStatus::Corrupted:
      return "Corrupted";
    case DeviceIdStatus::Failed:
      return "Failed";
    default:
      return "Unknown";
  }
}

bool DeviceId::IsValidGUID(const std::string& value) {
  return IsValidGuid(value);
}

std::string DeviceId::GetStorageDirectory() {
  char* local_app_data = nullptr;
  size_t length = 0;
  if (_dupenv_s(&local_app_data, &length, "LOCALAPPDATA") != 0 ||
      local_app_data == nullptr || length <= 1) {
    std::free(local_app_data);
    return {};
  }
  const std::filesystem::path path =
      std::filesystem::path(local_app_data) / "Microsoft" / "DeveloperTools" / ".onnxruntime";
  std::free(local_app_data);
  return path.string();
}

std::string DeviceId::EnsureStorageDirectory() {
  const std::string path = GetStorageDirectory();
  if (path.empty() || !CreateDirectoryTree(path)) {
    return {};
  }
  return path;
}

bool DeviceId::CreateDirectoryTree(const std::string& path, bool /*leaf*/) {
  std::error_code error;
  std::filesystem::create_directories(path, error);
  return !error && std::filesystem::is_directory(path, error);
}

void DeviceId::InitializeInternal() {
  if (initialized_) {
    return;
  }
  initialized_ = true;

  RegistryRead existing = ReadDeviceIdRegistryValue();
  if (existing.result == RegistryReadResult::Valid) {
    device_id_ = std::move(existing.value);
    status_ = DeviceIdStatus::Existing;
    return;
  }
  if (existing.result == RegistryReadResult::Failed) {
    device_id_ = GenerateGuidV4();
    status_ = DeviceIdStatus::Failed;
    return;
  }
  const bool was_corrupted = existing.result == RegistryReadResult::Invalid;

  ScopedDeviceIdMutex lock;
  if (!lock) {
    device_id_ = GenerateGuidV4();
    status_ = DeviceIdStatus::Failed;
    return;
  }

  existing = ReadDeviceIdRegistryValue();
  if (existing.result == RegistryReadResult::Valid) {
    device_id_ = std::move(existing.value);
    status_ = DeviceIdStatus::Existing;
    return;
  }
  if (existing.result == RegistryReadResult::Failed) {
    device_id_ = GenerateGuidV4();
    status_ = DeviceIdStatus::Failed;
    return;
  }

  device_id_ = GenerateGuidV4();
  if (!WriteDeviceIdRegistryValue(device_id_)) {
    status_ = DeviceIdStatus::Failed;
    return;
  }
  status_ = (was_corrupted || existing.result == RegistryReadResult::Invalid)
                ? DeviceIdStatus::Corrupted
                : DeviceIdStatus::New;
}

}  // namespace onnxruntime
