// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#include <winternl.h>

#include "core/platform/telemetry_1ds.h"
#include "core/platform/telemetry_1ds_platform.h"
#include "core/platform/windows/telemetry_1ds.h"
#include "core/platform/windows/telemetry.h"
#include "core/platform/telemetry_no_throw.h"
#include "core/platform/telemetry_sampling.h"
#include "core/platform/telemetry_environment.h"
#include "core/platform/telemetry_redaction.h"

#include <EventProperties.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>
#include <limits>
#include <string_view>
#include <utility>
#include <vector>

#include "core/common/common.h"
#include "core/common/logging/logging.h"

using namespace Microsoft::Applications::Events;

namespace onnxruntime {
namespace {

std::string GetFileName(std::string_view path) {
  const size_t separator = path.find_last_of("/\\");
  return std::string(path.substr(separator == std::string_view::npos ? 0 : separator + 1));
}

EventProperties NewEvent(std::string name) {
  EventProperties event(std::move(name));
  event.SetLatency(EventLatency_Normal);
  event.SetPopsample(100.0);
  event.SetLevel(DIAG_LEVEL_REQUIRED);
  return event;
}

bool PrepareSampledProcessEvent(EventProperties& event) {
  if (!telemetry_internal::ShouldSampleSession(
          telemetry_internal::GetAppSessionGuid(), 0,
          telemetry_internal::kOtherProcessEventSampleRatePercent)) {
    return false;
  }
  event.SetPopsample(telemetry_internal::kOtherProcessEventSampleRatePercent);
  return true;
}

template <typename Operation>
void RunWindowsTelemetryOperation(const char* name, Operation&& operation) noexcept {
  telemetry_internal::RunTelemetryOperationNoThrow(std::forward<Operation>(operation), [name](const char* message) {
    if (logging::LoggingManager::HasDefaultLogger()) {
      if (message != nullptr) {
        LOGS_DEFAULT(WARNING) << "[Telemetry] " << name << " failed: " << message;
      } else {
        LOGS_DEFAULT(WARNING) << "[Telemetry] " << name << " failed with an unknown exception";
      }
    }
  });
}

}  // namespace

namespace telemetry_internal {

const char* GetDefaultEncodedToken() {
  return "fllXSmJHWBYMX1cqWldIYU0KTA0IVnpWD0lkRloSWwhILF5WGmBDDUdEC1csC0NMZhZYWVEPBH9DWEhlR15MCl1UfgwPVWVDX0U=";
}

int32_t GetProcessorCount() {
  const DWORD count = ::GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
  return count > static_cast<DWORD>(std::numeric_limits<int32_t>::max())
             ? std::numeric_limits<int32_t>::max()
             : static_cast<int32_t>(count);
}

std::string GetCachePath(const std::string& directory) {
  return directory + "\\onnxruntime.db";
}

EventProperties BuildExecutionProviderEvent(const LUID& adapter_luid) {
  auto event = NewEvent("ExecutionProviderEvent");
  event.SetProperty("adapterLuidLowPart", static_cast<int64_t>(adapter_luid.LowPart));
  event.SetProperty("adapterLuidHighPart", static_cast<int64_t>(static_cast<uint32_t>(adapter_luid.HighPart)));
  return event;
}

EventProperties BuildDriverInfoEvent(
    std::string_view device_class, std::wstring_view driver_names, std::wstring_view driver_versions) {
  auto event = NewEvent("DriverInfo");
  event.SetProperty("schemaVersion", int64_t{0});
  event.SetProperty("deviceClass", std::string(device_class));
  event.SetProperty("driverNames", ToUTF8String(driver_names));
  event.SetProperty("driverVersions", ToUTF8String(driver_versions));
  return event;
}

EventProperties BuildProviderOptionsEvent(
    const std::string& provider_id, const std::string& provider_options, bool capture_state) {
  auto event = NewEvent(capture_state ? "ProviderOptions_CaptureState" : "ProviderOptions");
  event.SetProperty("schemaVersion", int64_t{0});
  event.SetProperty("providerId", provider_id);
  event.SetProperty("providerOptions", ScrubStringForTelemetry(provider_options));
  return event;
}

}  // namespace telemetry_internal

std::string OneDsTelemetry::GetPlatformInfo() const {
  return "Windows";
}

std::string OneDsTelemetry::GetOsDescription() const {
  using RtlGetVersionFn = LONG(WINAPI*)(PRTL_OSVERSIONINFOW);
  const HMODULE ntdll = ::GetModuleHandleW(L"ntdll.dll");
  const auto rtl_get_version =
      ntdll == nullptr ? nullptr
                       : reinterpret_cast<RtlGetVersionFn>(::GetProcAddress(ntdll, "RtlGetVersion"));
  if (rtl_get_version != nullptr) {
    RTL_OSVERSIONINFOW version_info{};
    version_info.dwOSVersionInfoSize = sizeof(version_info);
    if (rtl_get_version(&version_info) == 0) {
      std::array<char, 64> description{};
      std::snprintf(description.data(), description.size(), "Windows %lu.%lu (Build %lu)",
                    version_info.dwMajorVersion, version_info.dwMinorVersion,
                    version_info.dwBuildNumber);
      return description.data();
    }
  }
  return "Windows";
}

std::string OneDsTelemetry::GetCpuModel() const {
  HKEY key{};
  if (::RegOpenKeyExA(HKEY_LOCAL_MACHINE,
                      "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                      0, KEY_READ, &key) != ERROR_SUCCESS) {
    return {};
  }

  char cpu_model[256]{};
  DWORD value_type = REG_SZ;
  DWORD size = sizeof(cpu_model);
  const LSTATUS status = ::RegQueryValueExA(
      key, "ProcessorNameString", nullptr, &value_type,
      reinterpret_cast<LPBYTE>(cpu_model), &size);
  ::RegCloseKey(key);
  if (status != ERROR_SUCCESS || value_type != REG_SZ || size == 0) {
    return {};
  }
  cpu_model[sizeof(cpu_model) - 1] = '\0';
  return cpu_model;
}

std::string OneDsTelemetry::GetDeviceClass() const {
  return "Desktop";
}

telemetry_detail::HostEnvironmentInfo OneDsTelemetry::GetHostEnvironmentInfo() {
  return telemetry_detail::ClassifyHostEnvironment({});
}

std::string OneDsTelemetry::GetProcessName() {
  std::vector<wchar_t> path(MAX_PATH);
  for (;;) {
    const DWORD length = ::GetModuleFileNameW(nullptr, path.data(), static_cast<DWORD>(path.size()));
    if (length == 0) {
      return {};
    }
    if (length < path.size()) {
      std::string process_name = GetFileName(ToUTF8String(std::wstring_view(path.data(), length)));
      constexpr std::string_view executable_extension = ".exe";
      if (process_name.size() > executable_extension.size() &&
          std::equal(executable_extension.begin(), executable_extension.end(),
                     process_name.end() - executable_extension.size(),
                     [](char lhs, char rhs) {
                       return std::tolower(static_cast<unsigned char>(lhs)) ==
                              std::tolower(static_cast<unsigned char>(rhs));
                     })) {
        process_name.resize(process_name.size() - executable_extension.size());
      }
      return process_name;
    }
    if (path.size() >= 32768) {
      return {};
    }
    path.resize(std::min<size_t>(path.size() * 2, 32768));
  }
}

std::string OneDsTelemetry::GetArchitecture() {
#if defined(_M_X64) && !defined(_M_ARM64EC)
  return "x86_64";
#elif defined(_M_IX86)
  return "x86";
#elif defined(_M_ARM64) || defined(_M_ARM64EC)
  return "arm64";
#elif defined(_M_ARM)
  return "arm";
#else
  return "unknown";
#endif
}

int64_t OneDsTelemetry::GetTotalMemoryMB() {
  MEMORYSTATUSEX memory_status{};
  memory_status.dwLength = sizeof(memory_status);
  if (::GlobalMemoryStatusEx(&memory_status) != 0) {
    return static_cast<int64_t>(memory_status.ullTotalPhys / (1024 * 1024));
  }
  return 0;
}

void OneDsTelemetry::LogExecutionProviderEvent(LUID* adapter_luid) const {
  RunWindowsTelemetryOperation("LogExecutionProviderEvent", [&]() {
    if (!IsEnabled() || adapter_luid == nullptr) {
      return;
    }
    auto event = telemetry_internal::BuildExecutionProviderEvent(*adapter_luid);
    if (PrepareSampledProcessEvent(event)) {
      LogEventAsync(std::move(event));
    }
  });
}

void OneDsTelemetry::LogDriverInfoEvent(
    std::string_view device_class, const std::wstring_view& driver_names,
    const std::wstring_view& driver_versions) const {
  RunWindowsTelemetryOperation("LogDriverInfoEvent", [&]() {
    if (!IsEnabled()) {
      return;
    }
    auto event = telemetry_internal::BuildDriverInfoEvent(device_class, driver_names, driver_versions);
    if (PrepareSampledProcessEvent(event)) {
      LogEventAsync(std::move(event));
    }
  });
}

void OneDsTelemetry::LogProviderOptions(
    const std::string& provider_id, const std::string& provider_options_string, bool capture_state) const {
  RunWindowsTelemetryOperation("LogProviderOptions", [&]() {
    if (!IsEnabled()) {
      return;
    }
    WindowsTelemetry::LogLocalProviderOptions(provider_id, provider_options_string, capture_state);
    auto event = telemetry_internal::BuildProviderOptionsEvent(provider_id, provider_options_string, capture_state);
    if (PrepareSampledProcessEvent(event)) {
      LogEventAsync(std::move(event));
    }
  });
}

}  // namespace onnxruntime
