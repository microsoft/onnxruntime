// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/telemetry_1ds.h"
#include "core/platform/telemetry_1ds_platform.h"
#include "core/platform/telemetry_environment.h"
#include "core/common/logging/logging.h"

#ifdef __APPLE__
#include <TargetConditionals.h>
#include <mach-o/dyld.h>
#include <sys/sysctl.h>
#include <sys/types.h>
#endif

#include <unistd.h>

#include <array>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>
#include <vector>

namespace onnxruntime {
namespace {
std::string GetFileName(std::string_view path) {
  const size_t separator = path.find_last_of("/\\");
  return telemetry_detail::BoundedTelemetryString(path.substr(separator == std::string_view::npos ? 0 : separator + 1));
}

#if defined(__linux__) || defined(__ANDROID__)
std::string ReadBoundedFile(const char* path, size_t max_bytes = telemetry_detail::kMaxTelemetryProbeBytes) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::string buffer(max_bytes, '\0');
  input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
  buffer.resize(static_cast<size_t>(input.gcount()));
  return buffer;
}
#endif

}  // namespace

namespace telemetry_internal {
const char* GetDefaultEncodedToken() {
  return "eg8KQWRGDBBdD1YuWl9JahRaTFhZVX4NDUhgRF9MXlhIegxYHGpFXxJEXVd/V0NMa0FXWVEOA3hDCxw3RFhCUF8Edl0NVWRMWEM=";
}

int32_t GetProcessorCount() {
  auto n = sysconf(_SC_NPROCESSORS_ONLN);
  if (n <= 0) {
    return 0;
  }
  if (n > static_cast<long>(std::numeric_limits<int32_t>::max())) {
    return std::numeric_limits<int32_t>::max();
  }
  return static_cast<int32_t>(n);
}

std::string GetCachePath(const std::string& directory) {
  return directory + "/onnxruntime.db";
}
}  // namespace telemetry_internal

std::string OneDsTelemetry::GetPlatformInfo() const {
#if defined(__APPLE__)
#if TARGET_OS_IOS
  return "iOS";
#elif TARGET_OS_MAC
  return "macOS";
#else
  return "Apple";
#endif
#elif defined(__ANDROID__)
  return "Android";
#elif defined(__linux__)
  return "Linux";
#else
  return "Unknown";
#endif
}

// ---------------------------------------------------------------------------
// Process / system info helpers for LogProcessInfo
// ---------------------------------------------------------------------------

// Get detailed OS version string (e.g., "macOS 15.2", "Ubuntu 22.04 LTS")
std::string OneDsTelemetry::GetOsDescription() const {
#if defined(__APPLE__)
  char version[64] = {};
  size_t len = sizeof(version);
  if (sysctlbyname("kern.osproductversion", version, &len, nullptr, 0) == 0) {
    version[sizeof(version) - 1] = '\0';
#if TARGET_OS_IOS
    return std::string("iOS ") + version;
#else
    return std::string("macOS ") + version;
#endif
  }
  return GetPlatformInfo();

#elif defined(__ANDROID__)
  // Read Android system properties via /system/build.prop
  std::string release, sdk;
  std::istringstream prop(ReadBoundedFile("/system/build.prop"));
  if (prop) {
    std::string line;
    while (std::getline(prop, line)) {
      if (line.rfind("ro.build.version.release=", 0) == 0)
        release = telemetry_detail::BoundedTelemetryString(std::string_view(line).substr(25));
      else if (line.rfind("ro.build.version.sdk=", 0) == 0)
        sdk = telemetry_detail::BoundedTelemetryString(std::string_view(line).substr(21));
    }
  }
  if (!release.empty()) {
    std::string result = "Android ";
    telemetry_detail::AppendTelemetryString(result, release);
    if (!sdk.empty()) {
      telemetry_detail::AppendTelemetryString(result, " (API ");
      telemetry_detail::AppendTelemetryString(result, sdk);
      telemetry_detail::AppendTelemetryString(result, ")");
    }
    return result;
  }
  return "Android";

#elif defined(__linux__)
  // Parse /etc/os-release for PRETTY_NAME (e.g., "Ubuntu 22.04.3 LTS")
  std::istringstream os_release(ReadBoundedFile("/etc/os-release"));
  if (os_release) {
    std::string line;
    while (std::getline(os_release, line)) {
      if (line.rfind("PRETTY_NAME=", 0) == 0) {
        std::string_view value = std::string_view(line).substr(12);
        if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
          value = value.substr(1, value.size() - 2);
        }
        return telemetry_detail::BoundedTelemetryString(value);
      }
    }
  }
  return "Linux";

#else
  return "Unknown";
#endif
}

// Get the CPU brand string (e.g. "Intel(R) Core(TM) i7-10700K"). Empty when unavailable.
std::string OneDsTelemetry::GetCpuModel() const {
#if defined(__APPLE__)
  // macOS/iOS expose the CPU brand string via sysctl.
  char buf[256] = {0};
  size_t size = sizeof(buf);
  if (sysctlbyname("machdep.cpu.brand_string", buf, &size, nullptr, 0) == 0) {
    buf[sizeof(buf) - 1] = '\0';
    return std::string(buf);
  }
  return "";

#elif defined(__linux__) || defined(__ANDROID__)
  // /proc/cpuinfo exposes the CPU brand as "model name" (x86) or "Hardware" (ARM).
  std::istringstream cpuinfo(ReadBoundedFile("/proc/cpuinfo"));
  std::string line;
  std::string hardware;
  while (std::getline(cpuinfo, line)) {
    const size_t colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    std::string key = telemetry_detail::BoundedTelemetryString(std::string_view(line).substr(0, colon));
    while (!key.empty() && std::isspace(static_cast<unsigned char>(key.back()))) {
      key.pop_back();
    }
    std::string_view value = std::string_view(line).substr(colon + 1);
    const size_t start = value.find_first_not_of(" \t");
    value = (start == std::string::npos) ? std::string_view() : value.substr(start);
    if (key == "model name") {
      return telemetry_detail::BoundedTelemetryString(value);
    }
    if (hardware.empty() && key == "Hardware") {
      hardware = telemetry_detail::BoundedTelemetryString(value);
    }
  }
  return hardware;

#else
  return "";
#endif
}

// Coarse device class for the host: "Mobile" on Android/iOS, "Desktop" elsewhere.
std::string OneDsTelemetry::GetDeviceClass() const {
#if defined(__ANDROID__) || (defined(__APPLE__) && TARGET_OS_IOS)
  return "Mobile";
#else
  return "Desktop";
#endif
}

namespace {

#if defined(__linux__) || defined(__ANDROID__)
bool FileExists(const char* path) {
  std::ifstream input(path);
  return input.good();
}
#endif

}  // namespace

telemetry_detail::HostEnvironmentInfo OneDsTelemetry::GetHostEnvironmentInfo() {
  telemetry_detail::HostEnvironmentEvidence evidence;
#if defined(__linux__) || defined(__ANDROID__)
  evidence.docker_marker = FileExists("/.dockerenv");
  evidence.podman_marker = FileExists("/run/.containerenv");
  const auto read_evidence = [](const char* name) {
    const auto value = telemetry_detail::ReadTelemetryEnvironment(name);
    if (!value) {
      if (logging::LoggingManager::HasDefaultLogger()) {
        LOGS_DEFAULT(WARNING) << "Ignoring oversized or unreadable telemetry environment evidence " << name;
      }
      return std::string{};
    }
    return *value;
  };
  evidence.kubernetes = !read_evidence("KUBERNETES_SERVICE_HOST").empty();
  evidence.aws_ecs = !read_evidence("ECS_CONTAINER_METADATA_URI").empty() ||
                     !read_evidence("ECS_CONTAINER_METADATA_URI_V4").empty();
  evidence.generic_container =
      telemetry_detail::IsTruthyCiValue(read_evidence("DOTNET_RUNNING_IN_CONTAINER"));
  evidence.systemd_container =
      ReadBoundedFile("/run/systemd/container") + read_evidence("container");
  evidence.cgroup = ReadBoundedFile("/proc/1/cgroup") + ReadBoundedFile("/proc/self/cgroup");
  evidence.cpu_info = ReadBoundedFile("/proc/cpuinfo");
  evidence.kernel_release = ReadBoundedFile("/proc/sys/kernel/osrelease");
  evidence.dmi = ReadBoundedFile("/sys/class/dmi/id/sys_vendor") +
                 ReadBoundedFile("/sys/class/dmi/id/product_name") +
                 ReadBoundedFile("/sys/class/dmi/id/board_vendor");
#if defined(__ANDROID__)
  const std::string android_properties = ReadBoundedFile("/system/build.prop");
  evidence.android_emulator = telemetry_detail::ContainsAscii(android_properties, "ro.kernel.qemu=1") ||
                              telemetry_detail::ContainsAscii(android_properties, "ro.boot.qemu=1") ||
                              telemetry_detail::ContainsAscii(android_properties, "ro.product.manufacturer=genymotion");
#endif
#elif defined(__APPLE__) && !TARGET_OS_IOS
  int is_virtual_machine = 0;
  size_t size = sizeof(is_virtual_machine);
  evidence.apple_virtual_machine =
      sysctlbyname("kern.hv_vmm_present", &is_virtual_machine, &size, nullptr, 0) == 0 &&
      is_virtual_machine != 0;
#endif
  return telemetry_detail::ClassifyHostEnvironment(evidence);
}

std::string OneDsTelemetry::GetProcessName() {
#if defined(__APPLE__)
  uint32_t path_size = 1024;
  std::vector<char> path(path_size);
  if (_NSGetExecutablePath(path.data(), &path_size) != 0) {
    if (path_size > telemetry_detail::kMaxTelemetryPathBytes) {
      return {};
    }
    path.resize(path_size);
    if (_NSGetExecutablePath(path.data(), &path_size) != 0) {
      return {};
    }
  }
  return GetFileName(path.data());
#elif defined(__linux__) || defined(__ANDROID__)
  std::istringstream cmdline(ReadBoundedFile("/proc/self/cmdline", telemetry_detail::kMaxTelemetryPathBytes + 1));
  std::string first_argument;
  if (cmdline && std::getline(cmdline, first_argument, '\0') &&
      first_argument.size() <= telemetry_detail::kMaxTelemetryPathBytes) {
    return GetFileName(first_argument);
  }
  return {};
#else
  return {};
#endif
}

// Get the CPU architecture the binary was compiled for
std::string OneDsTelemetry::GetArchitecture() {
#if defined(__x86_64__)
  return "x86_64";
#elif defined(__i386__)
  return "x86";
#elif defined(__aarch64__)
  return "arm64";
#elif defined(__arm__)
  return "arm";
#elif defined(__riscv)
  return "riscv";
#elif defined(__wasm__)
  return "wasm";
#else
  return "unknown";
#endif
}

// Get total physical memory in MB
int64_t OneDsTelemetry::GetTotalMemoryMB() {
#if defined(__APPLE__)
  int64_t mem = 0;
  size_t len = sizeof(mem);
  if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) {
    return mem / (1024 * 1024);
  }
  return 0;

#elif defined(__linux__) || defined(__ANDROID__)
  long pages = sysconf(_SC_PHYS_PAGES);
  long page_size = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && page_size > 0) {
    return static_cast<int64_t>(pages) * page_size / (1024 * 1024);
  }
  return 0;

#else
  return 0;
#endif
}

void OneDsTelemetry::LogExecutionProviderEvent(LUID* adapter_luid) const {
  (void)adapter_luid;
}

void OneDsTelemetry::LogDriverInfoEvent(
    std::string_view device_class,
    const std::wstring_view& driver_names,
    const std::wstring_view& driver_versions) const {
  (void)device_class;
  (void)driver_names;
  (void)driver_versions;
}

}  // namespace onnxruntime
