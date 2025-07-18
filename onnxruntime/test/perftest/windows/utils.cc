// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/perftest/utils.h"
#include "test/perftest/strings_helper.h"
#include <core/platform/path_lib.h>

#include <cstdint>

#include <Windows.h>
#include <psapi.h>
#include <filesystem>

namespace onnxruntime {
namespace perftest {
namespace utils {

size_t GetPeakWorkingSetSize() {
  PROCESS_MEMORY_COUNTERS pmc;
  if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
    return pmc.PeakWorkingSetSize;
  }

  return 0;
}

static std::uint64_t SubtractFILETIME(const FILETIME& ft_a, const FILETIME& ft_b) {
  LARGE_INTEGER a, b;
  a.LowPart = ft_a.dwLowDateTime;
  a.HighPart = ft_a.dwHighDateTime;

  b.LowPart = ft_b.dwLowDateTime;
  b.HighPart = ft_b.dwHighDateTime;

  return a.QuadPart - b.QuadPart;
}

class CPUUsage : public ICPUUsage {
 public:
  CPUUsage() {
    Reset();
  }

  short GetUsage() const override {
    FILETIME sys_idle_ft, sys_kernel_ft, sys_user_ft, proc_creation_ft, proc_exit_ft, proc_kernel_ft, proc_user_ft;
    GetSystemTimes(&sys_idle_ft, &sys_kernel_ft, &sys_user_ft);
    GetProcessTimes(GetCurrentProcess(), &proc_creation_ft, &proc_exit_ft, &proc_kernel_ft, &proc_user_ft);

    std::uint64_t sys_kernel_ft_diff = SubtractFILETIME(sys_kernel_ft, sys_kernel_ft_);
    std::uint64_t sys_user_ft_diff = SubtractFILETIME(sys_user_ft, sys_user_ft_);

    std::uint64_t proc_kernel_diff = SubtractFILETIME(proc_kernel_ft, proc_kernel_ft_);
    std::uint64_t proc_user_diff = SubtractFILETIME(proc_user_ft, proc_user_ft_);

    std::uint64_t total_sys = sys_kernel_ft_diff + sys_user_ft_diff;
    std::uint64_t total_proc = proc_kernel_diff + proc_user_diff;

    return total_sys > 0 ? static_cast<short>((100.0 * total_proc) / total_sys) : 0;
  }

  void Reset() override {
    FILETIME sys_idle_ft, proc_creation_ft, proc_exit_ft;
    GetSystemTimes(&sys_idle_ft, &sys_kernel_ft_, &sys_user_ft_);
    GetProcessTimes(GetCurrentProcess(), &proc_creation_ft, &proc_exit_ft, &proc_kernel_ft_, &proc_user_ft_);
  }

 private:
  // system total times
  FILETIME sys_kernel_ft_;
  FILETIME sys_user_ft_;

  // process times
  FILETIME proc_kernel_ft_;
  FILETIME proc_user_ft_;
};

std::unique_ptr<ICPUUsage> CreateICPUUsage() {
  return std::make_unique<CPUUsage>();
}

#ifdef _WIN32
std::vector<std::string> ConvertArgvToUtf8Strings(int argc, wchar_t* argv[]) {
  std::vector<std::string> utf8_args;
  utf8_args.reserve(argc);
  for (int i = 0; i < argc; ++i) {
    utf8_args.push_back(ToUTF8String(argv[i]));
  }
  return utf8_args;
}

std::vector<const char*> ConvertArgvToUtf8CharPtrs(std::vector<std::string>& utf8_args) {
  std::vector<const char*> utf8_argv;
  utf8_argv.reserve(utf8_args.size());
  for (auto& str : utf8_args) {
    utf8_argv.push_back(&str[0]);  // safe since std::string is mutable
  }
  return utf8_argv;
}
#endif

std::basic_string<ORTCHAR_T> Utf8ToOrtString(const std::string& utf8_str) {
  // ORTCHAR_T == char -> just convert to std::basic_string<char>
  if constexpr (std::is_same_v<ORTCHAR_T, char>) {
    return std::basic_string<ORTCHAR_T>(utf8_str.begin(), utf8_str.end());
  }

  if (utf8_str.empty()) return std::basic_string<ORTCHAR_T>();

  int size_needed = MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, nullptr, 0);
  if (size_needed <= 0) return std::basic_string<ORTCHAR_T>();

  std::basic_string<ORTCHAR_T> wide_str(size_needed, 0);
  MultiByteToWideChar(CP_UTF8, 0, utf8_str.c_str(), -1, &wide_str[0], size_needed);
  wide_str.pop_back();  // Remove null terminator added by API

  return wide_str;
}

void list_devices(Ort::Env& env) {
  std::vector<Ort::ConstEpDevice> ep_devices = env.GetEpDevices();

  for (size_t i = 0; i < ep_devices.size(); ++i) {
    auto device = ep_devices[i];
    std::string device_info_msg = "===== device id " + std::to_string(i) + " ======\n";
    device_info_msg += "name: " + std::string(device.EpName()) + "\n";
    device_info_msg += "vendor: " + std::string(device.EpVendor()) + "\n";

    auto metadata = device.EpMetadata();
    std::unordered_map<std::string, std::string> metadata_entries = metadata.GetKeyValuePairs();
    if (!metadata_entries.empty()) {
      device_info_msg += "metadata:\n";
    }

    for (auto& entry : metadata_entries) {
      device_info_msg += "  " + entry.first + ": " + entry.second + "\n";
    }
    device_info_msg += "\n";
    fprintf(stdout, device_info_msg.c_str());
  }
}

bool RegisterExecutionProviderLibrary(Ort::Env& env, PerformanceTestConfig& test_config) {
  if (!test_config.plugin_ep_names_and_libs.empty()) {
    std::unordered_map<std::string, std::string> ep_names_to_libs;
    ParseSessionConfigs(ToUTF8String(test_config.plugin_ep_names_and_libs), ep_names_to_libs);
    if (ep_names_to_libs.size() > 0) {
      for (auto& pair : ep_names_to_libs) {
        const std::filesystem::path library_path = pair.second;
        const std::string registration_name = pair.first;
        env.RegisterExecutionProviderLibrary(registration_name.c_str(), Utf8ToWide(library_path.string()));
        test_config.registered_plugin_eps.push_back(registration_name);
      }
    }
  }
  return true;
}

bool UnregisterExecutionProviderLibrary(Ort::Env& env, PerformanceTestConfig& test_config) {
  for (auto& registration_name : test_config.registered_plugin_eps) {
    env.UnregisterExecutionProviderLibrary(registration_name.c_str());
  }
  return true;
}

}  // namespace utils
}  // namespace perftest
}  // namespace onnxruntime
