// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "test/perftest/test_configuration.h"
#include <core/session/onnxruntime_cxx_api.h>
#include <memory>
#include <optional>
#include <vector>

namespace onnxruntime {
namespace perftest {
namespace utils {

size_t GetPeakWorkingSetSize();

class ICPUUsage {
 public:
  virtual ~ICPUUsage() = default;

  virtual short GetUsage() const = 0;

  virtual void Reset() = 0;
};

std::unique_ptr<ICPUUsage> CreateICPUUsage();

std::vector<std::string> ConvertArgvToUtf8Strings(int argc, ORTCHAR_T* argv[]);

std::vector<char*> CStringsFromStrings(std::vector<std::string>& utf8_args);

void RegisterExecutionProviderLibrary(Ort::Env& env, PerformanceTestConfig& test_config);

void UnregisterExecutionProviderLibrary(Ort::Env& env, PerformanceTestConfig& test_config);

void ListEpDevices(const Ort::Env& env);

// Appends the plugin EP devices selected by the test config to the session options.
// Returns the list of OrtEpDevice instances that were added to the session.
std::vector<Ort::ConstEpDevice> AppendPluginExecutionProviders(Ort::Env& env,
                                                               Ort::SessionOptions& session_options,
                                                               const PerformanceTestConfig& test_config);

bool UsesNvidiaDevice(Ort::Env& env, const PerformanceTestConfig& test_config);

// An allocator selected for a plugin EP device, along with whether its memory can be written to
// directly from the host (e.g. CPU or pinned/shared memory), or whether it is device-only memory
// (e.g. plain GPU memory) that requires a device data transfer to populate from the host.
struct PluginEpAllocatorSelection {
  Ort::UnownedAllocator allocator;
  bool is_host_accessible = false;
};

// Selects an allocator to use for input/output buffers when running with plugin EP devices.
// Preference order: the EP device's default (device) allocator, then its host accessible allocator,
// then std::nullopt to indicate the caller should fall back to the CPU allocator.
std::optional<PluginEpAllocatorSelection> GetPluginEpAllocator(Ort::Env& env,
                                                               const std::vector<Ort::ConstEpDevice>& ep_devices);

}  // namespace utils
}  // namespace perftest
}  // namespace onnxruntime
