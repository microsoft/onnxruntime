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

// Returns the OrtEpDevice instances that were added to the session.
// If compute_stream is non-null, a sync stream is created for the first EP that has a single selected device and
// supports sync streams. It is stored in *compute_stream and passed to that EP via the user_compute_stream option.
std::vector<Ort::ConstEpDevice> AppendPluginExecutionProviders(Ort::Env& env,
                                                               Ort::SessionOptions& session_options,
                                                               const PerformanceTestConfig& test_config,
                                                               Ort::SyncStream* compute_stream = nullptr);

struct PluginEpAllocatorSelection {
  Ort::UnownedAllocator allocator;
  bool is_host_accessible = false;
};

// Preference order: default (device) allocator, then host accessible allocator, then nullopt
// (caller falls back to the CPU allocator).
std::optional<PluginEpAllocatorSelection> GetPluginEpAllocator(Ort::Env& env,
                                                               const std::vector<Ort::ConstEpDevice>& ep_devices);

}  // namespace utils
}  // namespace perftest
}  // namespace onnxruntime
