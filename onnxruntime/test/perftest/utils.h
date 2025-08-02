// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "test/perftest/test_configuration.h"
#include <core/session/onnxruntime_cxx_api.h>
#include <memory>

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

#ifdef _WIN32
std::vector<std::string> ConvertArgvToUtf8Strings(int argc, wchar_t* argv[]);

std::vector<char*> CStringsFromStrings(std::vector<std::string>& utf8_args);
#endif

void RegisterExecutionProviderLibrary(Ort::Env& env, PerformanceTestConfig& test_config);

void UnregisterExecutionProviderLibrary(Ort::Env& env, PerformanceTestConfig& test_config);

void ListDevices(const Ort::Env& env);

std::string_view GetBasename(std::string_view filename);

}  // namespace utils
}  // namespace perftest
}  // namespace onnxruntime
