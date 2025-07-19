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

std::vector<const char*> ConvertArgvToUtf8CharPtrs(std::vector<std::string>& utf8_args);
#endif

std::basic_string<ORTCHAR_T> Utf8ToOrtString(const std::string& utf8_str);

bool RegisterExecutionProviderLibrary(Ort::Env& env, PerformanceTestConfig& test_config);

bool UnregisterExecutionProviderLibrary(Ort::Env& env, PerformanceTestConfig& test_config);

void list_devices(Ort::Env& env);

}  // namespace utils
}  // namespace perftest
}  // namespace onnxruntime
