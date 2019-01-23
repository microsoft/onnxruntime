// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstdint>
#include <string>

#include "core/graph/constants.h"

namespace onnxruntime {
namespace perftest {

enum class TestMode : std::uint8_t {
  kFixDurationMode = 0,
  KFixRepeatedTimesMode
};

enum class Platform : std::uint8_t {
  kWindows = 0,
  kLinux
};

struct ModelInfo {
  std::string model_name;
  std::string model_file_path;
  std::string input_file_path;
  std::string result_file_path;
};

struct MachineConfig {
  Platform platform{Platform::kWindows};
  std::string provider_type_name{onnxruntime::kCpuExecutionProvider};
};

struct RunConfig {
  std::string profile_file;
  TestMode test_mode{TestMode::kFixDurationMode};
  size_t repeated_times{1000};
  size_t duration_in_seconds{600};
  size_t concurrent_run{0};
  bool f_dump_statistics{false};
  bool f_verbose{false};
  bool enable_sequential_execution{true};
};

struct PerformanceTestConfig {
  ModelInfo model_info;
  MachineConfig machine_config;
  RunConfig run_config;
};

}  // namespace perftest
}  // namespace onnxruntime
