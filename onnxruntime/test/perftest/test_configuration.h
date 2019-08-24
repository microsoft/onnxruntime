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
  std::basic_string<ORTCHAR_T> model_file_path;
  std::basic_string<ORTCHAR_T> input_file_path;
  std::basic_string<ORTCHAR_T> result_file_path;
};

struct MachineConfig {
  Platform platform{Platform::kWindows};
  std::string provider_type_name{onnxruntime::kCpuExecutionProvider};
};

struct RunConfig {
  std::basic_string<ORTCHAR_T> profile_file;
  TestMode test_mode{TestMode::kFixDurationMode};
  size_t repeated_times{1000};
  size_t duration_in_seconds{600};
  size_t concurrent_session_runs{1};
  bool f_dump_statistics{false};
  bool f_verbose{false};
  bool enable_memory_pattern{true};
  bool enable_cpu_mem_arena{true};
  bool enable_sequential_execution{true};
  int session_thread_pool_size{-1};
  GraphOptimizationLevel optimization_level{ORT_ENABLE_EXTENDED};
};

struct PerformanceTestConfig {
  ModelInfo model_info;
  MachineConfig machine_config;
  RunConfig run_config;
  std::basic_string<ORTCHAR_T> backend = ORT_TSTR("ort");
};

}  // namespace perftest
}  // namespace onnxruntime
