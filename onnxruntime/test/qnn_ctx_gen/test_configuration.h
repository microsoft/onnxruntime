// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <map>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "core/graph/constants.h"
#include "core/framework/session_options.h"

namespace onnxruntime {
namespace qnnctxgen {

struct RunConfig {
  bool f_verbose{false};
  GraphOptimizationLevel optimization_level{ORT_ENABLE_ALL};
  std::basic_string<ORTCHAR_T> optimized_model_path;
  std::unordered_map<std::string, std::string> session_config_entries;
  std::unordered_map<std::string, std::string> qnn_options;
};

struct PerformanceTestConfig {
  std::vector<std::basic_string<ORTCHAR_T>> model_file_paths;
  RunConfig run_config;
};

}  // namespace qnnctxgen
}  // namespace onnxruntime
