// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/session/onnxruntime_c_api.h>
#include <string>
#include <unordered_map>

namespace onnxruntime {
namespace perftest {

struct PerformanceTestConfig;

class CommandLineParser {
 public:
  static void ShowUsage();
  static bool ParseArguments(PerformanceTestConfig& test_config, int argc, ORTCHAR_T* argv[]);
  static bool ParseProviderOptions(const std::string& options_string,
                                   std::unordered_map<std::string, std::string>& options);
};

}  // namespace perftest
}  // namespace onnxruntime
