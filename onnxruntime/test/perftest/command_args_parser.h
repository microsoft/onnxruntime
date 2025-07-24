// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/session/onnxruntime_c_api.h>

namespace onnxruntime {
namespace perftest {

struct PerformanceTestConfig;

class CommandLineParser {
 public:
  static void ShowUsage();
  static bool ParseArguments(PerformanceTestConfig& test_config, int argc, ORTCHAR_T* argv[]);
};

}  // namespace perftest
}  // namespace onnxruntime
