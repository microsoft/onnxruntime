// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace perftest {

struct PerformanceTestConfig;

class CommandLineParser {
 public:
  static void ShowUsage();

  static bool ParseArguments(PerformanceTestConfig& test_config, int argc, char* argv[]);
};

}  // namespace perftest
}  // namespace onnxruntime
