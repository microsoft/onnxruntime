// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <core/session/onnxruntime_c_api.h>

namespace onnxruntime {
namespace qnnctxgen {

struct TestConfig;

class CommandLineParser {
 public:
  static void ShowUsage();
  static bool ParseArguments(TestConfig& test_config, int argc, ORTCHAR_T* argv[]);
};

}  // namespace qnnctxgen
}  // namespace onnxruntime
