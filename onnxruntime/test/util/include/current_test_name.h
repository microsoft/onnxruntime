// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>

#include "gtest/gtest.h"

namespace onnxruntime::test {

// Returns the current test's name ("<test suite name>.<test name>") if a test is running, or "unknown".
inline std::string CurrentTestName() {
  const auto* const test_info = testing::UnitTest::GetInstance()->current_test_info();
  if (test_info == nullptr) {
    return "unknown";
  }
  return std::string{test_info->test_suite_name()} + "." + test_info->name();
}

}  // namespace onnxruntime::test
