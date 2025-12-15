// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_set>

#include "gsl/gsl"

#include "gtest/gtest.h"

namespace onnxruntime::test {

// A test event listener that skips the specified tests.
class SkippingTestListener : public ::testing::EmptyTestEventListener {
 public:
  explicit SkippingTestListener(gsl::span<const std::string> tests_to_skip)
      : tests_to_skip_(tests_to_skip.begin(), tests_to_skip.end()) {
  }

 private:
  void OnTestStart(const ::testing::TestInfo& test_info) override {
    const auto full_test_name = std::string(test_info.test_suite_name()) + "." + test_info.name();
    if (tests_to_skip_.find(full_test_name) != tests_to_skip_.end()) {
      GTEST_SKIP();
    }
  }

  std::unordered_set<std::string> tests_to_skip_;
};

}  // namespace onnxruntime::test
