// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "gsl/gsl"

#include "gtest/gtest.h"

namespace onnxruntime::test {

// A test event listener that skips the specified tests.
class SkippingTestListener : public ::testing::EmptyTestEventListener {
 public:
  explicit SkippingTestListener(gsl::span<const std::string> tests_to_skip)
      : tests_to_skip_(tests_to_skip.begin(), tests_to_skip.end()) {
    for (const auto& test_to_skip : tests_to_skip) {
      if (test_to_skip.find_first_of("*?") != std::string::npos) {
        test_patterns_to_skip_.push_back(test_to_skip);
      }
    }
  }

 private:
  void OnTestStart(const ::testing::TestInfo& test_info) override {
    const auto full_test_name = std::string(test_info.test_suite_name()) + "." + test_info.name();
    if (tests_to_skip_.find(full_test_name) != tests_to_skip_.end() || MatchesAnyPattern(full_test_name)) {
      GTEST_SKIP();
    }
  }

  static bool MatchesPattern(const std::string& value, const std::string& pattern) {
    size_t value_index = 0;
    size_t pattern_index = 0;
    size_t star_index = std::string::npos;
    size_t value_after_star_index = 0;

    while (value_index < value.size()) {
      if (pattern_index < pattern.size() &&
          (pattern[pattern_index] == '?' || pattern[pattern_index] == value[value_index])) {
        ++value_index;
        ++pattern_index;
      } else if (pattern_index < pattern.size() && pattern[pattern_index] == '*') {
        star_index = pattern_index++;
        value_after_star_index = value_index;
      } else if (star_index != std::string::npos) {
        pattern_index = star_index + 1;
        value_index = ++value_after_star_index;
      } else {
        return false;
      }
    }

    while (pattern_index < pattern.size() && pattern[pattern_index] == '*') {
      ++pattern_index;
    }

    return pattern_index == pattern.size();
  }

  bool MatchesAnyPattern(const std::string& full_test_name) const {
    for (const auto& test_pattern_to_skip : test_patterns_to_skip_) {
      if (MatchesPattern(full_test_name, test_pattern_to_skip)) {
        return true;
      }
    }

    return false;
  }

  std::unordered_set<std::string> tests_to_skip_;
  std::vector<std::string> test_patterns_to_skip_;
};

}  // namespace onnxruntime::test
