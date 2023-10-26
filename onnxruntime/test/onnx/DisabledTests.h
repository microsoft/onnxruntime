// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <memory>
#include <set>
#include <string>
#include <unordered_set>

#include "core/session/onnxruntime_c_api.h"

struct BrokenTest {
  std::string test_name_;
  std::string reason_;
  std::set<std::string> broken_opset_versions_ = {};  // apply to all versions if empty
  BrokenTest(std::string name, std::string reason) : test_name_(std::move(name)), reason_(std::move(reason)) {}
  BrokenTest(std::string name, std::string reason, const std::initializer_list<std::string>& versions) : test_name_(std::move(name)), reason_(std::move(reason)), broken_opset_versions_(versions) {}
  bool operator<(const struct BrokenTest& test) const {
    return strcmp(test_name_.c_str(), test.test_name_.c_str()) < 0;
  }
};

void LoadTests(const std::vector<std::basic_string<PATH_CHAR_TYPE>>& input_paths,
               const std::vector<std::basic_string<PATH_CHAR_TYPE>>& whitelisted_test_cases,
               const TestTolerances& tolerances,
               const std::unordered_set<std::basic_string<ORTCHAR_T>>& disabled_tests,
               std::unique_ptr<std::set<BrokenTest>> broken_test_list,
               std::unique_ptr<std::set<std::string>> broken_tests_keyword_set,
               const std::function<void(std::unique_ptr<ITestCase>)>& process_function);

std::unique_ptr<std::set<BrokenTest>> GetBrokenTests(const std::string& provider_name);

std::unique_ptr<std::set<std::string>> GetBrokenTestsKeyWordSet(const std::string& provider_name);

std::unique_ptr<std::set<std::string>> GetBrokenTestsKeyWordSet(const std::string& provider_name);

std::unique_ptr<std::unordered_set<std::basic_string<ORTCHAR_T>>> GetAllDisabledTests(const std::string& provider_name);
