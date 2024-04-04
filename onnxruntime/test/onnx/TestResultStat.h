// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdlib.h>
#include <unordered_set>
#include <string>
#include <atomic>
#include <core/platform/ort_mutex.h>
#include <cstring>
#include <set>

class TestResultStat {
 public:
  size_t total_test_case_count = 0;
  size_t total_model_count = 0;
  std::atomic_int succeeded;
  std::atomic_int not_implemented;
  std::atomic_int load_model_failed;
  std::atomic_int throwed_exception;
  std::atomic_int result_differs;
  std::atomic_int skipped;
  std::atomic_int invalid_graph;

  TestResultStat() : succeeded(0), not_implemented(0), load_model_failed(0), throwed_exception(0), result_differs(0), skipped(0), invalid_graph(0) {}

  void AddNotImplementedKernels(const std::string& s) {
    std::lock_guard<onnxruntime::OrtMutex> l(m_);
    not_implemented_kernels.insert(s);
  }

  void AddFailedKernels(const std::string& s) {
    std::lock_guard<onnxruntime::OrtMutex> l(m_);
    failed_kernels.insert(s);
  }

  void AddFailedTest(const std::pair<std::string, std::string>& p) {
    std::lock_guard<onnxruntime::OrtMutex> l(m_);
    failed_test_cases.insert(p);
  }

  const std::set<std::pair<std::string, std::string>>& GetFailedTest() const {
    std::lock_guard<onnxruntime::OrtMutex> l(m_);
    return failed_test_cases;
  }

  std::string ToString();

  TestResultStat& operator+=(const TestResultStat& result) {
    total_test_case_count += result.total_test_case_count;
    total_model_count += result.total_model_count;
    succeeded += result.succeeded;
    not_implemented += result.not_implemented;
    load_model_failed += result.load_model_failed;
    throwed_exception += result.throwed_exception;
    result_differs += result.result_differs;
    skipped += result.skipped;
    invalid_graph += result.invalid_graph;

    for (const auto& s : result.not_implemented_kernels) {
      AddNotImplementedKernels(s);
    }

    for (const auto& s : result.failed_kernels) {
      AddFailedKernels(s);
    }

    for (const auto& p : result.failed_test_cases) {
      AddFailedTest(p);
    }

    return *this;
  }

 private:
  mutable onnxruntime::OrtMutex m_;
  std::unordered_set<std::string> not_implemented_kernels;
  std::unordered_set<std::string> failed_kernels;
  std::set<std::pair<std::string, std::string>> failed_test_cases;  // pairs of test name and version
};
