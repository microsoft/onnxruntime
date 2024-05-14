// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "testcase_request.h"
#include "testenv.h"

#include "core/common/common.h"
#include <mutex>
#include <memory>
#include <condition_variable>
#include <vector>

class TestCaseResult;

namespace onnxruntime {
namespace test {

/// <summary>
/// Drives test execution
/// </summary>
class TestCaseDriver {
 public:
  /// <summary>
  /// Runs all the test cases sequentially. If there are several data tests
  /// in each test case they would be run in parallel.
  /// </summary>
  /// <param name="env">test environment</param>
  /// <param name="concurrent_runs">number of data tests within the model to run concurrently if,
  /// if 1, data tests are ran sequentially.
  /// </param>
  /// <param name="repeat_count">Repeat each data tests this many times (only for non-concurrent execution)</param>
  /// <returns>All tests results</returns>
  static std::vector<std::shared_ptr<TestCaseResult>> Run(const TestEnv& env,
                                                          size_t concurrent_runs, size_t repeat_count);

  /// <summary>
  /// Runs all test cases(models) concurrently but not more than
  /// parallel_models
  /// </summary>
  /// <param name="env">test environment</param>
  /// <param name="parallel_models">number of parallel models (test cases) to run concurrently</param>
  /// <param name="concurrent_runs">number of data tests to run concurrently on a specific test case(model)</param>
  /// <returns>All test results</returns>
  static std::vector<std::shared_ptr<TestCaseResult>> RunParallel(const TestEnv& env, size_t parallel_models,
                                                                  size_t concurrent_runs);

  ORT_DISALLOW_ASSIGNMENT(TestCaseDriver);

 private:
  TestCaseDriver(const TestEnv& env, size_t concurrent_runs);

  /// This makes the __Dtor private because the lifespan is managed by the class itself
  ~TestCaseDriver() = default;

  void RunModelsAsync(size_t parallel_models);

  void Wait() const;

  std::vector<std::shared_ptr<TestCaseResult>> TakeResults() {
    return std::move(results_);
  }

  void OnTestCaseComplete(size_t, std::shared_ptr<TestCaseResult>);

  const TestEnv& env_;
  size_t concurrent_runs_;
  std::vector<std::shared_ptr<TestCaseResult>> results_;
  TestCaseRequestContext::Callback on_test_case_complete_;

  mutable std::atomic_size_t tests_started_;
  mutable std::atomic_size_t tests_inprogress_;
  // This mutex is overloaded, serves both for results_ and finished
  // but that's okay since finished is only used once.
  mutable std::mutex mut_;
  mutable std::condition_variable cond_;
  mutable bool finished_ = false;
};

}  // namespace test
}  // namespace onnxruntime
