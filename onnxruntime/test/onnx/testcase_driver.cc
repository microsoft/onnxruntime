// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "TestCaseResult.h"
#include "testcase_request.h"
#include "testenv.h"

#include <mutex>
#include <condition_variable>
#include <vector>

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

  TestCaseDriver(const TestEnv& env)
      : env_(env) {
    CallableFactory<TestCaseDriver, void, const std::shared_ptr<TestCaseResult>&> f(this);
    on_test_case_complete_ = f.GetCallable<&TestCaseDriver::OnTestCaseComplete>();
  }

  ~TestCaseDriver() = default;

  void RunModelsAsync(size_t parallel_models, size_t concurrent_runs);

  void Wait() const;

  const std::vector<std::shared_ptr<TestCaseResult>>& GetResults() const {
    return results_;
  }

  void OnTestCaseComplete(const std::shared_ptr<TestCaseResult>&);

  const TestEnv& env_;
  std::vector<std::shared_ptr<TestCaseResult>>  results_;
  TestCaseRequestContext::Callback on_test_case_complete_;

  mutable std::atomic_size_t tests_started_;
  mutable std::atomic_size_t tests_inprogress_;
  mutable std::mutex mut_;
  mutable std::condition_variable cond_;
  mutable bool finished_ = false;
};

std::vector<std::shared_ptr<TestCaseResult>> TestCaseDriver::Run(const TestEnv& env, size_t concurrent_runs, size_t repeat_count) {
  std::vector<std::shared_ptr<TestCaseResult>> results;
  for (const auto& c : env.GetTests()) {
    auto result = TestCaseRequestContext::Run(env.GetThreadPool(),
         *c, env.Env(), env.GetSessionOptions(), concurrent_runs, repeat_count);
    results.push_back(std::move(result));
  }
  return results;
}

std::vector<std::shared_ptr<TestCaseResult>> TestCaseDriver::RunParallel(const TestEnv& env, size_t parallel_models, 
  size_t concurrent_runs) {

  assert(parallel_models > 1);
  parallel_models = std::min(parallel_models, env.GetTests().size());

  TestCaseDriver driver(env);
  driver.RunModelsAsync(parallel_models, concurrent_runs);
  driver.Wait();
  auto results = driver.GetResults();
  return results;
}

void TestCaseDriver::RunModelsAsync(size_t parallel_models, size_t concurrent_runs) {
  const auto& tests = env_.GetTests();
  const auto total_models = tests.size();
  for (size_t i = 0; i < parallel_models; ++i) {
    auto next_to_run = tests_started_.fetch_add(1, std::memory_order_relaxed);
    if (next_to_run < total_models) {
      tests_inprogress_.fetch_add(1, std::memory_order_relaxed);
      TestCaseRequestContext::Request(on_test_case_complete_, env_.GetThreadPool(), *tests[i],
        env_.Env(), env_.GetSessionOptions(), concurrent_runs);
    } else {
      break;
    }
  }
}

void onnxruntime::test::TestCaseDriver::OnTestCaseComplete(const std::shared_ptr<TestCaseResult>& result) {
}

void TestCaseDriver::Wait() const {
  std::unique_lock<std::mutex> ul(mut_);
  while (!finished_) {
    cond_.wait(ul);
  }
}

}  // namespace test
}  // namespace onnxruntime
