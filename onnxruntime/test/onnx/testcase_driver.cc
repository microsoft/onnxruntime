// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "TestCaseResult.h"
#include "testcase_driver.h"
#include "testcase_request.h"
#include "testenv.h"

#include <core/common/logging/logging.h>

namespace onnxruntime {
namespace test {

TestCaseDriver::TestCaseDriver(const TestEnv& env, size_t concurrent_runs)
    : env_(env),
      concurrent_runs_(concurrent_runs),
      tests_started_(0),
      tests_inprogress_(0),
      finished_(false) {
  results_.resize(env.GetTests().size());
  CallableFactory<TestCaseDriver, void, size_t, std::shared_ptr<TestCaseResult>> f(this);
  on_test_case_complete_ = f.GetCallable<&TestCaseDriver::OnTestCaseComplete>();
}

std::vector<std::shared_ptr<TestCaseResult>> TestCaseDriver::Run(const TestEnv& env, size_t concurrent_runs, size_t repeat_count) {
  std::vector<std::shared_ptr<TestCaseResult>> results;
  for (const auto& c : env.GetTests()) {
    auto result = TestCaseRequestContext::Run(env.GetThreadPool(),
                                              *c, env.Env(), env.GetSessionOptions(), concurrent_runs, repeat_count);
    results.push_back(std::move(result));
  }
  return results;
}

std::vector<std::shared_ptr<TestCaseResult>> TestCaseDriver::RunParallel(const TestEnv& test_env, size_t parallel_models,
                                                                         size_t concurrent_runs) {
  assert(parallel_models > 1);
  parallel_models = std::min(parallel_models, test_env.GetTests().size());
  LOGF_DEFAULT(ERROR, "Running tests in parallel: at most %u models at any time", static_cast<unsigned int>(parallel_models));
  TestCaseDriver driver(test_env, concurrent_runs);
  driver.RunModelsAsync(parallel_models);
  auto results = driver.TakeResults();
  return results;
}

void TestCaseDriver::RunModelsAsync(size_t parallel_models) {
  const auto& tests = env_.GetTests();
  const auto total_models = tests.size();
  for (size_t i = 0; i < parallel_models; ++i) {
    auto next_to_run = tests_started_.fetch_add(1, std::memory_order_relaxed);
    if (next_to_run >= total_models) {
      break;
    }
    tests_inprogress_.fetch_add(1, std::memory_order_relaxed);
    TestCaseRequestContext::Request(on_test_case_complete_, env_.GetThreadPool(), *tests[next_to_run],
                                    env_.Env(), env_.GetSessionOptions(), next_to_run, concurrent_runs_);
  }
  // This thread is not on a threadpool so we are not using it
  // to run anything. Just wait.
  Wait();
  LOGF_DEFAULT(ERROR, "Running tests finished. Generating report");
}

void TestCaseDriver::OnTestCaseComplete(size_t test_case_id, std::shared_ptr<TestCaseResult> result) {
  assert(test_case_id < results_.size());
  {
    std::lock_guard<std::mutex> g(mut_);
    results_[test_case_id] = std::move(result);
  }

  const auto& tests = env_.GetTests();
  const auto total_models = tests.size();
  auto next_to_run = tests_started_.fetch_add(1, std::memory_order_relaxed);
  if (next_to_run < total_models) {
    tests_inprogress_.fetch_add(1, std::memory_order_relaxed);
    TestCaseRequestContext::Request(on_test_case_complete_, env_.GetThreadPool(), *tests[next_to_run],
                                    env_.Env(), env_.GetSessionOptions(), next_to_run, concurrent_runs_);
  }

  auto before_we_done = tests_inprogress_.fetch_sub(1, std::memory_order_acq_rel);
  assert(before_we_done > 0);
  if (before_we_done == 1U) {
    std::lock_guard<std::mutex> g(mut_);
    finished_ = true;
    cond_.notify_one();
  }
}

void TestCaseDriver::Wait() const {
  std::unique_lock<std::mutex> ul(mut_);
  while (!finished_) {
    cond_.wait(ul);
  }
}

}  // namespace test
}  // namespace onnxruntime
