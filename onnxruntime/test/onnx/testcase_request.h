// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "callables.h"
#include "TestCaseResult.h"
#include "test_allocator.h"
#include <core/platform/env_time.h>
#include "core/platform/threadpool.h"
#include <core/session/onnxruntime_cxx_api.h>

#include <atomic>
#include <memory>
#include <mutex>
#include <condition_variable>

class ITestCase;

namespace onnxruntime {

namespace concurrency {
class ThreadPool;
}

using PThreadPool = onnxruntime::concurrency::ThreadPool*;

namespace test {

/// <summary>
/// Runs a single Test Case (model)
/// </summary>
class TestCaseRequestContext {
 public:
  using Callback = Callable<void, size_t, std::shared_ptr<TestCaseResult>>;

  /// <summary>
  /// Runs data tests on the model sequentially (concurrent_runs < 2)
  ///  and repeats them (repeat_count > 1) or concurrently (never repeats)
  /// repeat_count is ignored if concurrent_runs > 1 and a test case has more than
  /// one data task to run.
  /// </summary>
  /// <param name="tpool">ThreadPool</param>
  /// <param name="c">TestCase</param>
  /// <param name="env">test env</param>
  /// <param name="sf">SessionOptions from the command line</param>
  /// <param name="concurrent_runs">Number of data tests to run on the model concurrently</param>
  /// <param name="repeat_count">Repeat count for sequential execution</param>
  /// <returns>Test case result</returns>
  static std::shared_ptr<TestCaseResult> Run(PThreadPool tpool,
                                             const ITestCase& c, Ort::Env& env,
                                             const Ort::SessionOptions& sf,
                                             size_t concurrent_runs, size_t repeat_count);

  /// <summary>
  /// Schedules a TestCase to asynchronously on a TP. The function returns immediately.
  /// The completion is reported via  a supplied callback
  /// </summary>
  /// <param name="cb"></param>
  /// <param name="tpool"></param>
  /// <param name="c"></param>
  /// <param name="env"></param>
  /// <param name="sf"></param>
  /// <param name="concurrent_runs"></param>
  static void Request(const Callback& cb, PThreadPool tpool, const ITestCase& c,
                      Ort::Env& env, const Ort::SessionOptions& sf,
                      size_t test_case_id, size_t concurrent_runs);

  const TIME_SPEC& GetTimeSpend() const {
    return test_case_time_;
  }

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(TestCaseRequestContext);

  /// The lifespan of objects of the class is managed within the
  /// the class so the __dtor should really be private. However, on one
  /// occasion we use std:uniue_ptr to instantiate it so need public __dtor
  /// The impact is mitigated by the fact that __Ctor is still private.
  ~TestCaseRequestContext() = default;

  TestCaseRequestContext(const Callback& cb, PThreadPool tp, const ITestCase& test_case, Ort::Env& env,
                         const Ort::SessionOptions& session_opts, size_t test_case_id);

 private:
  bool SetupSession();

  std::shared_ptr<TestCaseResult> GetResult() const {
    return result_;
  }

  void RunSequentially(size_t repeat_count);
  // Wait for all concurrent data tasks to finish
  void Wait() const;

  void RunAsync(size_t concurrent_runs);
  // Callback for datatasks
  void OnDataTaskComplete(size_t task_id, EXECUTE_RESULT result, const TIME_SPEC& spent_time);
  void OnTestCaseComplete();

  void CalculateAndLogStats() const;

  Callback cb_;
  PThreadPool tp_;
  const ITestCase& test_case_;
  Ort::Env& env_;
  Ort::SessionOptions session_opts_;
  Ort::Session session_;
  size_t test_case_id_;
  MockedOrtAllocator allocator_;
  std::shared_ptr<TestCaseResult> result_;
  TIME_SPEC test_case_time_;
  Callable<void, size_t, EXECUTE_RESULT, const TIME_SPEC&> on_data_task_cb_;

  mutable std::atomic_size_t data_tasks_started_;
  mutable std::atomic_size_t data_tasks_inprogress_;
  mutable std::mutex mut_;
  mutable std::condition_variable cond_;
  mutable bool finished_ = false;
};

}  // namespace test
}  // namespace onnxruntime
