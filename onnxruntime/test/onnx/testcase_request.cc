// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testcase_request.h"
#include "dataitem_request.h"
#include "TestCase.h"

#include "core/common/logging/logging.h"
#include "core/common/logging/macros.h"

#include <utility>

namespace onnxruntime {
namespace test {

TestCaseRequestContext::TestCaseRequestContext(const Callback& cb, PThreadPool tp, const ITestCase& test_case, Ort::Env& env,
                                               const Ort::SessionOptions& session_opts, size_t test_case_id)
    : cb_(cb),
      tp_(tp),
      test_case_(test_case),
      env_(env),
      session_opts_(session_opts.Clone()),
      session_(nullptr),
      test_case_id_(test_case_id),
      allocator_(),
      result_(),
      data_tasks_started_(0),
      data_tasks_inprogress_(0) {
  result_ = std::make_shared<TestCaseResult>(test_case_.GetDataCount(), EXECUTE_RESULT::NOT_SET, test_case_.GetTestCaseName());
  SetTimeSpecToZero(&test_case_time_);
  CallableFactory<TestCaseRequestContext, void, size_t, EXECUTE_RESULT, const TIME_SPEC&> f(this);
  on_data_task_cb_ = f.GetCallable<&TestCaseRequestContext::OnDataTaskComplete>();
}

bool TestCaseRequestContext::SetupSession() {
  ORT_TRY {
    const auto* test_case_name = test_case_.GetTestCaseName().c_str();
    session_opts_.SetLogId(test_case_name);
    Ort::Session session{env_, test_case_.GetModelUrl().native().c_str(), session_opts_};
    session_ = std::move(session);
    LOGF_DEFAULT(INFO, "Testing %s\n", test_case_name);
    return true;
  }
  ORT_CATCH(const Ort::Exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      LOGF_DEFAULT(ERROR, "Model %s failed to load:%s", test_case_.GetTestCaseName().c_str(), ex.what());
      result_ = std::make_shared<TestCaseResult>(test_case_.GetDataCount(), EXECUTE_RESULT::NOT_SUPPORT, "");
    });
  }
  return false;
}

std::shared_ptr<TestCaseResult> TestCaseRequestContext::Run(PThreadPool tpool,
                                                            const ITestCase& c, Ort::Env& env,
                                                            const Ort::SessionOptions& session_opts,
                                                            size_t concurrent_runs, size_t repeat_count) {
  // temp hack. Because we have no resource control. We may not have enough memory to run this test in parallel
  if (c.GetTestCaseName() == "coreml_FNS-Candy_ImageNet") {
    concurrent_runs = 1;
  }

  // No callback, test_case_id is zero.
  Callback empty_cb;
  TestCaseRequestContext ctx(empty_cb, tpool, c, env, session_opts, 0U);

  const size_t data_count = c.GetDataCount();
  if (concurrent_runs > 1 && data_count > 1) {
    ctx.RunAsync(concurrent_runs);
    ctx.Wait();
  } else {
    ctx.RunSequentially(repeat_count);
  }
  auto result = ctx.GetResult();
  return result;
}

void TestCaseRequestContext::Request(const Callback& cb, PThreadPool tpool,
                                     const ITestCase& c,
                                     Ort::Env& env,
                                     const Ort::SessionOptions& session_opts,
                                     size_t test_case_id,
                                     size_t concurrent_runs) {
  // temp hack. Because we have no resource control. We may not have enough memory to run this test in parallel
  if (c.GetTestCaseName() == "coreml_FNS-Candy_ImageNet") {
    concurrent_runs = 1;
  }

  std::unique_ptr<TestCaseRequestContext> self = std::make_unique<TestCaseRequestContext>(cb, tpool, c, env, session_opts, test_case_id);
  CallableFactory<TestCaseRequestContext, void, size_t> f(self.get());
  auto runnable = f.GetCallable<&TestCaseRequestContext::RunAsync>();
  onnxruntime::concurrency::ThreadPool::Schedule(tpool, [runnable, concurrent_runs]() { runnable.Invoke(concurrent_runs); });
  self.release();
}

void TestCaseRequestContext::RunAsync(size_t concurrent_runs) {
  assert(concurrent_runs > 0);

  if (!SetupSession()) {
    return OnTestCaseComplete();
  }

  concurrent_runs = std::min(concurrent_runs, test_case_.GetDataCount());
  // Reserve one refcount for this thread so the object does not get deleted when
  // several TestCases are run in parallel
  // by worker threads before this thread finishes. In exchange, we run one of the tasks.
  auto this_task_id = data_tasks_started_.fetch_add(1, std::memory_order_relaxed);
  data_tasks_inprogress_.fetch_add(1, std::memory_order_relaxed);

  for (size_t i = 1; i < concurrent_runs; ++i) {
    auto next_to_run = data_tasks_started_.fetch_add(1, std::memory_order_relaxed);
    if (next_to_run >= test_case_.GetDataCount()) {
      break;
    }
    data_tasks_inprogress_.fetch_add(1, std::memory_order_relaxed);
    DataTaskRequestContext::Request(on_data_task_cb_, tp_, test_case_, session_, &allocator_, next_to_run);
  }
  // This runs in this thread and we should invoke the callback for it.
  auto result = DataTaskRequestContext::Run(test_case_, session_, &allocator_, this_task_id);
  OnDataTaskComplete(this_task_id, result.first, result.second);
}

void TestCaseRequestContext::OnDataTaskComplete(size_t task_id, EXECUTE_RESULT result, const TIME_SPEC& spent_time) {
  TIME_SPEC zero;
  SetTimeSpecToZero(&zero);
  AccumulateTimeSpec(&test_case_time_, &zero, &spent_time);
  result_->SetResult(task_id, result);

  auto next_to_run = data_tasks_started_.fetch_add(1, std::memory_order_relaxed);
  if (next_to_run < test_case_.GetDataCount()) {
    data_tasks_inprogress_.fetch_add(1, std::memory_order_relaxed);
    DataTaskRequestContext::Request(on_data_task_cb_, tp_, test_case_, session_, &allocator_, next_to_run);
  }

  auto before_we_done = data_tasks_inprogress_.fetch_sub(1, std::memory_order_acq_rel);
  assert(before_we_done > 0);
  if (before_we_done == 1U) {
    CalculateAndLogStats();
    OnTestCaseComplete();
  }
}

void TestCaseRequestContext::OnTestCaseComplete() {
  if (cb_) {
    std::unique_ptr<TestCaseRequestContext> self(this);
    cb_.Invoke(test_case_id_, std::move(result_));
    // No member access beyond this point
  } else {
    std::lock_guard<std::mutex> g(mut_);
    finished_ = true;
    // We do not unlock here before notifying
    // so the Waiting thread does not destroy us before
    // we access cond_ in case it discovers finished_ is already true
    cond_.notify_one();
  }
}

void TestCaseRequestContext::RunSequentially(size_t repeat_count) {
  if (!SetupSession()) {
    return;
  }

  TIME_SPEC zero;
  SetTimeSpecToZero(&zero);

  const size_t data_count = test_case_.GetDataCount();
  for (size_t idx_repeat = 0; idx_repeat < repeat_count; ++idx_repeat) {
    for (size_t idx_data = 0; idx_data != data_count; ++idx_data) {
      auto result = DataTaskRequestContext::Run(test_case_, session_, &allocator_, idx_data);
      result_->SetResult(idx_data, result.first);
      AccumulateTimeSpec(&test_case_time_, &zero, &result.second);
    }
  }

  CalculateAndLogStats();
}

void TestCaseRequestContext::Wait() const {
  std::unique_lock<std::mutex> ul(mut_);
  while (!finished_) {
    cond_.wait(ul);
  }
}

void TestCaseRequestContext::CalculateAndLogStats() const {
  result_->SetSpentTime(test_case_time_);
  const auto& test_case_name = test_case_.GetTestCaseName();
  const std::vector<EXECUTE_RESULT>& er = result_->GetExcutionResult();
  for (size_t i = 0; i != er.size(); ++i) {
    EXECUTE_RESULT r = er[i];
    if (r == EXECUTE_RESULT::SUCCESS) continue;
    std::string s = test_case_.GetDatasetDebugInfoString(i);
    switch (r) {
      case EXECUTE_RESULT::RESULT_DIFFERS:
        LOGF_DEFAULT(ERROR, "%s: result differs. Dataset:%s\n", test_case_name.c_str(), s.c_str());
        break;
      case EXECUTE_RESULT::SHAPE_MISMATCH:
        LOGF_DEFAULT(ERROR, "%s: shape mismatch. Dataset:%s\n", test_case_name.c_str(), s.c_str());
        break;
      case EXECUTE_RESULT::TYPE_MISMATCH:
        LOGF_DEFAULT(ERROR, "%s: type mismatch. Dataset:%s\n", test_case_name.c_str(), s.c_str());
        break;
      case EXECUTE_RESULT::MODEL_SHAPE_MISMATCH:
        LOGF_DEFAULT(ERROR, "%s: shape in model file mismatch. Dataset:%s\n", test_case_name.c_str(), s.c_str());
        break;
      case EXECUTE_RESULT::MODEL_TYPE_MISMATCH:
        LOGF_DEFAULT(ERROR, "%s: type in model file mismatch. Dataset:%s\n", test_case_name.c_str(), s.c_str());
        break;
      default:
        // nothing to do
        break;
    }
    break;
  }
}
}  // namespace test
}  // namespace onnxruntime
