// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "callables.h"

#include "testenv.h"
#include "TestCase.h"
#include "TestCaseResult.h"
#include "test_allocator.h"
#include "core/common/logging/logging.h"
#include "core/common/common.h"
#include "core/platform/env.h"
#include <core/session/onnxruntime_cxx_api.h>
#include "pb_helper.h"
#include "test/compare_ortvalue.h"

#include "dataitem_request.h"

// Class that allows to run a single TestCase either sync or async.
namespace onnxruntime {
namespace test {

// This runs data tasks related to a single model in parallel
class TestCaseRequestContext {
 public:
  using Callback = Callable<void, std::shared_ptr<TestCaseResult>>;

  // Run a single test case. However, the individual data items may still run parallel
  static std::shared_ptr<TestCaseResult> RunTestCase(PThreadPool tpool,
                                                     const ITestCase& info, Ort::Env& env,
                                                     const Ort::SessionOptions& sf,
                                                     size_t concurrent_runs, size_t repeat_count);
  // Run async
  static void RequestRunTestCase(Callback cb, PThreadPool tpool, const ITestCase& info,
                                 Ort::Env& env, const Ort::SessionOptions& sf,
                                 size_t concurrent_runs, size_t repeat_count);

  const std::shared_ptr<TestCaseResult>& GetResult() const {
    return result_;
  }

  ~TestCaseRequestContext() = default;

 private:
  void RunSequentially(size_t p_models);

  Callback cb_;
  PThreadPool tp_;
  const ITestCase& c_;
  MockedOrtAllocator allocator_;
  Ort::Session session_{nullptr};
  std::shared_ptr<TestCaseResult> result_;
};

void TestCaseRequestContext::RunSequentially(size_t repeat_count) {
  const size_t data_count = c_.GetDataCount();
  for (size_t idx_repeat = 0; idx_repeat < repeat_count; ++idx_repeat) {
    for (size_t idx_data = 0; idx_data != data_count; ++idx_data) {
      auto result = DataTaskRequestContext::Run(c_, session_, &allocator_, idx_data);
      result_->SetResult(idx_data, result);
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

using onnxruntime::Status;
TestEnv::TestEnv(const std::vector<ITestCase*>& tests1, TestResultStat& stat1, Ort::Env& env1,
                 Ort::SessionOptions& sf1, PThreadPool tp)
    : tests(tests1),
      next_test_to_run(0),
      stat(stat1),
      // finished(new FixedCountFinishCallback<TestCaseResult>(static_cast<int>(tests1.size()))),
      env(env1),
      sf(sf1),
      tp_(tp) {
}

TestEnv::~TestEnv() {
  // need dtor in .cc so 'finished' can be cleaned up as TestCaseResult only has a forward declaration in the header.
}
