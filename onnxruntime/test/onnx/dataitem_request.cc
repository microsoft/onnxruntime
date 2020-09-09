// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "dataitem_request.h"

#include "TestCase.h"
#include "TestCaseResult.h"
#include "heap_buffer.h"

#include "core/common/logging/logging.h"
#include "core/common/common.h"
#include "core/platform/env.h"
#include "core/platform/threadpool.h"

#include <stdexcept>
#include <memory>

namespace onnxruntime {
namespace test {

EXECUTE_RESULT DataTaskRequestContext::Run(const ITestCase& c, ::Ort::Session& session,
                                           OrtAllocator* allocator, size_t task_id) {
  EXECUTE_RESULT result;
  Callback empty_cb;
  DataTaskRequestContext ctx(empty_cb, c, session, allocator, task_id);
  ORT_TRY {
    result = ctx.RunImpl();
  }
  ORT_CATCH(const ::std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      result = EXECUTE_RESULT::WITH_EXCEPTION;
      LOGS_DEFAULT(ERROR) << ctx.c_.GetTestCaseName() << ":" << ex.what();
    });
  }
  return result;
}

void DataTaskRequestContext::Request(const Callback& cb, concurrency::ThreadPool* tp,
                                     const ITestCase& c, Ort::Session& session,
                                     OrtAllocator* allocator, size_t task_id) {
  assert(cb);
  std::unique_ptr<DataTaskRequestContext> self(new DataTaskRequestContext(cb, c, session, allocator, task_id));
  CallableFactory<DataTaskRequestContext, void> f(self.get());
  auto runnable = f.GetCallable<&DataTaskRequestContext::RunAsync>();
  tp->Schedule([runnable]() { runnable.Invoke(); });
  self.release();
}

void DataTaskRequestContext::RunAsync() {
  EXECUTE_RESULT result;
  ORT_TRY {
    result = RunImpl();
  }
  ORT_CATCH(const ::std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      result = EXECUTE_RESULT::WITH_EXCEPTION;
      LOGS_DEFAULT(ERROR) << c_.GetTestCaseName() << ":" << ex.what();
    });
  }

  assert(cb_);
  std::unique_ptr<DataTaskRequestContext> self(this);
  cb_.Invoke(task_id_, result, spent_time_);
}

}  // namespace test
}  // namespace onnxruntime
