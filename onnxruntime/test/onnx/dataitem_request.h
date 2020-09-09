// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "callables.h"
#include "core/platform/env_time.h"

enum EXECUTE_RESULT;

class ITestCase;
struct OrtAllocator;

namespace Ort {
struct Session;
}

// Class that allows to run a single TestCase either sync or async.
namespace onnxruntime {

namespace concurrency {
class ThreadPool;
}

namespace test {

// This runs a single DataTask on a threadpool and
// invokes a callback to TestCaseRequestContext that orchestrates
// the DataTasks related to the model
class DataTaskRequestContext {
 public:
  // This is a callback that will be invoked by the individual task
  // when it completes
  using Callback = Callable<void, size_t, EXECUTE_RESULT, const TIME_SPEC&>;
  static EXECUTE_RESULT Run(const ITestCase& c, ::Ort::Session& session,
                            OrtAllocator* allocator, size_t task_id);

  static void Request(const Callback& cb, concurrency::ThreadPool* tp,
                      const ITestCase& c, ::Ort::Session& session,
                      OrtAllocator* allocator, size_t task_id);

  DataTaskRequestContext(const DataTaskRequestContext&) = delete;
  DataTaskRequestContext& operator=(const DataTaskRequestContext&) = delete;

  ~DataTaskRequestContext() = default;

 private:
  DataTaskRequestContext(const Callback& cb,
                         const ITestCase& c, ::Ort::Session& session,
                         OrtAllocator* allocator, size_t task_id)
      : cb_(cb),
        c_(c),
        session_(session),
        default_allocator_(allocator),
        task_id_(task_id) {
    SetTimeSpecToZero(&spent_time_);
  }

  void RunAsync();
  EXECUTE_RESULT RunImpl();

  Callback cb_;
  const ITestCase& c_;
  ::Ort::Session& session_;
  OrtAllocator* default_allocator_;
  size_t task_id_;
  TIME_SPEC spent_time_;
};
