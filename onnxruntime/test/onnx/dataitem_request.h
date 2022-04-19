// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "callables.h"
#include "TestCaseResult.h"
#include "core/common/common.h"
#include "core/platform/env_time.h"

class ITestCase;
struct OrtAllocator;

namespace Ort {
struct Session;
}

namespace onnxruntime {
namespace concurrency {
class ThreadPool;
}

namespace test {

/// <summary>
/// This runs a single DataTask either on a threadpool or
/// in the same thread
/// </summary>
class DataTaskRequestContext {
 public:
  using Callback = Callable<void, size_t, EXECUTE_RESULT, const TIME_SPEC&>;

  /// <summary>
  /// Runs a single data task on the same thread
  /// </summary>
  /// <param name="c">TestCase</param>
  /// <param name="session">session handle</param>
  /// <param name="allocator">allocator to use</param>
  /// <param name="task_id">this task id</param>
  /// <returns>execution result and elapsed time</returns>
  static std::pair<EXECUTE_RESULT, TIME_SPEC> Run(const ITestCase& c, ::Ort::Session& session,
                                                  OrtAllocator* allocator, size_t task_id);

  /// <summary>
  /// Schedules a data task to run on a threadpool. The function
  /// returns immediately. Completion is reported via  callback
  /// </summary>
  /// <param name="cb">callback that will be invoked on completion</param>
  /// <param name="tp">threadpool</param>
  /// <param name="c">testcase</param>
  /// <param name="session">session handle</param>
  /// <param name="allocator">allocator to use</param>
  /// <param name="task_id">this taks id</param>
  static void Request(const Callback& cb, concurrency::ThreadPool* tp,
                      const ITestCase& c, ::Ort::Session& session,
                      OrtAllocator* allocator, size_t task_id);

  ORT_DISALLOW_COPY_AND_ASSIGNMENT(DataTaskRequestContext);

  /// The lifespan of objects of the class is managed within the
  /// the class so the __dtor should really be private. However, on one
  /// occasion we use std:uniue_ptr to instantiate it so need public __dtor
  /// The impact is mitigated by the fact that __Ctor is still private.
  ~DataTaskRequestContext() = default;

  const TIME_SPEC& GetTimeSpent() const {
    return spent_time_;
  }

  DataTaskRequestContext(const Callback& cb,
                         const ITestCase& test_case, ::Ort::Session& session,
                         OrtAllocator* allocator, size_t task_id)
      : cb_(cb),
        test_case_(test_case),
        session_(session),
        default_allocator_(allocator),
        task_id_(task_id) {
    SetTimeSpecToZero(&spent_time_);
  }

 private:
  void RunAsync();
  std::pair<EXECUTE_RESULT, TIME_SPEC> RunImpl();

  Callback cb_;
  const ITestCase& test_case_;
  Ort::Session& session_;
  OrtAllocator* default_allocator_;
  size_t task_id_;
  TIME_SPEC spent_time_;
};

}  // namespace test
}  // namespace onnxruntime
