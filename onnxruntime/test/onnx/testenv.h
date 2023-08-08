// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include <vector>
#include <core/common/common.h>

class ITestCase;
class TestCaseResult;
class TestResultStat;

namespace Ort {
struct Env;
struct SessionOptions;
}  // namespace Ort

namespace onnxruntime {
class Env;
namespace common {
class Status;
}
namespace concurrency {
class ThreadPool;
}
}  // namespace onnxruntime

using OrtThreadPool = onnxruntime::concurrency::ThreadPool;
using PThreadPool = OrtThreadPool*;

/// <summary>
/// Facilitates running tests
/// </summary>
class TestEnv {
 public:
  TestEnv(Ort::Env& env, Ort::SessionOptions& sf1, PThreadPool tp,
          std::vector<ITestCase*>&& tests, TestResultStat& stat1);

  ~TestEnv();

  static std::unique_ptr<OrtThreadPool> CreateThreadPool(onnxruntime::Env& env);

  /// <summary>
  /// Runs all tests cases either concurrently or sequentially
  /// </summary>
  /// <param name="models">parallel_models - number of models to run concurrently</param>
  /// <param name="concurrent_runs"></param>
  /// <param name="repeat_count"></param>
  /// <returns></returns>
  onnxruntime::common::Status Run(size_t parallel_models, int concurrent_runs, size_t repeat_count);

  Ort::Env& Env() const {
    return env_;
  }

  const Ort::SessionOptions& GetSessionOptions() const {
    return so_;
  }

  PThreadPool GetThreadPool() const {
    return tp_;
  }

  const std::vector<ITestCase*>& GetTests() const {
    return tests_;
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TestEnv);

 private:
  void CalculateStats(const std::vector<std::shared_ptr<TestCaseResult>>&);

  Ort::Env& env_;
  const Ort::SessionOptions& so_;
  PThreadPool tp_;
  std::vector<ITestCase*> tests_;
  TestResultStat& stat_;
};
