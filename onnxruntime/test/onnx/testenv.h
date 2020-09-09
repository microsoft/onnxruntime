// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include <vector>
#include "TestResultStat.h"
#include <core/common/common.h>

class ITestCase;
class TestCaseResult;

namespace Ort {
struct Env;
struct SessionOptions;
}

namespace Eigen {
class ThreadPoolInterface;
}

using PThreadPool = Eigen::ThreadPoolInterface*;

class TestEnv {
 public:
  TestEnv(const std::vector<ITestCase*>& tests, TestResultStat& stat1, Ort::Env& env, Ort::SessionOptions& sf1, 
    PThreadPool tp);
  ~TestEnv();

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(TestEnv);

  std::vector<ITestCase*> tests;
  std::atomic_int next_test_to_run;
  TestResultStat& stat;
  // FixedCountFinishCallbackImpl<TestCaseResult> finished;
  Ort::Env& env;
  const Ort::SessionOptions& sf;
  PThreadPool tp_;
};
