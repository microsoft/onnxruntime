// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testenv.h"
#include "FixedCountFinishCallback.h"
#include <core/session/onnxruntime_cxx_api.h>

using onnxruntime::Status;
TestEnv::TestEnv(const std::vector<ITestCase*>& tests1, TestResultStat& stat1, Ort::Env& env1,
                 Ort::SessionOptions& sf1)
    : tests(tests1),
      next_test_to_run(0),
      stat(stat1),
      finished(new FixedCountFinishCallback(static_cast<int>(tests1.size()))),
      env(env1),
      sf(sf1) {
}

TestEnv::~TestEnv() {
  // need dtor in .cc so 'finished' can be cleaned up as TestCaseResult only has a forward declaration in the header.
}
