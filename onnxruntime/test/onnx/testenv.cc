// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testenv.h"
#include "FixedCountFinishCallback.h"
#include <core/session/onnxruntime_cxx_api.h>

using onnxruntime::SessionOptionsWrapper;

using onnxruntime::Status;
TestEnv::TestEnv(const std::vector<ITestCase*>& tests1, TestResultStat& stat1, SessionOptionsWrapper& sf1)
    : tests(tests1), next_test_to_run(0), stat(stat1), finished(new FixedCountFinishCallback(static_cast<int>(tests1.size()))), sf(sf1) {
}

TestEnv::~TestEnv() {
  delete finished;
}
