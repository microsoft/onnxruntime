// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <iostream>
#include <memory>

#include "core/session/onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include "test/util/include/telemetry_test_environment.h"

std::unique_ptr<Ort::Env> ort_env;

extern "C" void ortenv_setup() {
  onnxruntime::test::SuppressTelemetryForTests();
  ort_env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "BazelSharedApiTests");
}

extern "C" void ortenv_teardown() {
  ort_env.reset();
}

int OrtTestMain(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  try {
    ortenv_setup();
    const int result = RUN_ALL_TESTS();
    ortenv_teardown();
    return result;
  } catch (const std::exception& error) {
    std::cerr << "Shared API test environment failed: " << error.what()
              << ". Check the runtime shared-library runfiles and selected fixture requirements.\n";
    return 1;
  }
}
