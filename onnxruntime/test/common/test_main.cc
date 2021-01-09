// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/test_environment.h"

GTEST_API_ int main(int argc, char** argv) {
  int status = 0;

  ORT_TRY {
    const bool create_default_logger = false;
    onnxruntime::test::TestEnvironment environment{argc, argv, create_default_logger};

    status = RUN_ALL_TESTS();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what();
      status = -1;
    });
  }

  return status;
}
