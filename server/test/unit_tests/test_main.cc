// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "environment.h"
#include "test_server_environment.h"

GTEST_API_ int main(int argc, char** argv) {
  int status = 0;
  try {
    testing::InitGoogleTest(&argc, argv);
    onnxruntime::server::test::TestServerEnvironment server_env{};
    status = RUN_ALL_TESTS();
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    status = -1;
  }

  return status;
}
