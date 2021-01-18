// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/test_environment.h"
#include "test_config.h"

using namespace onnxruntime::test;

GTEST_API_ int main(int argc, char** argv) {
  int status = 0;

  ENCLAVE_PATH = argv[1];
  std::cout << "Enclave path: " << ENCLAVE_PATH << std::endl;

  try {
    ::testing::InitGoogleTest(&argc, argv);
    status = RUN_ALL_TESTS();
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    status = -1;
  }

  return status;
}
