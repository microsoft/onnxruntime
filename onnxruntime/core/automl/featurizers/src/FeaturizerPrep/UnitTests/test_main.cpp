// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

GTEST_API_ int main(int argc, char** argv) {
  int status = 0;

  testing::InitGoogleTest(&argc, argv);
  try {
    status = RUN_ALL_TESTS();
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    status = -1;
  }

  return status;
}
