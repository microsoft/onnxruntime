// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <gtest/gtest.h>

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int ret = RUN_ALL_TESTS();
  //TODO: Linker on Mac OS X is kind of strange. The next line of code will trigger a crash
#ifndef __APPLE__
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
  return ret;
}