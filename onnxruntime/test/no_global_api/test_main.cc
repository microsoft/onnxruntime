// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef USE_ONNXRUNTIME_DLL
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include <google/protobuf/message_lite.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif

#include "gtest/gtest.h"

// The global singleton for Ort's GetApi() will already exists if
// gtest compiles multiple translation units into a single test EXE.
// This pollutes the test environment needed for the no-global-api tests
// because other TUs have already initialized `Global<void>::api_`.
// Therefore...this main() and the other TUs are a separate test EXE and
// their cmake defines ORT_API_MANUAL_INIT.
int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int status = RUN_ALL_TESTS();

#ifndef USE_ONNXRUNTIME_DLL
  //make memory leak checker happy
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
  return status;
}
