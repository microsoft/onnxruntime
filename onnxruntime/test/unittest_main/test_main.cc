// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef USE_ONNXRUNTIME_DLL
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#else
#endif
#include <google/protobuf/message_lite.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
#endif

#include "core/session/onnxruntime_cxx_api.h"
#include "core/util/thread_utils.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"

std::unique_ptr<Ort::Env> ort_env;
void ortenv_setup(){
  OrtThreadingOptions tpo;
  ort_env.reset(new Ort::Env(&tpo, ORT_LOGGING_LEVEL_WARNING, "Default"));
}

#define TEST_MAIN main

#if defined(__APPLE__)
  #include <TargetConditionals.h>
  #if TARGET_OS_SIMULATOR || TARGET_OS_IOS
    #undef TEST_MAIN
    #define TEST_MAIN main_no_link_  // there is a UI test app for iOS.
  #endif
#endif

int TEST_MAIN(int argc, char** argv) {
  int status = 0;

  ORT_TRY {
    ::testing::InitGoogleTest(&argc, argv);

    ortenv_setup();
    status = RUN_ALL_TESTS();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what();
      status = -1;
    });
  }

  //TODO: Fix the C API issue
  ort_env.reset();  //If we don't do this, it will crash

#ifndef USE_ONNXRUNTIME_DLL
  //make memory leak checker happy
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
  return status;
}
