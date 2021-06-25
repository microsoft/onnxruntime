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
#include "gtest/gtest.h"
#include "test/test_environment.h"
#include <thread>

std::unique_ptr<Ort::Env> ort_env;

#define ORT_RETURN_IF_NON_NULL_STATUS(arg) \
  if (arg) {                               \
    return -1;                             \
  }

int main(int argc, char** argv) {
  int status = 0;
  ORT_TRY {
    ::testing::InitGoogleTest(&argc, argv);
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtThreadingOptions* tp_options;
    std::unique_ptr<OrtStatus, decltype(OrtApi::ReleaseStatus)> st_ptr(nullptr, g_ort->ReleaseStatus);

    st_ptr.reset(g_ort->CreateThreadingOptions(&tp_options));
    ORT_RETURN_IF_NON_NULL_STATUS(st_ptr);

    st_ptr.reset(g_ort->SetGlobalSpinControl(tp_options, 0));
    ORT_RETURN_IF_NON_NULL_STATUS(st_ptr);

    st_ptr.reset(g_ort->SetGlobalIntraOpNumThreads(tp_options, std::thread::hardware_concurrency()));
    ORT_RETURN_IF_NON_NULL_STATUS(st_ptr);

    st_ptr.reset(g_ort->SetGlobalInterOpNumThreads(tp_options, std::thread::hardware_concurrency()));
    ORT_RETURN_IF_NON_NULL_STATUS(st_ptr);

    st_ptr.reset(g_ort->SetGlobalDenormalAsZero(tp_options));
    ORT_RETURN_IF_NON_NULL_STATUS(st_ptr);

    ort_env.reset(new Ort::Env(tp_options, ORT_LOGGING_LEVEL_VERBOSE, "Default"));  // this is the only change from test/providers/test_main.cc
    g_ort->ReleaseThreadingOptions(tp_options);
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
