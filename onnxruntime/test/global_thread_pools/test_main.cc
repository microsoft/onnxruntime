// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef USE_ONNXRUNTIME_DLL
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wignored-qualifiers"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#else
#pragma warning(push)
#pragma warning(disable : 4018) /*'expression' : signed/unsigned mismatch */
#pragma warning(disable : 4065) /*switch statement contains 'default' but no 'case' labels*/
#pragma warning(disable : 4100)
#pragma warning(disable : 4146) /*unary minus operator applied to unsigned type, result still unsigned*/
#pragma warning(disable : 4127)
#pragma warning(disable : 4244) /*'conversion' conversion from 'type1' to 'type2', possible loss of data*/
#pragma warning(disable : 4251) /*'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'*/
#pragma warning(disable : 4267) /*'var' : conversion from 'size_t' to 'type', possible loss of data*/
#pragma warning(disable : 4305) /*'identifier' : truncation from 'type1' to 'type2'*/
#pragma warning(disable : 4307) /*'operator' : integral constant overflow*/
#pragma warning(disable : 4309) /*'conversion' : truncation of constant value*/
#pragma warning(disable : 4334) /*'operator' : result of 32-bit shift implicitly converted to 64 bits (was 64-bit shift intended?)*/
#pragma warning(disable : 4355) /*'this' : used in base member initializer list*/
#pragma warning(disable : 4506) /*no definition for inline function 'function'*/
#pragma warning(disable : 4800) /*'type' : forcing value to bool 'true' or 'false' (performance warning)*/
#pragma warning(disable : 4996) /*The compiler encountered a deprecated declaration.*/
#pragma warning(disable : 6011) /*Dereferencing NULL pointer*/
#pragma warning(disable : 6387) /*'value' could be '0'*/
#pragma warning(disable : 26495) /*Variable is uninitialized.*/
#endif
#include <google/protobuf/message_lite.h>
#ifdef __GNUC__
#pragma GCC diagnostic pop
#else
#pragma warning(pop)
#endif
#endif

#include "core/session/onnxruntime_cxx_api.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"

std::unique_ptr<Ort::Env> ort_env;

int main(int argc, char** argv) {
  int status = 0;
  try {
    ::testing::InitGoogleTest(&argc, argv);
    const OrtApi* g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtThreadingOptions* tp_options;
    OrtStatus* st = g_ort->CreateThreadingOptions(&tp_options);
    if(st != nullptr) return -1;
    ort_env.reset(new Ort::Env(tp_options, ORT_LOGGING_LEVEL_VERBOSE, "Default"));  // this is the only change from test/providers/test_main.cc
    g_ort->ReleaseThreadingOptions(tp_options);
    status = RUN_ALL_TESTS();
  } catch (const std::exception& ex) {
    std::cerr << ex.what();
    status = -1;
  }
  //TODO: Fix the C API issue
  ort_env.reset();  //If we don't do this, it will crash

#ifndef USE_ONNXRUNTIME_DLL
  //make memory leak checker happy
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
  return status;
}
