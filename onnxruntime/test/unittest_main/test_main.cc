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
#pragma warning(disable : 4244)  /*'conversion' conversion from 'type1' to 'type2', possible loss of data*/
#pragma warning(disable : 4251)  /*'identifier' : class 'type' needs to have dll-interface to be used by clients of class 'type2'*/
#pragma warning(disable : 4267)  /*'var' : conversion from 'size_t' to 'type', possible loss of data*/
#pragma warning(disable : 4305)  /*'identifier' : truncation from 'type1' to 'type2'*/
#pragma warning(disable : 4307)  /*'operator' : integral constant overflow*/
#pragma warning(disable : 4309)  /*'conversion' : truncation of constant value*/
#pragma warning(disable : 4334)  /*'operator' : result of 32-bit shift implicitly converted to 64 bits (was 64-bit shift intended?)*/
#pragma warning(disable : 4355)  /*'this' : used in base member initializer list*/
#pragma warning(disable : 4506)  /*no definition for inline function 'function'*/
#pragma warning(disable : 4800)  /*'type' : forcing value to bool 'true' or 'false' (performance warning)*/
#pragma warning(disable : 4996)  /*The compiler encountered a deprecated declaration.*/
#pragma warning(disable : 6011)  /*Dereferencing NULL pointer*/
#pragma warning(disable : 6387)  /*'value' could be '0'*/
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
#include "core/util/thread_utils.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"

// We keep a list of functions to call on exit before we destroy the OrtEnv.
// This is needed to do any cleanup that must be done before the OrtEnv gets destroyed and all shared providers get unloaded.
static std::vector<std::function<void()>> exit_functions;
void CallOnTestExit(std::function<void()>&& function) { exit_functions.emplace_back(std::move(function)); }

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

  // Run exit functions that must be done before we delete the env (like allocators that reference shared providers)
  for (auto& exit_function : exit_functions)
    exit_function();

  //TODO: Fix the C API issue
  ort_env.reset();  //If we don't do this, it will crash

#ifndef USE_ONNXRUNTIME_DLL
  //make memory leak checker happy
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
  return status;
}
