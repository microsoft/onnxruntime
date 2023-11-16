// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>

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

#include "core/platform/env_var_utils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/util/thread_utils.h"
#include "test/test_environment.h"

std::unique_ptr<Ort::Env> ort_env;

void ortenv_setup() {
  OrtThreadingOptions tpo;
#if defined(__wasm__)
  OrtLoggingLevel logging_level = ORT_LOGGING_LEVEL_WARNING;
  OrtThreadingOptions tp_options;
  OrtEnv *env;
  // use the global thread pool
  (void)Ort::GetApi().CreateEnvWithGlobalThreadPools(logging_level, "Default", &tp_options, &env);
  ort_env.reset(new Ort::Env(env));
#else
  ort_env.reset(new Ort::Env(&tpo, ORT_LOGGING_LEVEL_WARNING, "Default"));
#endif
}

#ifdef USE_TENSORRT
// TensorRT will load/unload libraries as builder objects are created and torn down. This will happen for
// every single unit test, which leads to excessive test execution time due to that overhead.
// Nvidia suggests to keep a placeholder builder object around to avoid this.
#include "NvInfer.h"
class DummyLogger : public nvinfer1::ILogger {
 public:
  DummyLogger(Severity verbosity) {}
  void log(Severity severity, const char* msg) noexcept override {}
};
DummyLogger trt_logger(nvinfer1::ILogger::Severity::kWARNING);
auto const placeholder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(trt_logger));
#endif

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

    // allow verbose logging to be enabled by setting this environment variable to a numeric log level
    constexpr auto kLogLevelEnvironmentVariableName = "ORT_UNIT_TEST_MAIN_LOG_LEVEL";
    if (auto log_level = onnxruntime::ParseEnvironmentVariable<int>(kLogLevelEnvironmentVariableName);
        log_level.has_value()) {
      *log_level = std::clamp(*log_level,
                              static_cast<int>(ORT_LOGGING_LEVEL_VERBOSE),
                              static_cast<int>(ORT_LOGGING_LEVEL_FATAL));
      std::cout << "Setting log level to " << *log_level << "\n";
      ort_env->UpdateEnvWithCustomLogLevel(static_cast<OrtLoggingLevel>(*log_level));
    }

    status = RUN_ALL_TESTS();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what();
      status = -1;
    });
  }

  // TODO: Fix the C API issue
  ort_env.reset();  // If we don't do this, it will crash

#ifndef USE_ONNXRUNTIME_DLL
  // make memory leak checker happy
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
  return status;
}
