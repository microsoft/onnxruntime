// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#if defined(__wasm__)
#include <emscripten.h>
#endif

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
#include "core/common/logging/sinks/clog_sink.h"
#include "core/session/environment.h"

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#include "core/util/thread_utils.h"
#include "test/test_environment.h"

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

std::unique_ptr<Ort::Env> ort_env;
static onnxruntime::Status ortenv_setup() {
  emscripten_log(EM_LOG_INFO,"XXXXXXXXXXXXXXXXXXXX:ortenv_setup start\n");
  OrtThreadingOptions tpo;

  onnxruntime::Status status;

  std::unique_ptr<onnxruntime::logging::LoggingManager> lmgr;
  std::string name = "Default";
  lmgr = std::make_unique<onnxruntime::logging::LoggingManager>(std::make_unique<onnxruntime::logging::CLogSink>(),
                                                                onnxruntime::logging::Severity::kWARNING,
                                                                false,
                                                                onnxruntime::logging::LoggingManager::InstanceType::Default,
                                                                &name);
  std::unique_ptr<onnxruntime::Environment> env;
  ORT_RETURN_IF_ERROR(onnxruntime::Environment::Create(std::move(lmgr), env, &tpo, true));
  std::unique_ptr<OrtEnv> env2=std::make_unique<OrtEnv>(std::move(env));
  ort_env = std::make_unique<Ort::Env>(env2.release());
  emscripten_log(EM_LOG_INFO,"XXXXXXXXXXXXXXXXXXXX:ortenv_setup end\n");
  return status;
}

#if defined(__wasm__)
enum class EmStage {
  INIT,
  SETUP_ENV,
  RUN_TEST,
  FINI,
};

struct EmState {
  int ret = 0;
  EmStage stage = EmStage::INIT;
  int argc;
  char** argv;
};

void MainLoop(void* arg) {
  emscripten_log(EM_LOG_INFO, "Entering MainLoop ...\n");
  if (arg == nullptr) return;
  EmState& state = *(EmState*)arg;
  emscripten_log(EM_LOG_INFO, "stage %d ...\n", (int)state.stage);
  onnxruntime::Status status;
  switch (state.stage) {
    case EmStage::INIT: {
      status = ortenv_setup();
      if (!status.IsOK()) {
        state.ret = -1;
        state.stage = EmStage::FINI;
      } else {
        state.stage = EmStage::SETUP_ENV;
      }
    } break;
    case EmStage::SETUP_ENV:
      ::testing::InitGoogleTest(&state.argc, state.argv);
      state.stage = EmStage::RUN_TEST;
      break;
    case EmStage::RUN_TEST:
      state.ret = RUN_ALL_TESTS();
      state.stage = EmStage::FINI;
      break;
    default:
      emscripten_log(EM_LOG_INFO, "Release ORT Env\n");      
      ort_env.reset();
      emscripten_cancel_main_loop();
      break;
  }
  emscripten_log(EM_LOG_INFO, "Exiting MainLoop ...\n");
  return;
}

static EmState em_global_state;
int TEST_MAIN(int argc, char** argv) {
  std::cout << "start: argc=" << argc << std::endl;
  em_global_state.argc = argc;
  em_global_state.argv = argv;
  emscripten_set_main_loop_arg(MainLoop, &em_global_state, 0, 0);
  return em_global_state.ret;
}
#else
int TEST_MAIN(int argc, char** argv) {
  int status = 0;

  ORT_TRY {
    ::testing::InitGoogleTest(&argc, argv);
    auto st = ortenv_setup();
    if (!st.IsOK()) {
      return -1;
    }

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
#endif
