// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#ifdef _WIN32
#include <iostream>
#include <locale>
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

#include "core/common/common.h"
#include "core/platform/env_var_utils.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/util/thread_utils.h"

#if defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP)
#include "test/util/include/test_dynamic_plugin_ep.h"
#endif  // defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP)

std::unique_ptr<Ort::Env> ort_env;

// define environment variable name constants here
namespace env_var_names {
// Set ORT log level to the specified numeric log level.
constexpr const char* kLogLevel = "ORT_UNIT_TEST_MAIN_LOG_LEVEL";

#if defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP)
// Specify dynamic plugin EP configuration JSON.
// Refer to `onnxruntime::test::dynamic_plugin_ep_infra::ParseInitializationConfig()` for more information.
constexpr const char* kDynamicPluginEpConfigJson = "ORT_UNIT_TEST_MAIN_DYNAMIC_PLUGIN_EP_CONFIG_JSON";
#endif  // defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP)
}  // namespace env_var_names

// ortenv_setup() and ortenv_teardown() are used by onnxruntime/test/xctest/xcgtest.mm so can't be file local
extern "C" void ortenv_setup() {
  ORT_TRY {
#ifdef _WIN32
    // Set the locale to UTF-8 to ensure proper handling of wide characters on Windows
    std::wclog.imbue(std::locale(".UTF-8", std::locale::ctype));
#endif

    OrtThreadingOptions tpo;

    OrtLoggingLevel log_level = ORT_LOGGING_LEVEL_WARNING;
    if (auto log_level_override = onnxruntime::ParseEnvironmentVariable<int>(env_var_names::kLogLevel);
        log_level_override.has_value()) {
      *log_level_override = std::clamp(*log_level_override,
                                       static_cast<int>(ORT_LOGGING_LEVEL_VERBOSE),
                                       static_cast<int>(ORT_LOGGING_LEVEL_FATAL));
      std::cout << "Setting log level to " << *log_level_override << "\n";
      log_level = static_cast<OrtLoggingLevel>(*log_level_override);
    }

    ort_env.reset(new Ort::Env(&tpo, log_level, "Default"));

#if defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP)

    {
      namespace dynamic_plugin_ep_infra = onnxruntime::test::dynamic_plugin_ep_infra;
      if (auto dynamic_plugin_ep_config_json = onnxruntime::ParseEnvironmentVariable<std::string>(
              env_var_names::kDynamicPluginEpConfigJson);
          dynamic_plugin_ep_config_json.has_value()) {
        std::cout << "Initializing dynamic plugin EP infrastructure with configuration:\n"
                  << *dynamic_plugin_ep_config_json << "\n";
        dynamic_plugin_ep_infra::InitializationConfig config{};
        ORT_THROW_IF_ERROR(dynamic_plugin_ep_infra::ParseInitializationConfig(*dynamic_plugin_ep_config_json, config));
        ORT_THROW_IF_ERROR(dynamic_plugin_ep_infra::Initialize(*ort_env, config));
      }
    }

#endif  // defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP)
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what();
      std::exit(1);
    });
  }
}

extern "C" void ortenv_teardown() {
#if defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP)
  {
    namespace dynamic_plugin_ep_infra = onnxruntime::test::dynamic_plugin_ep_infra;
    dynamic_plugin_ep_infra::Shutdown();
  }
#endif  // defined(ORT_UNIT_TEST_ENABLE_DYNAMIC_PLUGIN_EP)

  ort_env.reset();
}

#ifdef USE_TENSORRT

#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)  // Ignore warning C4100: unreferenced format parameter.
#pragma warning(disable : 4996)  // Ignore warning C4996: 'nvinfer1::IPluginV2' was declared deprecated
#endif

// TensorRT will load/unload libraries as builder objects are created and torn down. This will happen for
// every single unit test, which leads to excessive test execution time due to that overhead.
// Nvidia suggests to keep a placeholder builder object around to avoid this.
#include "NvInfer.h"

#if defined(_MSC_VER)
#pragma warning(pop)
#endif

class DummyLogger : public nvinfer1::ILogger {
 public:
  DummyLogger(Severity /*verbosity*/) {}
  void log(Severity /*severity*/, const char* /*msg*/) noexcept override {}
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
    ortenv_setup();
    ::testing::InitGoogleTest(&argc, argv);

    status = RUN_ALL_TESTS();
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      std::cerr << ex.what();
      status = -1;
    });
  }

  // TODO: Fix the C API issue
  ortenv_teardown();  // If we don't do this, it will crash

#ifndef USE_ONNXRUNTIME_DLL
  // make memory leak checker happy
  ::google::protobuf::ShutdownProtobufLibrary();
#endif
  return status;
}
