// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <core/session/onnxruntime_c_api.h>
#include <random>
#include "command_args_parser.h"
#include "performance_runner.h"
#include "strings_helper.h"
#include <google/protobuf/stubs/common.h>

using namespace onnxruntime;
const OrtApi* g_ort = NULL;

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[]) {
#else
int real_main(int argc, char* argv[]) {
#endif
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  perftest::PerformanceTestConfig test_config;
  if (!perftest::CommandLineParser::ParseArguments(test_config, argc, argv)) {
    perftest::CommandLineParser::ShowUsage();
    return -1;
  }
  Ort::Env env{nullptr};
  {
    bool failed = false;
    ORT_TRY {
      OrtLoggingLevel logging_level = test_config.run_config.f_verbose
                                          ? ORT_LOGGING_LEVEL_VERBOSE
                                          : ORT_LOGGING_LEVEL_WARNING;
      env = Ort::Env(logging_level, "Default");
    }
    ORT_CATCH(const Ort::Exception& e) {
      ORT_HANDLE_EXCEPTION([&]() {
        fprintf(stderr, "Error creating environment: %s \n", e.what());
        failed = true;
      });
    }

    if (failed)
      return -1;
  }

  auto status = Status::OK();

  {
    std::random_device rd;
    perftest::PerformanceRunner perf_runner(env, test_config, rd);

    // Exit if user enabled -n option so that user can measure session creation time
    if (test_config.run_config.exit_after_session_creation) {
      perf_runner.LogSessionCreationTime();
      return 0;
    }

    status = perf_runner.Run();

    if (!status.IsOK()) {
      printf("Run failed:%s\n", status.ErrorMessage().c_str());
    } else {
      perf_runner.SerializeResult();
    }
  }

  // Unregister all registered plugin EP libraries before program exits.
  //
  // This is necessary because unregistering the plugin EP also unregisters any associated shared allocators.
  // If we don't do this first and program returns, the factories stored inside the environment will be destroyed when the environment goes out of scope.
  // Later, when the shared allocator's deleter runs, it may cause a segmentation fault because it attempts to use the already-destroyed factory to call ReleaseAllocator.
  //
  // See "ep_device.ep_factory->ReleaseAllocator" in Environment::CreateSharedAllocatorImpl.
  std::unordered_map<std::string, std::string> ep_names_to_libs;
#ifdef _MSC_VER
  std::string ep_names_and_libs_string = ToUTF8String(test_config.plugin_ep_names_and_libs);
#else
  std::string ep_names_and_libs_string = performance_test_config.plugin_ep_names_and_libs;
#endif
  onnxruntime::perftest::ParseSessionConfigs(ep_names_and_libs_string, ep_names_to_libs);
  for (auto& pair : ep_names_to_libs) {
    env.UnregisterExecutionProviderLibrary(pair.first.c_str());
  }

  if (!status.IsOK()) {
    return -1;
  }

  return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  int retval = -1;
  ORT_TRY {
    retval = real_main(argc, argv);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      fprintf(stderr, "%s\n", ex.what());
      retval = -1;
    });
  }

  ::google::protobuf::ShutdownProtobufLibrary();

  return retval;
}
