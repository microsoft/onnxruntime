// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <core/session/onnxruntime_c_api.h>
#include <random>
#include "command_args_parser.h"
#include "utils.h"
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
  if (!perftest::CommandLineParser::ParseArgumentsV2(test_config, argc, argv)) {
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

  if (!test_config.plugin_ep_names_and_libs.empty()) {
    perftest::utils::RegisterExecutionProviderLibrary(env, test_config);
  }

  if (test_config.list_available_devices) {
    perftest::utils::list_devices(env);
    if (test_config.registered_plugin_eps.empty()) {
      fprintf(stdout, "No plugin execution provider libraries are registered. Please specify them using \"--plugin_ep_libs\"; otherwise, only CPU may be available.\n");
    } else {
      perftest::utils::UnregisterExecutionProviderLibrary(env, test_config);
    }
    return 0;
  }

  auto status = Status::OK();

  try {
    std::random_device rd;
    perftest::PerformanceRunner perf_runner(env, test_config, rd);

    // Exit if user enabled -n option so that user can measure session creation time
    if (test_config.run_config.exit_after_session_creation) {
      perf_runner.LogSessionCreationTime();
      return 0;
    }

    throw std::runtime_error("Something went wrong");

    status = perf_runner.Run();

    if (!status.IsOK()) {
      printf("Run failed:%s\n", status.ErrorMessage().c_str());
    } else {
      perf_runner.SerializeResult();
    }
  } catch (const std::exception&) {
    if (!test_config.registered_plugin_eps.empty()) {
      perftest::utils::UnregisterExecutionProviderLibrary(env, test_config);
      return -1;
    }
  }
  // The try/catch block above ensures the following:
  // 1) Plugin EP libraries are unregistered if an exception occurs.
  // 2) Objects are released in the correct order when running a plugin EP.
  //
  // Proper destruction order is critical to avoid use-after-free issues. The expected order of deleters is:
  // session -> session allocator (accessed via EP factory) -> plugin EP -> env ->
  // shared allocator (accessed via EP factory) -> plugin EP factory (owned by env)
  //
  // Without this order, the environment (`env`) might be destroyed first, and
  // any subsequent access to the session allocator's deleter (which depends on the EP factory)
  // can result in a segmentation fault because the factory has already been destroyed.

  // Unregister all registered plugin EP libraries before program exits.
  //
  // This is necessary because unregistering the plugin EP also unregisters any associated shared allocators.
  // If we don't do this first and program returns, the factories stored inside the environment will be destroyed when the environment goes out of scope.
  // Later, when the shared allocator's deleter runs, it may cause a segmentation fault because it attempts to use the already-destroyed factory to call ReleaseAllocator.
  //
  // See "ep_device.ep_factory->ReleaseAllocator" in Environment::CreateSharedAllocatorImpl.
  if (!test_config.registered_plugin_eps.empty()) {
    perftest::utils::UnregisterExecutionProviderLibrary(env, test_config);
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
