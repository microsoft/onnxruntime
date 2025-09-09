// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <core/session/onnxruntime_c_api.h>
#include <random>
#include "command_args_parser.h"
#include "performance_runner.h"
#include "utils.h"
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
    fprintf(stderr, "%s", "See 'onnxruntime_perf_test --help'.");
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
        std::cerr << "Error creating environment: " << e.what() << std::endl;
        failed = true;
      });
    }

    if (failed)
      return -1;
  }

  if (!test_config.plugin_ep_names_and_libs.empty()) {
    perftest::utils::RegisterExecutionProviderLibrary(env, test_config);
  }

  // Unregister all registered plugin EP libraries before program exits.
  // This is necessary because unregistering the plugin EP also unregisters any associated shared allocators.
  // If we don't do this and program returns, the factories stored inside the environment will be destroyed when the environment goes out of scope.
  // Later, when the shared allocator's deleter runs, it may cause a segmentation fault because it attempts to use the already-destroyed factory to call ReleaseAllocator.
  // See "ep_device.ep_factory->ReleaseAllocator" in Environment::CreateSharedAllocatorImpl.
  auto unregister_plugin_eps_at_scope_exit = gsl::finally([&]() {
    if (!test_config.registered_plugin_eps.empty()) {
      perftest::utils::UnregisterExecutionProviderLibrary(env, test_config);  // this won't throw
    }
  });

  if (test_config.list_available_ep_devices) {
    perftest::utils::ListEpDevices(env);
    if (test_config.registered_plugin_eps.empty()) {
      fprintf(stdout, "No plugin execution provider libraries are registered. Please specify them using \"--plugin_ep_libs\"; otherwise, only CPU may be available.\n");
    }
    return 0;
  }

  std::random_device rd;
  perftest::PerformanceRunner perf_runner(env, test_config, rd);

  // Exit if user enabled -n option so that user can measure session creation time
  if (test_config.run_config.exit_after_session_creation) {
    perf_runner.LogSessionCreationTime();
    return 0;
  }

  auto status = perf_runner.Run();
  if (!status.IsOK()) {
    printf("Run failed:%s\n", status.ErrorMessage().c_str());
    return -1;
  }

  perf_runner.SerializeResult();

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
      std::cerr << ex.what() << std::endl;
      retval = -1;
    });
  }

  ::google::protobuf::ShutdownProtobufLibrary();

  return retval;
}
