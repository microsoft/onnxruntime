// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <core/session/onnxruntime_c_api.h>
#include <random>
#include "command_args_parser.h"
#include "performance_runner.h"
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
  try {
    OrtLoggingLevel logging_level = test_config.run_config.f_verbose
                                        ? ORT_LOGGING_LEVEL_VERBOSE
                                        : ORT_LOGGING_LEVEL_WARNING;
    env = Ort::Env(logging_level, "Default");
  } catch (const Ort::Exception& e) {
    fprintf(stderr, "Error creating environment: %s \n", e.what());
    return -1;
  }
  if(test_config.machine_config.provider_type_name == onnxruntime::kOpenVINOExecutionProvider){
    if(test_config.run_config.concurrent_session_runs != 1){
      fprintf(stderr, "OpenVINO doesn't support more than 1 session running simultaneously default value of 1 will be set \n");
      test_config.run_config.concurrent_session_runs = 1;
    }
    if(test_config.run_config.execution_mode == ExecutionMode::ORT_PARALLEL){
      fprintf(stderr, "OpenVINO doesn't support parallel executor using sequential executor\n");
      test_config.run_config.execution_mode = ExecutionMode::ORT_SEQUENTIAL;
    }
  }
  std::random_device rd;
  perftest::PerformanceRunner perf_runner(env, test_config, rd);
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
  try {
    retval = real_main(argc, argv);
  } catch (std::exception& ex) {
    fprintf(stderr, "%s\n", ex.what());
    retval = -1;
  }

  ::google::protobuf::ShutdownProtobufLibrary();

  return retval;
}
