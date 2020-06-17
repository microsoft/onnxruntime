// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include <core/session/onnxruntime_c_api.h>
#include <random>
#include "command_args_parser.h"
#include "performance_result.h"
#include "ort_test_session.h"
#include "TFModelInfo.h"
#include "test_configuration.h"
#include "TestCase.h"
#include "utils.h"

using namespace onnxruntime;
using namespace perftest;

const OrtApi* c_api = nullptr;


static std::unique_ptr<TestModelInfo> CreateModelInfo(const PerformanceTestConfig& test_config) {
  if (CompareCString(test_config.backend.c_str(), ORT_TSTR("ort")) == 0) {
    return TestModelInfo::LoadOnnxModel(test_config.model_info.model_file_path.c_str());
  }
  if (CompareCString(test_config.backend.c_str(), ORT_TSTR("tf")) == 0) {
    return TFModelInfo::Create(test_config.model_info.model_file_path.c_str());
  }
  ORT_NOT_IMPLEMENTED(ToMBString(test_config.backend), " is not supported");
}


#ifdef _WIN32
int real_main(int argc, wchar_t* argv[]) {
#else
int real_main(int argc, char* argv[]) {
#endif
  c_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  perftest::PerformanceTestConfig test_config;
  if (!perftest::CommandLineParser::ParseArguments(test_config, argc, argv)) {
    perftest::CommandLineParser::ShowUsage();
    return -1;
  }
  OrtEnv* env = nullptr;
  OrtLoggingLevel logging_level = test_config.run_config.f_verbose
                                        ? ORT_LOGGING_LEVEL_WARNING:ORT_LOGGING_LEVEL_ERROR;
  ThrowOnError(c_api->CreateEnv(logging_level, "OrtPerfTest", &env));

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

  OrtSession* sess{nullptr};
  auto session_create_start_ = std::chrono::high_resolution_clock::now();
  sess = CreateOrtSession(env,test_config);
  auto session_create_end_ = std::chrono::high_resolution_clock::now();

  std::basic_string<ORTCHAR_T> test_case_dir;
  auto st = GetDirNameFromFilePath(test_config.model_info.model_file_path, test_case_dir);
  if (!st.IsOK()) {
    printf("input path is not a valid model\n");
    return -1;
  }
  std::basic_string<ORTCHAR_T> model_name = GetLastComponent(test_case_dir);
  // TODO: remove the input and model name's dependency on directory tree
  if (CompareCString(model_name.c_str(), ORT_TSTR("test_")) == 0) {
    model_name = model_name.substr(5);
  }
  std::string narrow_model_name = ToMBString(model_name);
  //performance_result_.model_name = narrow_model_name;

  auto test_case_ = CreateOnnxTestCase(narrow_model_name, CreateModelInfo(test_config), 0.0, 0.0);


  // TODO: Place input tensor on cpu memory if dnnl provider type to avoid CopyTensor logic in CopyInputAcrossDevices
  size_t test_data_count = test_case_->GetDataCount();
  if (test_data_count == 0) {
    std::cout << "there is no test data for model " << test_case_->GetTestCaseName() << std::endl;
    return -1;
  }

  SampleLoader s(sess,test_case_.get());
  OnnxRuntimeTestSession* session_ = new OnnxRuntimeTestSession(sess, &s, rd, test_config.run_config.concurrent_session_runs);

  mlperf::TestSettings test_settings;
  test_settings.scenario = test_config.run_config.concurrent_session_runs == 1 ? mlperf::TestScenario::SingleStream:
      mlperf::TestScenario::MultiStream;
  test_settings.min_duration_ms = static_cast<std::uint64_t>(test_config.run_config.duration_in_seconds) * 1000;
  //test_settings.min_query_count = test_config.run_config.repeated_times;
  test_settings.min_query_count = 1;
  mlperf::LogSettings log_settings;
  auto perf_run_start = std::chrono::high_resolution_clock::now();
  mlperf::StartTest(session_,&s,test_settings,log_settings);
  auto perf_run_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> session_create_duration = session_create_end_ - session_create_start_;
  std::chrono::duration<double> inference_duration = perf_run_end - perf_run_start;
  auto performance_result_ = session_->GetPerformanceResult();
  std::cout << "Session creation time cost:" << session_create_duration.count() << " s" << std::endl
            << "Total inference time cost:" << performance_result_.total_time_cost/1000000000.0 << " s" << std::endl  // sum of time taken by each request
            << "Total inference requests:" << performance_result_.time_costs.size() << std::endl
            << "Average inference time cost:" << performance_result_.total_time_cost / performance_result_.time_costs.size() / 1000000.0 << " ms" << std::endl
            // Time between start and end of run. Less than Total time cost when running requests in parallel.
            << "Total inference run time:" << inference_duration.count() << " s" << std::endl;
  performance_result_.DumpToFile(test_config.model_info.result_file_path,
                                 test_config.run_config.f_dump_statistics);
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

  return retval;
}
