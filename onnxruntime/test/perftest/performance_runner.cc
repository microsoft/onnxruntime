// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "performance_runner.h"
#include <iostream>

#include "TestCase.h"
#include "TFModelInfo.h"
#include "utils.h"
#include "ort_test_session.h"
#ifdef HAVE_TENSORFLOW
#include "tf_test_session.h"
#endif
using onnxruntime::Status;

namespace onnxruntime {
namespace perftest {
Status PerformanceRunner::Run() {
  if (!Initialize()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "failed to initialize.");
  }

  // warm up
  RunOneIteration(true /*isWarmup*/);

  // TODO: start profiling
  // if (!performance_test_config_.run_config.profile_file.empty())

  std::unique_ptr<utils::ICPUUsage> p_ICPUUsage = utils::CreateICPUUsage();
  switch (performance_test_config_.run_config.test_mode) {
    case TestMode::kFixDurationMode:
      ORT_RETURN_IF_ERROR(RunFixDuration());
      break;
    case TestMode::KFixRepeatedTimesMode:
      ORT_RETURN_IF_ERROR(RunRepeatedTimes());
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "unknown test mode.");
  }
  performance_result_.average_CPU_usage = p_ICPUUsage->GetUsage();
  performance_result_.peak_workingset_size = utils::GetPeakWorkingSetSize();

  // TODO: end profiling
  // if (!performance_test_config_.run_config.profile_file.empty()) session_object->EndProfiling();

  std::cout << "Total time cost:" << performance_result_.total_time_cost << std::endl
            << "Total iterations:" << performance_result_.time_costs.size() << std::endl
            << "Average time cost:" << performance_result_.total_time_cost / performance_result_.time_costs.size() * 1000 << " ms" << std::endl;
  return Status::OK();
}

Status PerformanceRunner::RunOneIteration(bool isWarmup) {
  std::chrono::duration<double> duration_seconds = session_->Run(inputs_.Data());
  if (!isWarmup) {
    performance_result_.time_costs.emplace_back(duration_seconds.count());
    performance_result_.total_time_cost += duration_seconds.count();
    if (performance_test_config_.run_config.f_verbose) {
      std::cout << "iteration:" << performance_result_.time_costs.size() << ","
                << "time_cost:" << performance_result_.time_costs.back() << std::endl;
    }
  }
  return Status::OK();
}

static TestModelInfo* CreateModelInfo(const PerformanceTestConfig& performance_test_config_) {
  if (CompareCString(performance_test_config_.backend.c_str(), ORT_TSTR("ort")) == 0) {
    return TestModelInfo::LoadOnnxModel(performance_test_config_.model_info.model_file_path.c_str());
  }
  if (CompareCString(performance_test_config_.backend.c_str(), ORT_TSTR("tf")) == 0) {
    return TFModelInfo::Create(performance_test_config_.model_info.model_file_path.c_str());
  }
  ORT_NOT_IMPLEMENTED(ToMBString(performance_test_config_.backend), " is not supported");
}

static TestSession* CreateSession(OrtEnv* env, const PerformanceTestConfig& performance_test_config_,
                                  TestModelInfo* test_model_info) {
  if (CompareCString(performance_test_config_.backend.c_str(), ORT_TSTR("ort")) == 0) {
    return new OnnxRuntimeTestSession(env, performance_test_config_, test_model_info);
  }
#ifdef HAVE_TENSORFLOW
  if (CompareCString(performance_test_config_.backend.c_str(), ORT_TSTR("tf")) == 0) {
    return new TensorflowTestSession(performance_test_config_, test_model_info);
  }
#endif
  ORT_NOT_IMPLEMENTED(ToMBString(performance_test_config_.backend), " is not supported");
}
PerformanceRunner::PerformanceRunner(OrtEnv* env, const PerformanceTestConfig& test_config)
    : performance_test_config_(test_config),
      test_model_info_(CreateModelInfo(test_config)),
      session_(CreateSession(env, test_config, test_model_info_)),
      inputs_(test_model_info_->GetInputCount()) {}

PerformanceRunner::~PerformanceRunner() = default;

bool PerformanceRunner::Initialize() {
  std::basic_string<PATH_CHAR_TYPE> test_case_dir;
  auto st = GetDirNameFromFilePath(performance_test_config_.model_info.model_file_path, test_case_dir);
  if (!st.IsOK()) {
    printf("input path is not a valid model\n");
    return false;
  }
  std::basic_string<PATH_CHAR_TYPE> model_name = GetLastComponent(test_case_dir);
  // TODO: remove the input and model name's dependency on directory tree
  if (CompareCString(model_name.c_str(), ORT_TSTR("test_")) == 0) {
    model_name = model_name.substr(5);
  }
  std::string narrow_model_name = ToMBString(model_name);
  performance_result_.model_name = narrow_model_name;

  test_case_.reset(CreateOnnxTestCase(narrow_model_name, test_model_info_, 0.0, 0.0));
  test_model_info_ = nullptr;

  // TODO: Place input tensor on cpu memory if mkldnn provider type to avoid CopyTensor logic in CopyInputAcrossDevices
  if (test_case_->GetDataCount() <= 0) {
    std::cout << "there is no test data for model " << test_case_->GetTestCaseName() << std::endl;
    return false;
  }
  std::unordered_map<std::string, OrtValue*> feeds;
  test_case_->LoadTestData(0 /* id */, b_, feeds, true);
  // Discard the names in feeds
  size_t input_index = 0;
  for (auto& kvp : feeds) {
    inputs_.Set(input_index, kvp.second);
    ++input_index;
  }
  test_case_.reset(nullptr);
  return true;
}

}  // namespace perftest

}  // namespace onnxruntime
