// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "performance_runner.h"
#include "TestCase.h"
#include "core/graph/graph_viewer.h"  //for onnxruntime::NodeArg
#include "core/session/inference_session.h"
#include "utils.h"
#include "testenv.h"
#include "providers.h"

using onnxruntime::Status;

namespace onnxruntime {
namespace perftest {
Status PerformanceRunner::Run() {
  if (!Initialize()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "failed to initialize.");
  }

  // warm up
  RunOneIteration(true /*isWarmup*/);
  InferenceSession* session_object = (InferenceSession*)session_object_;

  if (!performance_test_config_.run_config.profile_file.empty())
    session_object->StartProfiling(performance_test_config_.run_config.profile_file);

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

  if (!performance_test_config_.run_config.profile_file.empty()) session_object->EndProfiling();

  std::cout << "Total time cost:" << performance_result_.total_time_cost << std::endl
            << "Total iterations:" << performance_result_.time_costs.size() << std::endl
            << "Average time cost:" << performance_result_.total_time_cost / performance_result_.time_costs.size() * 1000 << " ms" << std::endl;
  return Status::OK();
}

Status PerformanceRunner::RunOneIteration(bool isWarmup) {
  auto start = std::chrono::high_resolution_clock::now();
  OrtRunOptions run_options;

  ORT_THROW_ON_ERROR(OrtRun(session_object_, nullptr, input_names_.data(), input_values_.data(), input_names_.size(),
                            output_names_raw_ptr.data(), output_names_raw_ptr.size(), output_values_.data()));
  auto end = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i != output_values_.size(); ++i) {
    OrtReleaseValue(output_values_[i]);
    output_values_[i] = nullptr;
  }
  if (!isWarmup) {
    std::chrono::duration<double> duration_seconds = end - start;
    performance_result_.time_costs.emplace_back(duration_seconds.count());
    performance_result_.total_time_cost += duration_seconds.count();
    if (performance_test_config_.run_config.f_verbose) {
      std::cout << "iteration:" << performance_result_.time_costs.size() << ","
                << "time_cost:" << performance_result_.time_costs.back() << std::endl;
    }
  }
  return Status::OK();
}

bool PerformanceRunner::Initialize() {
  bool has_valid_extension = HasExtensionOf(performance_test_config_.model_info.model_file_path, ORT_TSTR("onnx"));
  if (!has_valid_extension) {
    LOGF_DEFAULT(ERROR, "input path is not a valid model");
    return false;
  }
  std::basic_string<PATH_CHAR_TYPE> test_case_dir;
  auto st = GetDirNameFromFilePath(performance_test_config_.model_info.model_file_path, test_case_dir);
  if (!st.IsOK()) {
    LOGF_DEFAULT(ERROR, "input path is not a valid model");
    return false;
  }
  std::basic_string<PATH_CHAR_TYPE> model_name = GetLastComponent(test_case_dir);
  // TODO: remove the input and model name's dependency on directory tree
  if (CompareCString(model_name.c_str(), ORT_TSTR("test_")) == 0) {
    model_name = model_name.substr(5);
  }
  std::string narrow_model_name = ToMBString(model_name);
  performance_result_.model_name = narrow_model_name;

  auto p_model = TestModelInfo::LoadOnnxModel(performance_test_config_.model_info.model_file_path.c_str());
  std::unique_ptr<ITestCase> test_case(CreateOnnxTestCase(narrow_model_name, p_model, 0.0, 0.0));

  SessionOptionsWrapper sf(env_);
  const bool enable_cpu_mem_arena = true;
  const std::string& provider_name = performance_test_config_.machine_config.provider_type_name;
  if (provider_name == onnxruntime::kMklDnnExecutionProvider) {
#ifdef USE_MKLDNN
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Mkldnn(sf, enable_cpu_mem_arena ? 1 : 0));
#else
    fprintf(stderr, "MKL-DNN is not supported in this build");
    return false;
#endif
  } else if (provider_name == onnxruntime::kCudaExecutionProvider) {
#ifdef USE_CUDA
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, 0));
#else
    fprintf(stderr, "CUDA is not supported in this build");
    return false;
#endif
  } else if (provider_name == onnxruntime::kNupharExecutionProvider) {
#ifdef USE_NUPHAR
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Nuphar(sf, 0, ""));
#else
    fprintf(stderr, "Nuphar is not supported in this build");
    return false;
#endif
  } else if (provider_name == onnxruntime::kTensorrtExecutionProvider) {
#ifdef USE_TENSORRT
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sf));
    ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_CUDA(sf, 0));
#else
    fprintf(stderr, "TensorRT is not supported in this build");
    return false;
#endif
  } else if (!provider_name.empty() && provider_name != onnxruntime::kCpuExecutionProvider) {
    fprintf(stderr, "This backend is not included in perf test runner.");
    return false;
  }

  if (enable_cpu_mem_arena)
    sf.EnableCpuMemArena();
  else
    sf.DisableCpuMemArena();
  if (performance_test_config_.run_config.enable_sequential_execution)
    sf.EnableSequentialExecution();
  else
    sf.DisableSequentialExecution();
  fprintf(stdout, "Setting thread pool size to %d\n", performance_test_config_.run_config.session_thread_pool_size);
  sf.SetSessionThreadPoolSize(performance_test_config_.run_config.session_thread_pool_size);
  session_object_ = sf.OrtCreateSession(test_case->GetModelUrl());

  auto provider_type = performance_test_config_.machine_config.provider_type_name;
  // Place input tensor on cpu memory if mkldnn provider type to avoid CopyTensor logic in CopyInputAcrossDevices
  // TODO: find a better way to do this.
  if (provider_type == onnxruntime::kMklDnnExecutionProvider) {
    provider_type = onnxruntime::kCpuExecutionProvider;
  }

  if (test_case->GetDataCount() <= 0) {
    LOGS_DEFAULT(ERROR) << "there is no test data for model " << test_case->GetTestCaseName();
    return false;
  }

  test_case->LoadTestData(0 /* id */, b_, feeds_, true);

  input_names_.resize(feeds_.size());
  input_values_.resize(feeds_.size());
  size_t input_index = 0;
  for (auto& kvp : feeds_) {
    input_names_[input_index] = kvp.first.c_str();
    input_values_[input_index] = kvp.second;
    ++input_index;
  }
  size_t output_count;
  ORT_THROW_ON_ERROR(OrtSessionGetOutputCount(session_object_, &output_count));
  output_names_.resize(output_count);
  OrtAllocator* a;
  ORT_THROW_ON_ERROR(OrtCreateDefaultAllocator(&a));
  for (size_t i = 0; i != output_count; ++i) {
    char* output_name = nullptr;
    ORT_THROW_ON_ERROR(OrtSessionGetOutputName(session_object_, i, a, &output_name));
    assert(output_name != nullptr);
    output_names_[i] = output_name;
    a->Free(a, output_name);
  }
  output_names_raw_ptr.resize(output_count);
  for (size_t i = 0; i != output_count; ++i) {
    output_names_raw_ptr[i] = output_names_[i].c_str();
  }
  OrtReleaseAllocator(a);
  output_values_.resize(output_count);
  return true;
}

}  // namespace perftest

}  // namespace onnxruntime
