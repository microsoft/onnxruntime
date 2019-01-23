// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "performance_runner.h"
#include "TestCase.h"
#include <experimental/filesystem>
#ifdef _MSC_VER
#include <filesystem>
#endif
#include "core/graph/graph_viewer.h"  //for onnxruntime::NodeArg
#include "utils.h"
#include "testenv.h"
#include "providers.h"

using namespace std::experimental::filesystem::v1;
using onnxruntime::Status;

namespace onnxruntime {
namespace perftest {
Status PerformanceRunner::Run() {
  if (performance_test_config_.run_config.concurrent_run > 0) {
    if (!Initialize(static_cast<int>(performance_test_config_.run_config.concurrent_run))) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "failed to initialize.");
    }
    // warm up
    RunMultipleIteration(true /*isWarmup*/);
  } else {
    if (!Initialize()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "failed to initialize.");
    }
    // warm up
    RunOneIteration(true /*isWarmup*/);
  }

  if (!performance_test_config_.run_config.profile_file.empty())
    session_object_->StartProfiling(performance_test_config_.run_config.profile_file);

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

  if (!performance_test_config_.run_config.profile_file.empty())
    session_object_->EndProfiling();

  std::cout << "Total time cost:" << performance_result_.total_time_cost << std::endl
            << "Total iterations:" << performance_result_.time_costs.size() << std::endl
            << "Average time cost:" << performance_result_.total_time_cost / performance_result_.time_costs.size() * 1000 << " ms" << std::endl;
  return Status::OK();
}

bool PerformanceRunner::Initialize(int count) {
  path model_path(performance_test_config_.model_info.model_file_path);
  if (model_path.extension() != ".onnx") {
    LOGF_DEFAULT(ERROR, "input path is not a valid model");
    return false;
  }

  // TO DO: remove the input and model name's dependency on directory tree
  std::string model_name = model_path.parent_path().filename().string();
  if (model_name.compare(0, 5, "test_") == 0) model_name = model_name.substr(5);
  performance_result_.model_name = model_name;

  // TO DO: remove depedency on OnnxTestCase.
  std::unique_ptr<ITestCase> test_case(CreateOnnxTestCase(model_name));

  if (!test_case->SetModelPath(model_path).IsOK()) {
    LOGF_DEFAULT(ERROR, "load model failed");
    return false;
  }

  std::vector<std::string> provider_types;
  if (performance_test_config_.machine_config.provider_type_name == onnxruntime::kCpuExecutionProvider) {
    provider_types = {onnxruntime::kMklDnnExecutionProvider, onnxruntime::kCpuExecutionProvider};
  }
  provider_types = {performance_test_config_.machine_config.provider_type_name};
  SessionFactory sf(provider_types, true, true);
  sf.enable_sequential_execution = performance_test_config_.run_config.enable_sequential_execution;
  sf.session_thread_pool_size = count;

  sf.create(session_object_, test_case->GetModelUrl(), test_case->GetTestCaseName());

  // Initialize IO Binding
  for (int i = 0; i < count; ++i) {
    std::unique_ptr<IOBinding> io_binding;
    if (!session_object_->NewIOBinding(&io_binding).IsOK()) {
      LOGF_DEFAULT(ERROR, "Failed to init session and IO binding");
      return false;
    }
    io_bindings_.emplace_back(std::move(io_binding));
  }

  auto provider_type = performance_test_config_.machine_config.provider_type_name;
  // Place input tensor on cpu memory if mkldnn provider type to avoid CopyTensor logic in CopyInputAcrossDevices
  // TODO: find a better way to do this.
  if (provider_type == onnxruntime::kMklDnnExecutionProvider) {
    provider_type = onnxruntime::kCpuExecutionProvider;
  }
  // use the first allocator
  AllocatorPtr cpu_allocator = io_bindings_.front()->GetCPUAllocator(0, provider_type);
  test_case->SetAllocator(cpu_allocator);

  if (test_case->GetDataCount() <= 0) {
    LOGF_DEFAULT(ERROR, "there is no test data for model %s", test_case->GetTestCaseName().c_str());
    return false;
  }

  for (int i = 0; i < count; ++i) {
    std::unordered_map<std::string, ::onnxruntime::MLValue> feeds;
    test_case->LoadTestData(0 /* id */, feeds, true);  // load the same data, in case user not creating test set
    for (auto feed : feeds) {
      io_bindings_[i]->BindInput(feed.first, feed.second);
    }
    auto outputs = session_object_->GetModelOutputs();
    auto status = outputs.first;
    if (!outputs.first.IsOK()) {
      LOGF_DEFAULT(ERROR, "GetOutputs failed, TestCaseName:%s, ErrorMessage:%s",
                   test_case->GetTestCaseName().c_str(),
                   status.ErrorMessage().c_str());
      return false;
    }

    std::vector<MLValue> output_mlvalues(outputs.second->size());
    for (size_t i_output = 0; i_output < outputs.second->size(); ++i_output) {
      auto output = outputs.second->at(i_output);
      if (!output) continue;
      io_bindings_[i]->BindOutput(output->Name(), output_mlvalues[i_output]);
    }
  }

  return true;
}

}  // namespace perftest

}  // namespace onnxruntime
