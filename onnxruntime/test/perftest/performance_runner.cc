// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// TODO: Remove when removing Eigen
#if defined(_MSC_VER)
#pragma warning(disable : 4267)
#endif

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

// TODO: Temporary, while we bring up the threadpool impl...
#include "core/platform/threadpool.h"
#include "onnxruntime_config.h"
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-result"
// cmake/external/eigen/unsupported/Eigen/CXX11/../../../Eigen/src/Core/arch/NEON/PacketMath.h:1633:9:
// error: ‘void* memcpy(void*, const void*, size_t)’ copying an object of non-trivial type ‘Eigen::internal::Packet4c’
// {aka ‘struct Eigen::internal::eigen_packet_wrapper<int, 2>’} from an array of ‘const int8_t’
// {aka ‘const signed char’} [-Werror=class-memaccess]
#ifdef HAS_CLASS_MEMACCESS
#pragma GCC diagnostic ignored "-Wclass-memaccess"
#endif
// eigen-src/unsupported/Eigen/CXX11/src/ThreadPool/EventCount.h:231:56: error: implicit conversion loses integer
//   precision: 'uint64_t' (aka 'unsigned long long') to 'size_t' (aka 'unsigned long') [-Werror,-Wshorten-64-to-32]
// next = wnext == kStackMask ? nullptr : &waiters_[wnext];
//                                         ~~~~~~~~ ^~~~~
#ifdef HAS_SHORTEN_64_TO_32
#pragma GCC diagnostic ignored "-Wshorten-64-to-32"
#endif
#endif
#include <unsupported/Eigen/CXX11/ThreadPool>
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
using DefaultThreadPoolType = Eigen::ThreadPool;
static std::unique_ptr<DefaultThreadPoolType> default_pool;
static std::once_flag default_pool_init;
Eigen::ThreadPoolInterface* GetDefaultThreadPool(const onnxruntime::Env& env) {
  std::call_once(default_pool_init, [&env] {
    int core_num = env.GetNumPhysicalCpuCores();
    default_pool = std::make_unique<DefaultThreadPoolType>(core_num);
  });
  return default_pool.get();
}

namespace onnxruntime {
namespace perftest {

void PerformanceResult::DumpToFile(const std::basic_string<ORTCHAR_T>& path, bool f_include_statistics) const {
  bool have_file = !path.empty();
  std::ofstream outfile;

  if (have_file) {
    outfile.open(path, std::ofstream::out | std::ofstream::app);
    if (!outfile.good()) {
      // at least provide some info on the run
      std::cerr << "failed to open result file '" << ToUTF8String(path.c_str()) << "'. will dump stats to output.\n";
      have_file = false;
      f_include_statistics = true;
    }
  }

  if (have_file) {
    for (size_t runs = 0; runs < time_costs.size(); runs++) {
      outfile << model_name << "," << time_costs[runs] << "," << peak_workingset_size << ","
              << average_CPU_usage << "," << runs << std::endl;
    }
  } else {
    // match formatting of the initial output from PerformanceRunner::Run
    std::cout << "Avg CPU usage:" << average_CPU_usage
              << "\nPeak working set size:" << peak_workingset_size
              << "\nRuns:" << time_costs.size() << std::endl;
  }

  if (!time_costs.empty() && f_include_statistics) {
    std::vector<double> sorted_time = time_costs;

    size_t total = sorted_time.size();
    size_t n50 = static_cast<size_t>(total * 0.5);
    size_t n90 = static_cast<size_t>(total * 0.9);
    size_t n95 = static_cast<size_t>(total * 0.95);
    size_t n99 = static_cast<size_t>(total * 0.99);
    size_t n999 = static_cast<size_t>(total * 0.999);

    std::sort(sorted_time.begin(), sorted_time.end());

    auto output_stats = [&](std::ostream& ostream) {
      ostream << "Min Latency: " << sorted_time[0] << " s\n";
      ostream << "Max Latency: " << sorted_time[total - 1] << " s\n";
      ostream << "P50 Latency: " << sorted_time[n50] << " s\n";
      ostream << "P90 Latency: " << sorted_time[n90] << " s\n";
      ostream << "P95 Latency: " << sorted_time[n95] << " s\n";
      ostream << "P99 Latency: " << sorted_time[n99] << " s\n";
      ostream << "P999 Latency: " << sorted_time[n999] << " s" << std::endl;
    };

    if (have_file) {
      outfile << std::endl;
      output_stats(outfile);
    }

    output_stats(std::cout);
  }
}

void PerformanceRunner::LogSessionCreationTime() {
  std::chrono::duration<double> session_create_duration = session_create_end_ - session_create_start_;
  std::cout << "\nSession creation time cost: " << session_create_duration.count() << " s\n";
}

Status PerformanceRunner::Run() {
  if (!Initialize()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "failed to initialize.");
  }

  // warm up
  initial_inference_result_.start = std::chrono::high_resolution_clock::now();
  ORT_RETURN_IF_ERROR(RunOneIteration<true>());
  initial_inference_result_.end = std::chrono::high_resolution_clock::now();

  // TODO: start profiling
  // if (!performance_test_config_.run_config.profile_file.empty())
  performance_result_.start = std::chrono::high_resolution_clock::now();

  std::unique_ptr<utils::ICPUUsage> p_ICPUUsage = utils::CreateICPUUsage();
  switch (performance_test_config_.run_config.test_mode) {
    case TestMode::kFixDurationMode:
      ORT_RETURN_IF_ERROR(FixDurationTest());
      break;
    case TestMode::KFixRepeatedTimesMode:
      ORT_RETURN_IF_ERROR(RepeatedTimesTest());
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "unknown test mode.");
  }
  performance_result_.end = std::chrono::high_resolution_clock::now();

  performance_result_.average_CPU_usage = p_ICPUUsage->GetUsage();
  performance_result_.peak_workingset_size = utils::GetPeakWorkingSetSize();

  std::chrono::duration<double> session_create_duration = session_create_end_ - session_create_start_;
  // TODO: end profiling
  // if (!performance_test_config_.run_config.profile_file.empty()) session_object->EndProfiling();
  auto first_inference_duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(initial_inference_result_.end - initial_inference_result_.start).count();
  std::chrono::duration<double> inference_duration = performance_result_.end - performance_result_.start;

  std::cout << "Session creation time cost: " << session_create_duration.count() << " s\n"
            << "First inference time cost: " << first_inference_duration << " ms\n"
            << "Total inference time cost: " << performance_result_.total_time_cost << " s\n"  // sum of time taken by each request
            << "Total inference requests: " << performance_result_.time_costs.size() << "\n"
            << "Average inference time cost: " << performance_result_.total_time_cost / performance_result_.time_costs.size() * 1000 << " ms\n"
            // Time between start and end of run. Less than Total time cost when running requests in parallel.
            << "Total inference run time: " << inference_duration.count() << " s\n"
            << "Number of inferences per second: " << performance_result_.time_costs.size() / inference_duration.count() << " \n"
            << "Avg CPU usage: " << performance_result_.average_CPU_usage << " %\n"
            << "Peak working set size: " << performance_result_.peak_workingset_size << " bytes"
            << std::endl;

  return Status::OK();
}

Status PerformanceRunner::FixDurationTest() {
  if (performance_test_config_.run_config.concurrent_session_runs <= 1) {
    return RunFixDuration();
  }

  return RunParallelDuration();
}

Status PerformanceRunner::RepeatedTimesTest() {
  if (performance_test_config_.run_config.concurrent_session_runs <= 1) {
    return RunRepeatedTimes();
  }

  return ForkJoinRepeat();
}

Status PerformanceRunner::RunParallelDuration() {
  // Simple method to continually queue parallel work until the timer has run down.
  // TODO: Make each thread enqueue a new worker.
  auto tpool = GetDefaultThreadPool(Env::Default());
  std::atomic<int> counter = {0};
  OrtMutex m;
  OrtCondVar cv;

  auto start = std::chrono::high_resolution_clock::now();
  auto end = start;
  std::chrono::duration<double> duration_seconds;
  do {
    // We will queue work as deep as requested, ignoring the size of the threadpool itself
    int count = counter.load(std::memory_order_seq_cst);
    while (count < static_cast<int>(performance_test_config_.run_config.concurrent_session_runs)) {
      count++;
      counter++;
      tpool->Schedule([this, &counter, &m, &cv]() {
        auto status = RunOneIteration<false>();
        if (!status.IsOK())
          std::cerr << status.ErrorMessage();
        // Simplified version of Eigen::Barrier
        std::lock_guard<OrtMutex> lg(m);
        counter--;
        cv.notify_all();
      });
    }
    end = std::chrono::high_resolution_clock::now();
    duration_seconds = end - start;
  } while (duration_seconds.count() < performance_test_config_.run_config.duration_in_seconds);

  // Join
  std::unique_lock<OrtMutex> lock(m);
  cv.wait(lock, [&counter]() { return counter == 0; });

  return Status::OK();
}

Status PerformanceRunner::ForkJoinRepeat() {
  const auto& run_config = performance_test_config_.run_config;

  // create a threadpool with one thread per concurrent request
  auto tpool = std::make_unique<DefaultThreadPoolType>(run_config.concurrent_session_runs);
  std::atomic<int> counter{0}, requests{0};
  OrtMutex m;
  OrtCondVar cv;

  // Fork
  for (size_t i = 0; i != run_config.concurrent_session_runs; ++i) {
    counter++;
    tpool->Schedule([this, &counter, &requests, &m, &cv, &run_config]() {
      while (requests++ < static_cast<int>(run_config.repeated_times)) {
        auto status = RunOneIteration<false>();
        if (!status.IsOK())
          std::cerr << status.ErrorMessage();
      }

      // Simplified version of Eigen::Barrier
      std::lock_guard<OrtMutex> lg(m);
      counter--;
      cv.notify_all();
    });
  }

  // Join
  std::unique_lock<OrtMutex> lock(m);
  cv.wait(lock, [&counter]() { return counter == 0; });

  return Status::OK();
}

static std::unique_ptr<TestModelInfo> CreateModelInfo(const PerformanceTestConfig& performance_test_config_) {
  if (CompareCString(performance_test_config_.backend.c_str(), ORT_TSTR("ort")) == 0) {
    const auto& file_path = performance_test_config_.model_info.model_file_path;
#if !defined(ORT_MINIMAL_BUILD)
    if (HasExtensionOf(file_path, ORT_TSTR("onnx"))) {
      return TestModelInfo::LoadOnnxModel(performance_test_config_.model_info.model_file_path.c_str());
    }
#endif

    if (HasExtensionOf(file_path, ORT_TSTR("ort"))) {
      return TestModelInfo::LoadOrtModel(performance_test_config_.model_info.model_file_path.c_str());
    }

    ORT_NOT_IMPLEMENTED(ToUTF8String(file_path), " is not supported");
  }

  if (CompareCString(performance_test_config_.backend.c_str(), ORT_TSTR("tf")) == 0) {
    return TFModelInfo::Create(performance_test_config_.model_info.model_file_path.c_str());
  }

  ORT_NOT_IMPLEMENTED(ToUTF8String(performance_test_config_.backend), " is not supported");
}

static std::unique_ptr<TestSession> CreateSession(Ort::Env& env, std::random_device& rd,
                                                  const PerformanceTestConfig& performance_test_config_,
                                                  const TestModelInfo& test_model_info) {
  if (CompareCString(performance_test_config_.backend.c_str(), ORT_TSTR("ort")) == 0) {
    return std::make_unique<OnnxRuntimeTestSession>(env, rd, performance_test_config_, test_model_info);
  }
#ifdef HAVE_TENSORFLOW
  if (CompareCString(performance_test_config_.backend.c_str(), ORT_TSTR("tf")) == 0) {
    return new TensorflowTestSession(rd, performance_test_config_, test_model_info);
  }
#endif
  ORT_NOT_IMPLEMENTED(ToUTF8String(performance_test_config_.backend), " is not supported");
}

PerformanceRunner::PerformanceRunner(Ort::Env& env, const PerformanceTestConfig& test_config, std::random_device& rd)
    : performance_test_config_(test_config),
      test_model_info_(CreateModelInfo(test_config)) {
  session_create_start_ = std::chrono::high_resolution_clock::now();
  session_ = CreateSession(env, rd, test_config, *test_model_info_);
  session_create_end_ = std::chrono::high_resolution_clock::now();
}

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
  std::string narrow_model_name = ToUTF8String(model_name);
  performance_result_.model_name = narrow_model_name;

  // ownership semantics are a little unexpected here as the test case takes ownership of the model info
  TestModelInfo* test_model_info = test_model_info_.get();
  test_case_ = CreateOnnxTestCase(narrow_model_name, std::move(test_model_info_), 0.0, 0.0);

  if (performance_test_config_.run_config.generate_model_input_binding) {
    return static_cast<OnnxRuntimeTestSession*>(
               session_.get())
        ->PopulateGeneratedInputTestData(performance_test_config_.run_config.random_seed_for_input_data);
  }

  // TODO: Place input tensor on cpu memory if dnnl provider type to avoid CopyTensor logic in CopyInputAcrossDevices
  size_t test_data_count = test_case_->GetDataCount();
  if (test_data_count == 0) {
    std::cout << "there is no test data for model " << test_case_->GetTestCaseName() << std::endl;
    return false;
  }
  for (size_t test_data_id = 0; test_data_id != test_data_count; ++test_data_id) {
    std::unordered_map<std::string, Ort::Value> feeds;
    test_case_->LoadTestData(test_data_id /* id */, b_, feeds, true);
    // Discard the names in feeds
    int input_count = test_model_info->GetInputCount();
    for (int i = 0; i != input_count; ++i) {
      auto iter = feeds.find(test_model_info->GetInputName(i));
      if (iter == feeds.end()) {
        std::cout << "there is no test input data for input " << test_model_info->GetInputName(i) << " and model "
                  << test_case_->GetTestCaseName() << std::endl;
        return false;
      }
      session_->PreLoadTestData(test_data_id, static_cast<size_t>(i), std::move(iter->second));
    }
  }

  return true;
}

}  // namespace perftest

}  // namespace onnxruntime
