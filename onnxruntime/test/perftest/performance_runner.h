// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <random>
#include <chrono>
// onnxruntime dependencies
#include <core/common/common.h>
#include <core/common/status.h>
#include <core/platform/env.h>
#include <core/platform/ort_mutex.h>
#include <core/session/onnxruntime_cxx_api.h>
#include "test_configuration.h"
#include "heap_buffer.h"
#include "test_session.h"
#include "OrtValueList.h"

class ITestCase;
class TestModelInfo;

namespace onnxruntime {
namespace perftest {

struct PerformanceResult {
  std::chrono::time_point<std::chrono::high_resolution_clock> start;
  std::chrono::time_point<std::chrono::high_resolution_clock> end;
  size_t peak_workingset_size{0};
  short average_CPU_usage{0};
  double total_time_cost{0};
  std::vector<double> time_costs;
  std::string model_name;

  void DumpToFile(const std::basic_string<ORTCHAR_T>& path, bool f_include_statistics = false) const;
};

class PerformanceRunner {
 public:
  PerformanceRunner(Ort::Env& env, const PerformanceTestConfig& test_config, std::random_device& rd);

  ~PerformanceRunner();
  Status Run();

  void LogSessionCreationTime();

  inline const PerformanceResult& GetResult() const { return performance_result_; }

  inline void SerializeResult() const {
    performance_result_.DumpToFile(performance_test_config_.model_info.result_file_path,
                                   performance_test_config_.run_config.f_dump_statistics);
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PerformanceRunner);

 private:
  bool Initialize();

  template <bool isWarmup>
  Status RunOneIteration() {
    std::chrono::duration<double> duration_seconds(std::chrono::seconds(0));

    auto status = Status::OK();
    ORT_TRY {
      duration_seconds = session_->Run();
    }
    ORT_CATCH(const std::exception& ex) {
      ORT_HANDLE_EXCEPTION([&]() {
        status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "PerformanceRunner::RunOneIteration caught exception: ", ex.what());
      });
    }
    ORT_RETURN_IF_ERROR(status);

    if (!isWarmup) {
      std::lock_guard<OrtMutex> guard(results_mutex_);
      performance_result_.time_costs.emplace_back(duration_seconds.count());
      performance_result_.total_time_cost += duration_seconds.count();
      if (performance_test_config_.run_config.f_verbose) {
        std::cout << "iteration:" << performance_result_.time_costs.size() << ","
                  << "time_cost:" << performance_result_.time_costs.back() << std::endl;
      }
    }
    return Status::OK();
  }

  Status FixDurationTest();
  Status RepeatedTimesTest();
  Status ForkJoinRepeat();
  Status RunParallelDuration();

  inline Status RunFixDuration() {
    while (performance_result_.total_time_cost < performance_test_config_.run_config.duration_in_seconds) {
      ORT_RETURN_IF_ERROR(RunOneIteration<false>());
    }
    return Status::OK();
  }

  inline Status RunRepeatedTimes() {
    for (size_t ite = 0; ite < performance_test_config_.run_config.repeated_times; ite++) {
      ORT_RETURN_IF_ERROR(RunOneIteration<false>());
    }
    return Status::OK();
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock> session_create_start_;
  std::chrono::time_point<std::chrono::high_resolution_clock> session_create_end_;
  PerformanceResult initial_inference_result_;
  PerformanceResult performance_result_;
  PerformanceTestConfig performance_test_config_;
  std::unique_ptr<TestModelInfo> test_model_info_;
  std::unique_ptr<TestSession> session_;
  onnxruntime::test::HeapBuffer b_;
  std::unique_ptr<ITestCase> test_case_;

  OrtMutex results_mutex_;
};
}  // namespace perftest
}  // namespace onnxruntime
