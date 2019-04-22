// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <mutex>

// onnxruntime dependencies
#include <core/common/common.h>
#include <core/common/status.h>
#include <core/platform/env.h>
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
  size_t peak_workingset_size{0};
  short average_CPU_usage{0};
  double total_time_cost{0};
  std::vector<double> time_costs;
  std::string model_name;

  void DumpToFile(const std::basic_string<ORTCHAR_T>& path, bool f_include_statistics = false) const {
    std::ofstream outfile;
    outfile.open(path, std::ofstream::out | std::ofstream::app);
    if (!outfile.good()) {
      printf("failed to open result file");
      return;
    }

    for (size_t runs = 0; runs < time_costs.size(); runs++) {
      outfile << model_name << "," << time_costs[runs] << "," << peak_workingset_size << "," << average_CPU_usage << "," << runs << std::endl;
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

      outfile << std::endl;
      outfile << "Min Latency is " << sorted_time[0] << "sec" << std::endl;
      outfile << "Max Latency is " << sorted_time[total - 1] << "sec" << std::endl;
      outfile << "P50 Latency is " << sorted_time[n50] << "sec" << std::endl;
      outfile << "P90 Latency is " << sorted_time[n90] << "sec" << std::endl;
      outfile << "P95 Latency is " << sorted_time[n95] << "sec" << std::endl;
      outfile << "P99 Latency is " << sorted_time[n99] << "sec" << std::endl;
      outfile << "P999 Latency is " << sorted_time[n999] << "sec" << std::endl;
    }

    outfile.close();
  }
};

class PerformanceRunner {
 public:
  PerformanceRunner(OrtEnv* env, const PerformanceTestConfig& test_config);

  ~PerformanceRunner();
  Status Run();

  inline const PerformanceResult& GetResult() const { return performance_result_; }

  inline void SerializeResult() const {
    performance_result_.DumpToFile(performance_test_config_.model_info.result_file_path,
                                   performance_test_config_.run_config.f_dump_statistics);
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PerformanceRunner);

 private:
  bool Initialize();
  Status RunOneIteration(bool isWarmup = false);

  Status FixDurationTest();
  Status RepeatedTimesTest();
  Status ForkJoinRepeat();
  Status RunParallelDuration();

  inline Status RunFixDuration() {
    while (performance_result_.total_time_cost < performance_test_config_.run_config.duration_in_seconds) {
      ORT_RETURN_IF_ERROR(RunOneIteration());
    }
    return Status::OK();
  }

  inline Status RunRepeatedTimes() {
    for (size_t ite = 0; ite < performance_test_config_.run_config.repeated_times; ite++) {
      ORT_RETURN_IF_ERROR(RunOneIteration());
    }
    return Status::OK();
  }

 private:
  PerformanceResult performance_result_;
  PerformanceTestConfig performance_test_config_;
  TestModelInfo* test_model_info_;
  std::unique_ptr<TestSession> session_;
  OrtValueArray inputs_;
  HeapBuffer b_;
  std::unique_ptr<ITestCase> test_case_;

  // TODO: Convert to OrtMutex
  std::mutex results_mutex_;
};
}  // namespace perftest
}  // namespace onnxruntime
