// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <omp.h>

// onnxruntime dependencies
#include <core/common/common.h>
#include <core/common/logging/sinks/clog_sink.h>
#include <core/common/logging/logging.h>
#include <core/common/status.h>
#include <core/framework/environment.h>
#include <core/session/inference_session.h>
#include <core/platform/env.h>
#include <core/session/IOBinding.h>

#include "test_configuration.h"

namespace onnxruntime {
namespace perftest {

struct PerformanceResult {
  size_t peak_workingset_size{0};
  short average_CPU_usage{0};
  double total_time_cost{0};
  std::vector<double> time_costs;
  std::string model_name;

  void DumpToFile(const std::string& path, bool f_include_statistics = false) const {
    std::ofstream outfile;
    outfile.open(path, std::ofstream::out | std::ofstream::app);
    if (!outfile.good()) {
      LOGF_DEFAULT(ERROR, "failed to open result file");
      return;
    }

    for (size_t runs = 0; runs < time_costs.size(); runs++) {
      outfile << model_name << "," << time_costs[runs] << "," << peak_workingset_size << "," << average_CPU_usage << "," << runs << std::endl;
    }

    if (time_costs.size() > 0 && f_include_statistics) {
      std::vector<double> sorted_time = time_costs;

      size_t total = sorted_time.size();
      size_t n50 = static_cast<size_t>(total * 0.5);
      size_t n90 = static_cast<size_t>(total * 0.9);
      size_t n95 = static_cast<size_t>(total * 0.95);
      size_t n99 = static_cast<size_t>(total * 0.99);
      size_t n999 = static_cast<size_t>(total * 0.999);

      std::sort(sorted_time.begin(), sorted_time.end());

      outfile << std::endl;
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
  PerformanceRunner(const PerformanceTestConfig& test_config) : performance_test_config_(test_config) {}

  Status Run();

  inline const PerformanceResult& GetResult() const { return performance_result_; }

  inline void SerializeResult() const { performance_result_.DumpToFile(performance_test_config_.model_info.result_file_path, performance_test_config_.run_config.f_dump_statistics); }

 private:
  bool Initialize(int count = 1);

  inline Status RunOneIteration(bool isWarmup = false) {
    auto start = std::chrono::high_resolution_clock::now();
    ORT_RETURN_IF_ERROR(session_object_->Run(*io_bindings_.front()));
    auto end = std::chrono::high_resolution_clock::now();

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

  inline Status RunMultipleIteration(bool isWarmup = false, int test_count = 100) {
#pragma omp parallel for
    for (int i = 0; i < static_cast<int>(performance_test_config_.run_config.concurrent_run); ++i) {
      auto start = std::chrono::high_resolution_clock::now();
      for (int j = 0; j < test_count; ++j) {
        session_object_->Run(*io_bindings_[i]);
      }
      auto end = std::chrono::high_resolution_clock::now();

      if (!isWarmup && i == 0) {
        std::chrono::duration<double> duration_seconds = end - start;
        performance_result_.time_costs.emplace_back(duration_seconds.count() / test_count);
        performance_result_.total_time_cost += duration_seconds.count() / test_count;
        if (performance_test_config_.run_config.f_verbose) {
          std::cout << "iteration:" << performance_result_.time_costs.size() << ","
                    << "time_cost:" << performance_result_.time_costs.back() << std::endl;
        }
      }
    }

    return Status::OK();
  }

  inline Status RunFixDuration() {
    if (performance_test_config_.run_config.concurrent_run > 0) {
      while (performance_result_.total_time_cost < performance_test_config_.run_config.duration_in_seconds) {
        ORT_RETURN_IF_ERROR(RunMultipleIteration());
      }
    } else {
      while (performance_result_.total_time_cost < performance_test_config_.run_config.duration_in_seconds) {
        ORT_RETURN_IF_ERROR(RunOneIteration());
      }
    }
    return Status::OK();
  }

  inline Status RunRepeatedTimes() {
    if (performance_test_config_.run_config.concurrent_run > 0) {
      for (size_t ite = 0; ite < performance_test_config_.run_config.repeated_times; ite++) {
        ORT_RETURN_IF_ERROR(RunMultipleIteration());
      }
    } else {
      for (size_t ite = 0; ite < performance_test_config_.run_config.repeated_times; ite++) {
        ORT_RETURN_IF_ERROR(RunOneIteration());
      }
    }
    return Status::OK();
  }

 private:
  PerformanceResult performance_result_;
  PerformanceTestConfig performance_test_config_;
  bool enable_concurrent_ = false;

  std::shared_ptr<::onnxruntime::InferenceSession> session_object_;
  std::vector<std::unique_ptr<IOBinding>> io_bindings_;
};
}  // namespace perftest
}  // namespace onnxruntime
