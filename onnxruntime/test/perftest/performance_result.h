// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <mutex>
#include <iostream>
#include <random>
#include <chrono>
// onnxruntime dependencies
#include <core/common/common.h>
#include <core/common/status.h>
#include <core/platform/env.h>
#include <core/session/onnxruntime_c_api.h>
#include "test_configuration.h"
#include "query_sample.h"

namespace onnxruntime {
namespace perftest {

struct PerformanceResult {
  std::chrono::time_point<std::chrono::high_resolution_clock> start_;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_;
  size_t peak_workingset_size{0};
  short average_CPU_usage{0};
  mlperf::QuerySampleLatency total_time_cost{0};
  std::vector<mlperf::QuerySampleLatency> time_costs;
  std::string model_name;

  void DumpToFile(const std::basic_string<ORTCHAR_T>& path, bool f_include_statistics = false) const {
    std::ofstream outfile;
    outfile.open(path, std::ofstream::out | std::ofstream::app);
    if (!outfile.good()) {
      printf("failed to open result file");
      return;
    }

    for (size_t runs = 0; runs < time_costs.size(); runs++) {
      outfile << model_name << "," << time_costs[runs] << "," << peak_workingset_size << "," << average_CPU_usage << ","
              << runs << std::endl;
    }

    if (!time_costs.empty() && f_include_statistics) {
      std::vector<mlperf::QuerySampleLatency> sorted_time = time_costs;

      size_t total = sorted_time.size();
      size_t n50 = static_cast<size_t>(total * 0.5);
      size_t n90 = static_cast<size_t>(total * 0.9);
      size_t n95 = static_cast<size_t>(total * 0.95);
      size_t n99 = static_cast<size_t>(total * 0.99);
      size_t n999 = static_cast<size_t>(total * 0.999);

      std::sort(sorted_time.begin(), sorted_time.end());

      outfile << std::endl;
      auto output_stats = [&](std::ostream& ostream) {
        ostream << "Min Latency is " << sorted_time[0]/1000000000.0 << " sec" << std::endl;
        ostream << "Max Latency is " << sorted_time[total - 1]/1000000000.0 << " sec" << std::endl;
        ostream << "P50 Latency is " << sorted_time[n50]/1000000000.0 << " sec" << std::endl;
        ostream << "P90 Latency is " << sorted_time[n90]/1000000000.0 << " sec" << std::endl;
        ostream << "P95 Latency is " << sorted_time[n95]/1000000000.0 << " sec" << std::endl;
        ostream << "P99 Latency is " << sorted_time[n99]/1000000000.0 << " sec" << std::endl;
        ostream << "P999 Latency is " << sorted_time[n999]/1000000000.0 << " sec" << std::endl;
      };

      output_stats(outfile);
      output_stats(std::cout);
    }

    outfile.close();
  }
};

}  // namespace perftest
}  // namespace onnxruntime
