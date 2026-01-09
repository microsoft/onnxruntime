#pragma once

#include <string>
#include <vector>

/**
 * @struct PerformanceResult
 * @brief Struct for storing the results of a performance test.
 *
 * This struct was inspired by the implementation in the following repository:
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/perftest/performance_runner.cc
 */
struct PerformanceResult
{
    std::vector<int64_t> inference_time_costs_ms;
    size_t peak_workingset_size = 0;
    size_t peak_pagefile_usage = 0;
    short average_cpu_usage = 0;
};
