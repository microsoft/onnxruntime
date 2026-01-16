#pragma once

#include <string>
#include <vector>

/**
 * @struct CompilationResult
 * @brief Struct for storing the results of a model compilation.
 *
 * This struct was inspired by the implementation in the following repository:
 * https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/perftest/performance_runner.cc
 */
struct CompilationResult
{
    int64_t compilation_time_cost_ms = 0;
    size_t peak_workingset_size = 0;
    size_t peak_pagefile_usage = 0;
};
