// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iostream>
#include <chrono>
#include <functional>

namespace onnxruntime {

template <typename Func>
auto measure_latency(Func&& func) {
  auto start = std::chrono::high_resolution_clock::now();
  func();
  auto end = std::chrono::high_resolution_clock::now();

  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();  // Latency in microseconds
}

std::string current_time_string(bool no_date=true) {
  std::ostringstream os;
  auto now = std::chrono::system_clock::now();
  auto in_time_t = std::chrono::system_clock::to_time_t(now);
  os << std::put_time(std::localtime(&in_time_t), no_date ? "%H:%M:%S" : "%F %T%z");
  return os.str();
}

}  // namespace onnxruntime
