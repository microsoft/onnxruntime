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

}  // namespace onnxruntime
