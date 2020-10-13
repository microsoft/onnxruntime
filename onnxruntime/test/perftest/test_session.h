// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdlib.h>

#include "OrtValueList.h"

namespace onnxruntime {
namespace perftest {
class TestSession {
 public:
  virtual std::chrono::duration<double> Run() = 0;
  // TODO: implement it
  // This function won't return duration, because it may vary largely.
  // Please measure the perf at a higher level.
  void ThreadSafeRun() { abort(); }
  virtual void PreLoadTestData(size_t test_data_id, size_t input_id, Ort::Value&& value) = 0;

  virtual ~TestSession() = default;
};
}  // namespace perftest
}  // namespace onnxruntime