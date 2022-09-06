// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include "contrib_ops/rocm/bert/util.h"

using onnxruntime::contrib::rocm::Timer;

// To be deleted after we remove the use of namespace onnxruntime::rocm
// in contrib_ops/rocm/bert/fast_gelu_impl_kernel.h
namespace onnxruntime {
namespace rocm {
}  // namespace rocm
}  // namespace onnxruntime

class Operator {
 public:
  Operator() : stream_(0), repeats_(100) {}

  virtual void Run() = 0;

  void SetRepeats(int n) {
    repeats_ = n;
  }

  float Profile() {
    // warm up
    for (int i = 0; i < 5; i++) {
      Run();
    }
    Timer timer;
    timer.Start();
    for (int i = 0; i < repeats_; i++) {
      Run();
    }
    timer.End();
    return timer.time() / repeats_;
  }

  virtual ~Operator() {}

 protected:
  hipStream_t stream_;

 private:
  int repeats_;
};
