// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include "core/providers/rocm/tunable/tunable.h"
#include "contrib_ops/rocm/bert/util.h"

using onnxruntime::rocm::tunable::Timer;

/// Wrapping around Op and TunableOp
class IKernelExplorer {
 public:
  virtual void Run() = 0;

  void SetRepeats(int n) {
    repeats_ = n;
  }

  float Profile() {
    // warm up
    for (int i = 0; i < 5; i++) {
      Run();
    }
    Timer timer{Stream()};
    timer.Start();
    for (int i = 0; i < repeats_; i++) {
      Run();
    }
    timer.End();
    return timer.Duration() / repeats_;
  }

  virtual ~IKernelExplorer() {
    if (stream_ != nullptr) {
      HIP_CHECK(hipStreamDestroy(stream_));
    }
  }

 protected:
  hipStream_t Stream() { return stream_; }

 private:
  hipStream_t stream_{[]() { hipStream_t stream; HIP_CHECK(hipStreamCreate(&stream)); return stream; }()};
  int repeats_{100};
};
