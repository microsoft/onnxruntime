// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "core/providers/cuda/tunable/cuda_tunable.h"
#include "core/providers/cuda/tunable/util.h"
#elif USE_ROCM
#include <hip/hip_runtime.h>
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "core/providers/rocm/tunable/util.h"
#endif

#ifdef USE_CUDA
using onnxruntime::cuda::tunable::Timer;
using StreamT = cudaStream_t;
#elif USE_ROCM
using onnxruntime::rocm::tunable::Timer;
using StreamT = hipStream_t;
#endif

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

  virtual ~IKernelExplorer() = default;

 protected:
  StreamT Stream() { return stream_; }

 private:
  StreamT stream_{0};
  int repeats_{100};
};
