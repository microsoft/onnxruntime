// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/shared_library/provider_api.h"
#ifdef USE_CUDA
#include <cuda_runtime_api.h>
#include "core/providers/cuda/cuda_execution_provider.h"
#include "core/providers/cuda/tunable/cuda_tunable.h"
#include "core/providers/cuda/tunable/util.h"
#elif USE_ROCM
#include <hip/hip_runtime.h>
#include "core/providers/rocm/rocm_execution_provider.h"
#include "core/providers/rocm/tunable/rocm_tunable.h"
#include "core/providers/rocm/tunable/util.h"
#endif

#ifdef USE_CUDA
using onnxruntime::cuda::tunable::Timer;
using ExecutionProvider = onnxruntime::CUDAExecutionProvider;
using ExecutionProviderInfo = onnxruntime::CUDAExecutionProviderInfo;
using StreamT = cudaStream_t;
using TuningContextT = onnxruntime::cuda::tunable::CudaTuningContext;
#elif USE_ROCM
using onnxruntime::rocm::tunable::Timer;
using ExecutionProvider = onnxruntime::ROCMExecutionProvider;
using ExecutionProviderInfo = onnxruntime::ROCMExecutionProviderInfo;
using StreamT = hipStream_t;
using TuningContextT = onnxruntime::rocm::tunable::RocmTuningContext;
#else
#error "kernel explorer only supports CUDA or ROCM"
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
  TuningContextT* TuningContext() {
    if (ep_ == nullptr) {
      ExecutionProviderInfo info{};
      ep_ = std::make_unique<ExecutionProvider>(info);
    }

    return static_cast<TuningContextT*>(ep_->GetTuningContext());
  }

  StreamT Stream() { return stream_; }

 private:
  std::unique_ptr<ExecutionProvider> ep_{};
  StreamT stream_{0};
  int repeats_{100};
};
