// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime_api.h>

#include "core/providers/cuda/tunable/cuda_tunable.h"

namespace onnxruntime {
namespace cuda {
namespace tunable {

class Timer : public ::onnxruntime::tunable::Timer<cudaStream_t> {
 public:
  using TimerBase = ::onnxruntime::tunable::Timer<cudaStream_t>;

  explicit Timer(cudaStream_t stream);

  void Start() override;
  void End() override;
  float Duration() override;
  ~Timer();

 private:
  cudaEvent_t start_;
  cudaEvent_t end_;
};

}  // namespace tunable
}  // namespace cuda
}  // namespace onnxruntime
