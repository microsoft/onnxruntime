// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>

#include "core/providers/rocm/tunable/rocm_tunable.h"

namespace onnxruntime {
namespace rocm {
namespace tunable {

class Timer : public ::onnxruntime::tunable::Timer<hipStream_t> {
 public:
  using TimerBase = ::onnxruntime::tunable::Timer<hipStream_t>;

  explicit Timer(hipStream_t stream);

  void Start() override;
  void End() override;
  float Duration() override;
  ~Timer();

 private:
  hipEvent_t start_;
  hipEvent_t end_;
};

}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
