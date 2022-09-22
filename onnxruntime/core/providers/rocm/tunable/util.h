// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <hip/hip_runtime.h>
#include <cstdlib>
#include <iostream>

namespace onnxruntime {
namespace rocm {
namespace tunable {

class Timer {
 public:
  explicit Timer(hipStream_t stream);
  void Start();
  void End();
  float Duration();
  ~Timer();

 private:
  hipStream_t stream_;
  hipEvent_t start_;
  hipEvent_t end_;
};

}  // namespace tunable
}  // namespace rocm
}  // namespace onnxruntime
