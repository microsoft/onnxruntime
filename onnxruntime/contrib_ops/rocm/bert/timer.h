// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "hip/hip_runtime.h"

namespace onnxruntime {
namespace contrib {
namespace rocm {

class Timer {
 public:
  Timer();
  void Start();
  void End();
  float time();
  ~Timer();

 private:
  hipEvent_t start_, end_; 
};

}  // namespace rocm
}  // namespace contrib
}  // namespace onnxruntime
