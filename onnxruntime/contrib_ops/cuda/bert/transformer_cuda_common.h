// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// A wrapper class of cudaEvent_t to destroy the event automatically for avoiding memory leak.
class AutoDestoryCudaEvent {
 public:
  AutoDestoryCudaEvent() : cuda_event_(nullptr) {
  }

  ~AutoDestoryCudaEvent() {
    if (cuda_event_ != nullptr)
      (void)cudaEventDestroy(cuda_event_);
  }

  cudaEvent_t& Get() {
    return cuda_event_;
  }

 private:
  cudaEvent_t cuda_event_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
