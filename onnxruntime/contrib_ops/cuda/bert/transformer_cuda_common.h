// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_common.h"

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
      cudaEventDestroy(cuda_event_);
  }

  cudaEvent_t& Get() {
    return cuda_event_;
  }

 private:
  cudaEvent_t cuda_event_;
};

// A wrapper class of cudaStream_t to destroy the stream automatically for avoiding memory leak.
class AutoDestoryCudaStream {
 public:
  AutoDestoryCudaStream() : cuda_stream_(nullptr) {
  }

  ~AutoDestoryCudaStream() {
    if (cuda_stream_ != nullptr)
      cudaStreamDestroy(cuda_stream_);
  }

  cudaStream_t& Get() {
    return cuda_stream_;
  }

 private:
  cudaStream_t cuda_stream_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
