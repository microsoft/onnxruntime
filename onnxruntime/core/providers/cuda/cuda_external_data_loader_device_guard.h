// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace cuda {

// Makes `device_id` current on the calling thread and restores the previous device on destruction.
class CudaDeviceGuard {
 public:
  CudaDeviceGuard() = default;
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudaDeviceGuard);

  common::Status SetDevice(int device_id) {
    CUDA_RETURN_IF_ERROR(cudaGetDevice(&previous_device_));
    if (previous_device_ != device_id) {
      CUDA_RETURN_IF_ERROR(cudaSetDevice(device_id));
      restore_device_ = true;
    }
    return common::Status::OK();
  }

  ~CudaDeviceGuard() {
    if (restore_device_) {
      ORT_IGNORE_RETURN_VALUE(CUDA_CALL(cudaSetDevice(previous_device_)));
    }
  }

 private:
  int previous_device_{-1};
  bool restore_device_{false};
};

}  // namespace cuda
}  // namespace onnxruntime
