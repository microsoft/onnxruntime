// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/cuda_check_memory.h"
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
void CheckIfMemoryOnCurrentGpuDevice(const void* ptr) {
  cudaPointerAttributes attrs;
  CUDA_CALL(cudaPointerGetAttributes(&attrs, ptr));
  int current_device;
  CUDA_CALL(cudaGetDevice(&current_device));
  ORT_ENFORCE(attrs.device == current_device,
              "Current CUDA device is ", current_device,
              " but the memory of pointer ", ptr,
              " is allocated on device ", attrs.device);
}
}  // onnxruntime
