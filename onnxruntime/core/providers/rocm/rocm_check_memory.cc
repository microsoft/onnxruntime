// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "rocm_check_memory.h"
#include "rocm_common.h"

namespace onnxruntime {
void CheckIfMemoryOnCurrentGpuDevice(const void* ptr) {
  hipPointerAttributes_t attrs;
  HIP_CALL(hipPointerGetAttributes(&attrs, ptr));
  int current_device;
  HIP_CALL(hipGetDevice(&current_device));
  ORT_ENFORCE(attrs.device == current_device,
              "Current GPU device is ", current_device,
              " but the memory of pointer ", ptr,
              " is allocated on device ", attrs.device);
}
}  // onnxruntime
