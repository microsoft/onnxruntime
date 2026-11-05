// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// WebGPU is enabled through SessionOptionsAppendExecutionProvider. This header
// supplies the optional host clock-calibration contract and also identifies
// builds that include the WebGPU EP, including the ONNX Runtime C cocoapod.

#pragma once

#include <stdint.h>
#include "onnxruntime_c_api.h"

#ifdef __cplusplus
extern "C" {
#endif

// A sample correlating resolved WebGPU query timestamps with Windows QPC ticks.
// The host must account for its Dawn timestamp conversion/toggles: GPU values use
// exactly the same units as resolved queries, and gpu_to_cpu_scale converts their
// deltas to CPU nanoseconds. Return failure if this correlation is unavailable.
typedef struct OrtWebGpuClockCalibration {
  uint64_t gpu_timestamp_ns;
  uint64_t cpu_qpc_ticks;
  double gpu_to_cpu_scale;
} OrtWebGpuClockCalibration;

// Pass this struct's address, as a decimal uintptr_t string, through the WebGPU
// provider option "clockCalibration". ORT copies the struct during EP creation;
// callback and user_data must remain valid until the WebGPU context is released.
// Sessions sharing a context must supply the same callback/user_data. The host
// must use the Dawn implementation owning device (the opaque WGPUDevice handle).
// Called only during profiling on Windows D3D12, once per inference run. Return
// nonzero on success, zero for fallback. The callback must not throw or wait for
// GPU work; it must write all result fields on success.
typedef struct OrtWebGpuClockCalibrationCallbackInfo {
  uint32_t version;  // Must be 1.
  int(ORT_API_CALL* callback)(void* device, OrtWebGpuClockCalibration* result, void* user_data);
  void* user_data;
} OrtWebGpuClockCalibrationCallbackInfo;

#ifdef __cplusplus
}
#endif
