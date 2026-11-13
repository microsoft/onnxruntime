// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// The WebGPU EP can be enabled via the generic SessionOptionsAppendExecutionProvider method, so no direct usage of
// the provider factory is required.

#ifndef ORT_WEBGPU_PROVIDER_FACTORY_H
#define ORT_WEBGPU_PROVIDER_FACTORY_H

#include <stddef.h>
#include <stdint.h>
#include "onnxruntime_c_api.h"

#define ORT_DAWN_CACHE_INTERFACE_VERSION 1u
#define ORT_WEBGPU_DAWN_CACHE_INTERFACE_OPTION "dawnCacheInterface"

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

// Native WebGPU session option: ep.webgpuexecutionprovider.dawnCacheInterface.
// The value is the decimal address of this interface. Omission inherits an
// existing device's cache configuration, or disables caching on first creation.
// "0" explicitly requests disabled caching. Only trusted in-process pointers are supported; the interface,
// callback state and isolation key must outlive the WebGPU device and must not
// be mutated while attached. Callbacks
// may run concurrently and must not throw.
//
// Configure before the first session or shared allocation creates the device.
// Sessions sharing a device may omit the option or provide the same interface
// address. A different address, including "0" for a cached device, is rejected. A cache
// cannot be attached to an externally supplied device; configure that device's
// Dawn cache descriptor before passing it to ORT instead.
// Invalid configuration follows ORT's no-exceptions fail-fast policy when built
// without exceptions.
typedef struct OrtDawnCacheInterface {
  uint32_t version;
  // A null value requests the blob size. Otherwise copy only when value_size
  // matches that size. Return the size copied, or zero for a miss.
  size_t (*load)(const void* key, size_t key_size,
                 void* value, size_t value_size, void* userdata);
  void (*store)(const void* key, size_t key_size,
                const void* value, size_t value_size, void* userdata);
  void* userdata;
  // Optional null-terminated cache namespace. Null uses Dawn's default.
  const char* isolation_key;
} OrtDawnCacheInterface;

#ifdef __cplusplus
}
#endif

#endif  // ORT_WEBGPU_PROVIDER_FACTORY_H
