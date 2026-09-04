// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/device_discovery.h"

#include <emscripten.h>

namespace onnxruntime {

namespace {

// Returns true if the JS environment exposes a WebGPU entry point.
//
// This is a capability check rather than a hardware enumeration, because a browser does not offer anything
// better. A GPUAdapter is only obtainable through the asynchronous navigator.gpu.requestAdapter(), which cannot
// be called from here, and even once obtained GPUAdapterInfo masks vendor details for fingerprinting
// resistance. The presence of navigator.gpu is therefore the strongest signal available at discovery time. It is
// also the same signal ORT Web itself uses to decide whether WebGPU is usable (see initEp() in
// js/web/lib/wasm/wasm-core-impl.ts).
//
// navigator is defined in both window and worker scopes, so this works on either. Node.js has no native WebGPU
// implementation (nodejs/node#42896 was closed as not planned), but the userland Dawn based packages install their
// GPU object onto globalThis, so this check picks those up too. Where neither is present no GPU is reported.
bool IsWebGpuAvailable() {
  // clang-format off
  // The body below is JavaScript, not C++; clang-format would otherwise rewrite `!==` as `!= =`.
  return EM_ASM_INT({
    return (typeof navigator !== 'undefined' && navigator.gpu) ? 1 : 0;
  }) != 0;
  // clang-format on
}

}  // namespace

std::unordered_set<OrtHardwareDevice> DeviceDiscovery::DiscoverDevicesForPlatform() {
  std::unordered_set<OrtHardwareDevice> devices{};

  devices.emplace(GetCpuDeviceFromCPUIDInfo());

  // Report a GPU when the host exposes WebGPU. There is no OS-level device enumeration under Emscripten and a
  // browser will not identify the GPU it is running on, but what matters for EP selection is that a GPU exists
  // and is reachable, which is exactly what navigator.gpu establishes.
  //
  // vendor_id and device_id are left at 0 because they are unknown, not because the device is synthetic: this
  // entry describes real hardware, so it is deliberately not marked with kOrtHardwareDevice_MetadataKey_IsVirtual
  // and is a valid device to back an inference session.
  if (IsWebGpuAvailable()) {
    OrtHardwareDevice gpu_device{};
    gpu_device.type = OrtHardwareDeviceType_GPU;
    gpu_device.vendor_id = 0;
    gpu_device.device_id = 0;
    gpu_device.metadata.Add("Description", "WebGPU");

    devices.emplace(std::move(gpu_device));
  }

  return devices;
}

}  // namespace onnxruntime
