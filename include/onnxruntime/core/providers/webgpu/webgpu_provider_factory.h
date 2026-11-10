// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Dummy file to provide a signal in the ONNX Runtime C cocoapod as to whether the WebGPU EP was included in the build.
// If it was, this file will be included in the cocoapod, and a test like this can be used:
//
//   #if __has_include(<onnxruntime/webgpu_provider_factory.h>)
//     #define WEBGPU_EP_AVAILABLE 1
//   #else
//     #define WEBGPU_EP_AVAILABLE 0
//   #endif

// The WebGPU EP can be enabled via the generic SessionOptionsAppendExecutionProvider method, so no direct usage of
// the provider factory is required.

#ifndef ORT_WEBGPU_PROVIDER_FACTORY_H
#define ORT_WEBGPU_PROVIDER_FACTORY_H

#include <stddef.h>
#include <stdint.h>

#define ORT_DAWN_CACHE_INTERFACE_VERSION 1u
#define ORT_WEBGPU_DAWN_CACHE_INTERFACE_OPTION "dawnCacheInterface"

#ifdef __cplusplus
extern "C" {
#endif

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
