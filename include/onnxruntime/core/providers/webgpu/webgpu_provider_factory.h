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

#pragma once

#include <stddef.h>
#include <stdint.h>

// Define export macros without including onnxruntime_c_api.h to avoid conflicts
#if defined(_WIN32)
  #ifdef ORT_DLL_EXPORT
    #define ORT_WEBGPU_EXPORT __declspec(dllexport)
  #else
    #define ORT_WEBGPU_EXPORT __declspec(dllimport)
  #endif
  #define ORT_WEBGPU_API_CALL __stdcall
#elif defined(__APPLE__) || defined(__linux__)
  #define ORT_WEBGPU_EXPORT __attribute__((visibility("default")))
  #define ORT_WEBGPU_API_CALL
#else
  #define ORT_WEBGPU_EXPORT
  #define ORT_WEBGPU_API_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * \brief Get the Dawn proc table from WebGPU EP context
 * \param context_id The WebGPU context ID (0 for default context)
 * \return Pointer to the Dawn proc table, or nullptr if not available
 */
ORT_WEBGPU_EXPORT const void* ORT_WEBGPU_API_CALL OrtWebGpuGetDawnProcTable(int context_id);

/**
 * \brief Get the WebGPU instance from WebGPU EP context
 * \param context_id The WebGPU context ID (0 for default context)
 * \return Pointer to the WebGPU instance (WGPUInstance), or nullptr if not available
 */
ORT_WEBGPU_EXPORT void* ORT_WEBGPU_API_CALL OrtWebGpuGetInstance(int context_id);

/**
 * \brief Get the WebGPU device from WebGPU EP context
 * \param context_id The WebGPU context ID (0 for default context)
 * \return Pointer to the WebGPU device (WGPUDevice), or nullptr if not available
 */
ORT_WEBGPU_EXPORT void* ORT_WEBGPU_API_CALL OrtWebGpuGetDevice(int context_id);

#ifdef __cplusplus
}
#endif
