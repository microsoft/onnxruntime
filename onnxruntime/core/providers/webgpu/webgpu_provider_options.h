// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace webgpu {
namespace options {

// The following are the options that can be set in the WebGPU provider options.

constexpr const char* kPreferredLayout = "WebGPU:preferredLayout";
constexpr const char* kEnableGraphCapture = "WebGPU:enableGraphCapture";

constexpr const char* kDeviceId = "WebGPU:deviceId";
constexpr const char* kWebGpuInstance = "WebGPU:webgpuInstance";
constexpr const char* kWebGpuAdapter = "WebGPU:webgpuAdapter";
constexpr const char* kWebGpuDevice = "WebGPU:webgpuDevice";

constexpr const char* kStorageBufferCacheMode = "WebGPU:storageBufferCacheMode";
constexpr const char* kUniformBufferCacheMode = "WebGPU:uniformBufferCacheMode";
constexpr const char* kQueryResolveBufferCacheMode = "WebGPU:queryResolveBufferCacheMode";
constexpr const char* kDefaultBufferCacheMode = "WebGPU:defaultBufferCacheMode";

constexpr const char* kValidationMode = "WebGPU:validationMode";

constexpr const char* kForceCpuNodeNames = "WebGPU:forceCpuNodeNames";

// The following are the possible values for the provider options.

constexpr const char* kPreferredLayout_NCHW = "NCHW";
constexpr const char* kPreferredLayout_NHWC = "NHWC";

constexpr const char* kEnableGraphCapture_ON = "1";
constexpr const char* kEnableGraphCapture_OFF = "0";

constexpr const char* kBufferCacheMode_Disabled = "disabled";
constexpr const char* kBufferCacheMode_LazyRelease = "lazyRelease";
constexpr const char* kBufferCacheMode_Simple = "simple";
constexpr const char* kBufferCacheMode_Bucket = "bucket";

constexpr const char* kValidationMode_Disabled = "disabled";
constexpr const char* kValidationMode_wgpuOnly = "wgpuOnly";
constexpr const char* kValidationMode_basic = "basic";
constexpr const char* kValidationMode_full = "full";

}  // namespace options
}  // namespace webgpu
}  // namespace onnxruntime
