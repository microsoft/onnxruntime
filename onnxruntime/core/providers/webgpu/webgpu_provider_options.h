// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace webgpu {
namespace options {

// The following are the options that can be set in the WebGPU provider options.

constexpr const char* kPreferredLayout = "ep.WebGpuExecutionProvider.preferredLayout";
constexpr const char* kEnableGraphCapture = "ep.WebGpuExecutionProvider.enableGraphCapture";

constexpr const char* kDawnProcTable = "ep.WebGpuExecutionProvider.dawnProcTable";

constexpr const char* kDawnBackendType = "ep.WebGpuExecutionProvider.dawnBackendType";

constexpr const char* kDeviceId = "ep.WebGpuExecutionProvider.deviceId";
constexpr const char* kWebGpuInstance = "ep.WebGpuExecutionProvider.webgpuInstance";
constexpr const char* kWebGpuDevice = "ep.WebGpuExecutionProvider.webgpuDevice";

constexpr const char* kStorageBufferCacheMode = "ep.WebGpuExecutionProvider.storageBufferCacheMode";
constexpr const char* kUniformBufferCacheMode = "ep.WebGpuExecutionProvider.uniformBufferCacheMode";
constexpr const char* kQueryResolveBufferCacheMode = "ep.WebGpuExecutionProvider.queryResolveBufferCacheMode";
constexpr const char* kDefaultBufferCacheMode = "ep.WebGpuExecutionProvider.defaultBufferCacheMode";

constexpr const char* kValidationMode = "ep.WebGpuExecutionProvider.validationMode";

constexpr const char* kForceCpuNodeNames = "ep.WebGpuExecutionProvider.forceCpuNodeNames";
constexpr const char* kEnablePIXCapture = "ep.WebGpuExecutionProvider.enablePIXCapture";

constexpr const char* kPreserveDevice = "ep.WebGpuExecutionProvider.preserveDevice";

// The following are the possible values for the provider options.

constexpr const char* kDawnBackendType_D3D12 = "D3D12";
constexpr const char* kDawnBackendType_Vulkan = "Vulkan";

constexpr const char* kPreferredLayout_NCHW = "NCHW";
constexpr const char* kPreferredLayout_NHWC = "NHWC";

constexpr const char* kEnableGraphCapture_ON = "1";
constexpr const char* kEnableGraphCapture_OFF = "0";

constexpr const char* kEnablePIXCapture_ON = "1";
constexpr const char* kEnablePIXCapture_OFF = "0";

constexpr const char* kPreserveDevice_ON = "1";
constexpr const char* kPreserveDevice_OFF = "0";

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
