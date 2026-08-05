// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace webgpu {
namespace options {

// The following are the options that can be set in the WebGPU provider options.

constexpr const char* kPreferredLayout = "ep.webgpuexecutionprovider.preferredLayout";
constexpr const char* kEnableGraphCapture = "ep.webgpuexecutionprovider.enableGraphCapture";
// Number of generations of buffers to retain in the per-session pool for reuse
// across captured-graph lifetimes. 0 disables pooling. Default 1 caches one
// generator's worth of intermediate buffers.
constexpr const char* kSessionBufferPoolGenerations = "ep.webgpuexecutionprovider.sessionBufferPoolGenerations";
constexpr const char* kEnableInt64 = "ep.webgpuexecutionprovider.enableInt64";
constexpr const char* kMultiRotaryCacheConcatOffset = "ep.webgpuexecutionprovider.multiRotaryCacheConcatOffset";
constexpr const char* kKvCacheQuantizationBits = "ep.webgpuexecutionprovider.kvCacheQuantizationBits";

constexpr const char* kDawnProcTable = "ep.webgpuexecutionprovider.dawnProcTable";

constexpr const char* kDawnBackendType = "ep.webgpuexecutionprovider.dawnBackendType";
constexpr const char* kPowerPreference = "ep.webgpuexecutionprovider.powerPreference";

constexpr const char* kDeviceId = "ep.webgpuexecutionprovider.deviceId";
constexpr const char* kWebGpuInstance = "ep.webgpuexecutionprovider.webgpuInstance";
constexpr const char* kWebGpuDevice = "ep.webgpuexecutionprovider.webgpuDevice";

constexpr const char* kStorageBufferCacheMode = "ep.webgpuexecutionprovider.storageBufferCacheMode";
constexpr const char* kUniformBufferCacheMode = "ep.webgpuexecutionprovider.uniformBufferCacheMode";
constexpr const char* kQueryResolveBufferCacheMode = "ep.webgpuexecutionprovider.queryResolveBufferCacheMode";
constexpr const char* kDefaultBufferCacheMode = "ep.webgpuexecutionprovider.defaultBufferCacheMode";

constexpr const char* kValidationMode = "ep.webgpuexecutionprovider.validationMode";

constexpr const char* kForceCpuNodeNames = "ep.webgpuexecutionprovider.forceCpuNodeNames";
constexpr const char* kEnablePIXCapture = "ep.webgpuexecutionprovider.enablePIXCapture";

constexpr const char* kPreserveDevice = "ep.webgpuexecutionprovider.preserveDevice";

constexpr const char* kMaxStorageBufferBindingSize = "ep.webgpuexecutionprovider.maxStorageBufferBindingSize";
// Valid range: 1-4096. Larger values are rejected to avoid excessive
// query buffer sizing and unpredictable memory/performance behavior.
constexpr const char* kMaxNumPendingDispatches = "ep.webgpuexecutionprovider.maxNumPendingDispatches";

// The following are the possible values for the provider options.

constexpr const char* kDawnBackendType_D3D12 = "D3D12";
constexpr const char* kDawnBackendType_Vulkan = "Vulkan";

constexpr const char* kPowerPreference_HighPerformance = "high-performance";
constexpr const char* kPowerPreference_LowPower = "low-power";

constexpr const char* kPreferredLayout_NCHW = "NCHW";
constexpr const char* kPreferredLayout_NHWC = "NHWC";

constexpr const char* kEnableGraphCapture_ON = "1";
constexpr const char* kEnableGraphCapture_OFF = "0";

constexpr const char* kEnableInt64_ON = "1";
constexpr const char* kEnableInt64_OFF = "0";

constexpr const char* kEnablePIXCapture_ON = "1";
constexpr const char* kEnablePIXCapture_OFF = "0";

constexpr const char* kPreserveDevice_ON = "1";
constexpr const char* kPreserveDevice_OFF = "0";

// kKvCacheQuantizationBits value is the number of quantization bits as a string.
// "0" disables quantization; "4" enables 4-bit KV cache quantization.
// (Future: "8" for 8-bit.)
constexpr const char* kKvCacheQuantizationBits_OFF = "0";
constexpr const char* kKvCacheQuantizationBits_4Bit = "4";

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
