// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace webgpu {
namespace options {

// The following are the options that can be set in the WebGPU provider options.

constexpr const char* kPreferredLayout = "preferredLayout";
constexpr const char* kEnableGraphCapture = "enableGraphCapture";

constexpr const char* kDeviceId = "deviceId";
constexpr const char* kWebGpuInstance = "webgpuInstance";
constexpr const char* kWebGpuAdapter = "webgpuAdapter";
constexpr const char* kWebGpuDevice = "webgpuDevice";

constexpr const char* kStorageBufferCacheMode = "storageBufferCacheMode";
constexpr const char* kUniformBufferCacheMode = "uniformBufferCacheMode";
constexpr const char* kQueryResolveBufferCacheMode = "queryResolveBufferCacheMode";
constexpr const char* kDefaultBufferCacheMode = "defaultBufferCacheMode";

// The following are the possible values for the provider options.

constexpr const char* kPreferredLayout_NCHW = "NCHW";
constexpr const char* kPreferredLayout_NHWC = "NHWC";

constexpr const char* kkEnableGraphCapture_ON = "1";
constexpr const char* kkEnableGraphCapture_OFF = "0";

constexpr const char* kBufferCacheMode_Disabled = "disabled";
constexpr const char* kBufferCacheMode_LazyRelease = "lazyRelease";
constexpr const char* kBufferCacheMode_Simple = "simple";
constexpr const char* kBufferCacheMode_Bucket = "bucket";

}  // namespace options
}  // namespace webgpu
}  // namespace onnxruntime
