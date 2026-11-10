// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/webgpu_instance_wrapper.h"

namespace onnxruntime::webgpu {

WebGpuInstanceWrapper::WebGpuInstanceWrapper(const wgpu::InstanceDescriptor& instance_desc) {
#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
  platform_ = CreateBundledDawnPlatform();
  instance_ = CreateBundledDawnInstance(instance_desc, *platform_);
#else
  instance_ = wgpu::CreateInstance(&instance_desc);
#endif
  ORT_ENFORCE(instance_.Get() != nullptr, "Failed to create wgpu::Instance.");
}

WebGpuInstanceWrapper::~WebGpuInstanceWrapper() = default;

}  // namespace onnxruntime::webgpu
