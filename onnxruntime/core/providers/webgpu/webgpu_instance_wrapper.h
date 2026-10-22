// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/webgpu/webgpu_external_header.h"

#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
#include "core/providers/webgpu/webgpu_context_dawn_platform.h"
#endif

namespace onnxruntime::webgpu {

// Owns a WebGPU instance and, for bundled native Dawn, its custom platform.
// Dependent WebGPU objects must be destroyed before this owner.
class WebGpuInstanceWrapper final {
 public:
  explicit WebGpuInstanceWrapper(const wgpu::InstanceDescriptor& instance_desc);
  ~WebGpuInstanceWrapper();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(WebGpuInstanceWrapper);

  WGPUInstance Get() const { return instance_.Get(); }

 private:
#if !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)
  // Declared first so the platform outlives instance teardown.
  BundledDawnPlatformPtr platform_;
#endif
  wgpu::Instance instance_;
};

}  // namespace onnxruntime::webgpu
