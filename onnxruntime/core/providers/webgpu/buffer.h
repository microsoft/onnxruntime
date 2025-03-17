// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/webgpu/webgpu_external_header.h"

namespace onnxruntime {
namespace webgpu {

using WebGpuBuffer = WGPUBuffer;

inline wgpu::BufferUsage GetBufferUsage(WebGpuBuffer buffer) {
  return static_cast<wgpu::BufferUsage>(wgpuBufferGetUsage(buffer));
}

inline size_t GetBufferSize(WebGpuBuffer buffer) {
  return static_cast<size_t>(wgpuBufferGetSize(buffer));
}

}  // namespace webgpu
}  // namespace onnxruntime
