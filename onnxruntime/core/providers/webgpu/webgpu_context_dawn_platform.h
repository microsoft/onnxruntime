// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/webgpu_external_header.h"

struct DawnProcTable;

namespace onnxruntime::webgpu {

const DawnProcTable& GetBundledDawnProcs();
wgpu::Instance CreateBundledDawnInstance(wgpu::InstanceDescriptor instance_desc);
// Call after all instances created by CreateBundledDawnInstance() and their dependent objects have been destroyed.
// Must not run concurrently with CreateBundledDawnInstance().
void CleanupBundledDawnPlatform();
InlinedVector<wgpu::Adapter> EnumerateBundledDawnAdapters(WGPUInstance instance,
                                                          const wgpu::RequestAdapterOptions& options);

}  // namespace onnxruntime::webgpu
