// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

#include "core/common/inlined_containers.h"
#include "core/providers/webgpu/webgpu_external_header.h"

namespace dawn::platform {
class Platform;
}  // namespace dawn::platform

struct DawnProcTable;

namespace onnxruntime::webgpu {

// Defined out of line so unique_ptr can destroy the platform without callers needing
// the complete Platform type, as std::default_delete would require.
struct BundledDawnPlatformDeleter {
  void operator()(dawn::platform::Platform* platform) const;
};
using BundledDawnPlatformPtr = std::unique_ptr<dawn::platform::Platform, BundledDawnPlatformDeleter>;

BundledDawnPlatformPtr CreateBundledDawnPlatform();

// Creates a bundled Dawn instance using the caller-owned platform.
// Keep the platform alive until the instance and its dependent objects have been destroyed.
wgpu::Instance CreateBundledDawnInstance(wgpu::InstanceDescriptor instance_desc, dawn::platform::Platform& platform);

const DawnProcTable& GetBundledDawnProcs();

InlinedVector<wgpu::Adapter> EnumerateBundledDawnAdapters(WGPUInstance instance,
                                                          const wgpu::RequestAdapterOptions& options);

}  // namespace onnxruntime::webgpu
