// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/stream_handles.h"
#include "amdgpu_inc.h"
#include "amdgpu_call.h"

namespace onnxruntime {
void WaitAMDGPUNotificationOnDevice(Stream& stream, synchronize::Notification& notification);

struct AMDGPUStream : Stream {
  AMDGPUStream(hipStream_t stream,
                 const OrtDevice& device,
                 AllocatorPtr cpu_allocator,
                 bool release_cpu_buffer_on_amdgpu_stream);

  ~AMDGPUStream();

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) override;

  void Flush() override;

  Status CleanUpOnRunEnd() override;

  void EnqueDeferredCPUBuffer(void* cpu_buffer);

  bool own_stream_{true};

  virtual void* GetResource(int version, int id) const;

  virtual WaitNotificationFn GetWaitNotificationFn() const { return WaitAMDGPUNotificationOnDevice; }

 private:
  std::vector<void*> deferred_cpu_buffers_;
  AllocatorPtr cpu_allocator_;
  bool release_cpu_buffer_on_amdgpu_stream_{true};
};

void RegisterAMDGPUStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                                   const OrtDevice::DeviceType device_type,
                                   AllocatorPtr cpu_allocator,
                                   bool release_cpu_buffer_on_amdgpu_stream,
                                   hipStream_t external_stream,
                                   bool use_existing_stream);
}  // namespace onnxruntime
