// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/stream_handles.h"

namespace onnxruntime {
namespace vulkan {

// allocator for CPU scratch buffer memory. TBD what is required
// struct DeferredCpuAllocator : public OrtAllocator {
//  DeferredCpuAllocator(Stream&);
//  Stream& stream_;
//};

struct Stream : onnxruntime::Stream {
  Stream(void* stream,  // ??? TBD what we need to pass through here. VkCompute? and/or VulkanDevice? other?
         const OrtDevice& device);

  ~Stream();

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) override;

  void Flush() override;

  Status CleanUpOnRunEnd() override;

  void* GetResource(int version, int id) const override;

  static void WaitNotificationOnDevice(onnxruntime::Stream& stream, synchronize::Notification& notification);

  WaitNotificationFn GetWaitNotificationFn() const override {
    return WaitNotificationOnDevice;
  }

 private:
  void* stream_;
  // TBD if we need something to manage cleanup of scratch buffers
  // std::vector<void*> deferred_cpu_buffers_;
  // DeferredCpuAllocator deferred_cpu_allocator_;
};

void RegisterStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                           const OrtDevice::DeviceType device_type,
                           AllocatorPtr cpu_allocator,
                           void* stream);

}  // namespace vulkan
}  // namespace onnxruntime
