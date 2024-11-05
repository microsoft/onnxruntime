// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/framework/stream_handles.h"
#include "migraphx_inc.h"
#include "migraphx_call.h"

namespace onnxruntime {
void WaitMIGraphXNotificationOnDevice(Stream& stream, synchronize::Notification& notification);

struct MIGraphXStream : Stream {
  MIGraphXStream(hipStream_t stream,
                 const OrtDevice& device,
                 AllocatorPtr cpu_allocator,
                 bool release_cpu_buffer_on_migraphx_stream);

  ~MIGraphXStream();

  std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) override;

  void Flush() override;

  Status CleanUpOnRunEnd() override;

  void EnqueDeferredCPUBuffer(void* cpu_buffer);

  bool own_stream_{true};

  virtual void* GetResource(int version, int id) const;

  virtual WaitNotificationFn GetWaitNotificationFn() const { return WaitMIGraphXNotificationOnDevice; }

 private:
  std::vector<void*> deferred_cpu_buffers_;
  AllocatorPtr cpu_allocator_;
  bool release_cpu_buffer_on_migraphx_stream_{true};
};

void RegisterMIGraphXStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                                   const OrtDevice::DeviceType device_type,
                                   AllocatorPtr cpu_allocator,
                                   bool release_cpu_buffer_on_migraphx_stream,
                                   hipStream_t external_stream,
                                   bool use_existing_stream);
}  // namespace onnxruntime
