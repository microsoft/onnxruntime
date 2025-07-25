// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/rocm/rocm_resource.h>
#include "core/providers/migraphx/migraphx_stream_handle.h"

namespace onnxruntime {

struct MIGraphXNotification : public synchronize::Notification {
  MIGraphXNotification(Stream& s) : Notification(s) {
    HIP_CALL_THROW(hipEventCreateWithFlags(&event_, hipEventDisableTiming));
  }

  ~MIGraphXNotification() {
    if (event_)
      HIP_CALL_THROW(hipEventDestroy(event_));
  }

  void Activate() override {
    // record event with hipEventBlockingSync so we can support sync on host without busy wait.
    HIP_CALL_THROW(hipEventRecord(event_, static_cast<hipStream_t>(GetStream().GetHandle())));
  }

  void wait_on_device(Stream& device_stream) {
    ORT_ENFORCE(device_stream.GetDevice().Type() == OrtDevice::GPU, "Unexpected device:",
                device_stream.GetDevice().ToString());
    // launch a wait command to the migraphx stream
    HIP_CALL_THROW(hipStreamWaitEvent(static_cast<hipStream_t>(device_stream.GetHandle()), event_, 0));
  };

  void wait_on_host() {
    // CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
    HIP_CALL_THROW(hipEventSynchronize(event_));
  }

  hipEvent_t event_;
};

MIGraphXStream::MIGraphXStream(hipStream_t stream,
                               const OrtDevice& device,
                               AllocatorPtr cpu_allocator,
                               bool release_cpu_buffer_on_migraphx_stream)
    : Stream(stream, device),
      cpu_allocator_(cpu_allocator),
      release_cpu_buffer_on_migraphx_stream_(release_cpu_buffer_on_migraphx_stream) {
}

MIGraphXStream::~MIGraphXStream() {
  ORT_IGNORE_RETURN_VALUE(CleanUpOnRunEnd());
  if (own_stream_) {
    auto* handle = GetHandle();
    if (handle)
      HIP_CALL_THROW(hipStreamDestroy(static_cast<hipStream_t>(handle)));
  }
}

std::unique_ptr<synchronize::Notification> MIGraphXStream::CreateNotification(size_t /*num_consumers*/) {
  return std::make_unique<MIGraphXNotification>(*this);
}

void MIGraphXStream::Flush() {
  if (own_stream_)
    HIP_CALL_THROW(hipStreamSynchronize(static_cast<hipStream_t>(GetHandle())));
}

void MIGraphXStream::EnqueDeferredCPUBuffer(void* cpu_buffer) {
  // stream is per thread, so don't need lock
  deferred_cpu_buffers_.push_back(cpu_buffer);
}

struct CpuBuffersInfo {
  // This struct stores the information needed
  // to release CPU buffers allocated for GPU kernels.
  // It's used to enqueue their release after
  // associated GPU kernels in a MIGraphX stream.

  // This is a CPU allocator in MIGraphX EP.
  // It must be the one used to allocate the
  // following pointers.
  AllocatorPtr allocator;
  // buffers[i] is the i-th pointer added by
  // AddDeferredReleaseCPUPtr for a specific
  // MIGraphX stream. For example, this fields
  // should contain all values in
  // deferred_release_buffer_pool_[my_stream]
  // when release my_stream's buffers.
  std::unique_ptr<void*[]> buffers;
  // CPU buffer buffers[i].
  // Number of buffer points in "buffers".
  size_t n_buffers;
};

static void ReleaseCpuBufferCallback(void* raw_info) {
  std::unique_ptr<CpuBuffersInfo> info = std::make_unique<CpuBuffersInfo>();
  info.reset(reinterpret_cast<CpuBuffersInfo*>(raw_info));
  for (size_t i = 0; i < info->n_buffers; ++i) {
    info->allocator->Free(info->buffers[i]);
  }
}

Status MIGraphXStream::CleanUpOnRunEnd() {
  if (deferred_cpu_buffers_.empty())
    return Status::OK();
  // Release the ownership of cpu_buffers_info so that the underlying
  // object will keep alive until the end of ReleaseCpuBufferCallback.
  if (release_cpu_buffer_on_migraphx_stream_ && cpu_allocator_->Info().alloc_type == OrtArenaAllocator) {
    std::unique_ptr<CpuBuffersInfo> cpu_buffers_info = std::make_unique<CpuBuffersInfo>();
    cpu_buffers_info->allocator = cpu_allocator_;
    cpu_buffers_info->buffers = std::make_unique<void*[]>(deferred_cpu_buffers_.size());
    for (size_t i = 0; i < deferred_cpu_buffers_.size(); ++i) {
      cpu_buffers_info->buffers[i] = deferred_cpu_buffers_.at(i);
    }
    cpu_buffers_info->n_buffers = deferred_cpu_buffers_.size();
    HIP_RETURN_IF_ERROR(hipLaunchHostFunc(static_cast<hipStream_t>(GetHandle()), ReleaseCpuBufferCallback, cpu_buffers_info.release()));
  } else {
    HIP_RETURN_IF_ERROR(hipStreamSynchronize(static_cast<hipStream_t>(GetHandle())));
    for (auto* buffer : deferred_cpu_buffers_) {
      cpu_allocator_->Free(buffer);
    }
  }

  deferred_cpu_buffers_.clear();
  return Status::OK();
}

void* MIGraphXStream::GetResource(int version, int id) const {
  ORT_ENFORCE(version <= ORT_ROCM_RESOURCE_VERSION, "resource version unsupported!");
  void* resource{};
  switch (id) {
    case RocmResource::hip_stream_t:
      return reinterpret_cast<void*>(GetHandle());
    default:
      break;
  }
  return resource;
}

// CPU Stream command handles
void WaitMIGraphXNotificationOnDevice(Stream* stream, synchronize::Notification& notification) {
  static_cast<MIGraphXNotification*>(&notification)->wait_on_device(*stream);
}

void WaitMIGraphXNotificationOnHost(Stream* /*stream*/, synchronize::Notification& notification) {
  static_cast<MIGraphXNotification*>(&notification)->wait_on_host();
}

void RegisterMIGraphXStreamHandles(IStreamCommandHandleRegistry& stream_handle_registry,
                                   const OrtDevice::DeviceType device_type,
                                   AllocatorPtr cpu_allocator,
                                   bool release_cpu_buffer_on_migraphx_stream,
                                   hipStream_t external_stream,
                                   bool use_existing_stream) {
  // wait migraphx notification on migraphx ep
  stream_handle_registry.RegisterWaitFn(device_type, device_type, WaitMIGraphXNotificationOnDevice);
  // wait migraphx notification on cpu ep
  stream_handle_registry.RegisterWaitFn(device_type, OrtDevice::CPU, WaitMIGraphXNotificationOnHost);
  if (!use_existing_stream)
    stream_handle_registry.RegisterCreateStreamFn(device_type, [cpu_allocator, release_cpu_buffer_on_migraphx_stream](const OrtDevice& device) {
      HIP_CALL_THROW(hipSetDevice(device.Id()));
      hipStream_t stream = nullptr;
      HIP_CALL_THROW(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
      return std::make_unique<MIGraphXStream>(stream, device, cpu_allocator, release_cpu_buffer_on_migraphx_stream);
    });
  else
    stream_handle_registry.RegisterCreateStreamFn(device_type, [cpu_allocator,
                                                                release_cpu_buffer_on_migraphx_stream,
                                                                external_stream](const OrtDevice& device) {
      return std::make_unique<MIGraphXStream>(external_stream, device, cpu_allocator, release_cpu_buffer_on_migraphx_stream);
    });
}

}  // namespace onnxruntime
