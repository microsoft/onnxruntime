// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"

namespace onnxruntime {
namespace webgpu {

#if defined(ORT_USE_EP_API_ADAPTERS)
namespace {

struct WebGpuSyncStream final : OrtSyncStreamImpl {
  explicit WebGpuSyncStream(WebGpuExecutionProvider& ep) : ep_{ep} {
    ort_version_supported = ORT_API_VERSION;
    Release = [](OrtSyncStreamImpl* stream) noexcept { delete static_cast<WebGpuSyncStream*>(stream); };
    GetHandle = [](OrtSyncStreamImpl* stream) noexcept -> void* { return stream; };
    CreateNotification = CreateNotificationImpl;
    Flush = FlushImpl;
    OnSessionRunEnd = [](OrtSyncStreamImpl*) noexcept -> OrtStatus* { return nullptr; };
  }

  static const WebGpuSyncStream& From(const OrtSyncStream* stream) {
    const auto* impl = onnxruntime::ep::Api().ep.SyncStream_GetImpl(stream);
    ORT_ENFORCE(impl != nullptr && impl->Flush == FlushImpl, "Expected a WebGPU sync stream.");
    return *static_cast<const WebGpuSyncStream*>(impl);
  }

  static OrtStatus* ORT_API_CALL FlushImpl(OrtSyncStreamImpl* stream) noexcept {
    EXCEPTION_TO_RETURNED_STATUS_BEGIN
    auto& ep = static_cast<WebGpuSyncStream*>(stream)->ep_;
    auto& context = WebGpuContextFactory::GetContext(ep.GetDeviceId());
    std::lock_guard<std::recursive_mutex> lock{ep.Recording().mutex};
    ORT_THROW_IF_ERROR(context.Flush(ep.BufferManager(), ep.Recording()));
    wgpu::QueueWorkDoneStatus completion = wgpu::QueueWorkDoneStatus::Error;
    auto future = context.Device().GetQueue().OnSubmittedWorkDone(
        wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::QueueWorkDoneStatus status, wgpu::StringView, wgpu::QueueWorkDoneStatus* result) noexcept {
          *result = status;
        },
        &completion);
    ORT_THROW_IF_ERROR(context.Wait(future));
    ORT_ENFORCE(completion == wgpu::QueueWorkDoneStatus::Success, "WebGPU queue completion failed.");
    return nullptr;
    EXCEPTION_TO_RETURNED_STATUS_END
  }

  static OrtStatus* ORT_API_CALL CreateNotificationImpl(
      OrtSyncStreamImpl* stream, OrtSyncNotificationImpl** notification) noexcept;

  WebGpuExecutionProvider& ep_;
};

struct WebGpuSyncNotification final : OrtSyncNotificationImpl {
  explicit WebGpuSyncNotification(WebGpuSyncStream& stream) : stream_{stream} {
    ort_version_supported = ORT_API_VERSION;
    Release = [](OrtSyncNotificationImpl* notification) noexcept {
      delete static_cast<WebGpuSyncNotification*>(notification);
    };
    Activate = [](OrtSyncNotificationImpl* notification) noexcept -> OrtStatus* {
      auto& self = *static_cast<WebGpuSyncNotification*>(notification);
      return WebGpuSyncStream::FlushImpl(&self.stream_);
    };
    WaitOnDevice = [](OrtSyncNotificationImpl*, OrtSyncStream*) noexcept -> OrtStatus* { return nullptr; };
    WaitOnHost = [](OrtSyncNotificationImpl*) noexcept -> OrtStatus* { return nullptr; };
  }

  WebGpuSyncStream& stream_;
};

OrtStatus* ORT_API_CALL WebGpuSyncStream::CreateNotificationImpl(
    OrtSyncStreamImpl* stream, OrtSyncNotificationImpl** notification) noexcept {
  EXCEPTION_TO_RETURNED_STATUS_BEGIN
  *notification = new WebGpuSyncNotification(*static_cast<WebGpuSyncStream*>(stream));
  return nullptr;
  EXCEPTION_TO_RETURNED_STATUS_END
}

}  // namespace

OrtSyncStreamImpl* CreateWebGpuSyncStream(WebGpuExecutionProvider& ep) {
  return new WebGpuSyncStream(ep);
}

CommandRecordingState& GetWebGpuStreamCommandState(const OrtSyncStream* stream) {
  return WebGpuSyncStream::From(stream).ep_.Recording();
}

common::Status CopyTensorOnWebGpuStream(const OrtSyncStream* stream, const void* src_data,
                                      bool src_is_gpu, void* dst_data, bool dst_is_gpu, size_t bytes) {
  auto& ep = WebGpuSyncStream::From(stream).ep_;
  DataTransferImpl transfer(ep.BufferManager(), ep.Recording());
  return transfer.CopyTensor(src_data, src_is_gpu, dst_data, dst_is_gpu, bytes);
}
#endif

common::Status DataTransferImpl::CopyTensor(void const* src_data,
                                            bool src_is_gpu,
                                            void* dst_data,
                                            bool dst_is_gpu,
                                            size_t bytes) const {
  auto& command_state = recording_;
  std::lock_guard<std::mutex> lock{mutex_};
  std::lock_guard<std::recursive_mutex> recording_lock{command_state.mutex};
  if (bytes > 0) {
    if (dst_is_gpu) {
      if (src_is_gpu) {
        // copy from GPU to GPU
        buffer_manager_.MemCpy(command_state,
                               static_cast<WGPUBuffer>(const_cast<void*>(src_data)),
                               static_cast<WGPUBuffer>(dst_data),
                               bytes);
      } else {
        // copy from CPU to GPU
        buffer_manager_.Upload(command_state,
                               const_cast<void*>(src_data),
                               static_cast<WGPUBuffer>(dst_data),
                               bytes);
      }
    } else {
      // copy from GPU to CPU
      buffer_manager_.Download(command_state,
                               static_cast<WGPUBuffer>(const_cast<void*>(src_data)),
                               dst_data,
                               bytes);
    }
  }

  return Status::OK();
}

bool DataTransfer::IsSupportedDevicePair(const OrtDevice& src_device, const OrtDevice& dst_device) {
  // WebGPU allocations carry VendorIds::NONE. A vendor-tagged GPU handle belongs to another EP, and
  // reinterpreting it as a WGPUBuffer would be unsafe. The plugin EP transfer applies the same rule.
  if (src_device.Type() == OrtDevice::GPU && src_device.Vendor() != OrtDevice::VendorIds::NONE) {
    return false;
  }
  if (dst_device.Type() == OrtDevice::GPU && dst_device.Vendor() != OrtDevice::VendorIds::NONE) {
    return false;
  }

  return (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::CPU) ||
         (dst_device.Type() == OrtDevice::GPU && src_device.Type() == OrtDevice::GPU) ||
         (dst_device.Type() == OrtDevice::CPU && src_device.Type() == OrtDevice::GPU);
}

bool DataTransfer::CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const {
  return IsSupportedDevicePair(src_device, dst_device);
}

common::Status DataTransfer::CopyTensor(const Tensor& src, Tensor& dst) const {
  return impl_.CopyTensor(src.DataRaw(),
                          src.Location().device.Type() == OrtDevice::GPU,
                          dst.MutableDataRaw(),
                          dst.Location().device.Type() == OrtDevice::GPU,
                          src.SizeInBytes());
}

}  // namespace webgpu
}  // namespace onnxruntime
