// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sync_stream.h"

#include <mutex>

#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/webgpu_execution_provider.h"

namespace onnxruntime {
namespace webgpu {
namespace ep {
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
    // Submit before streamless readbacks (e.g., node dumps) use a different recording.
    // Readbacks wait for completion; GPU consumers rely on ordering on the shared queue.
    ORT_THROW_IF_ERROR(context.Flush(ep.BufferManager(), ep.Recording()));
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

}  // namespace ep
}  // namespace webgpu
}  // namespace onnxruntime
