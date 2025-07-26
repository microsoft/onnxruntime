// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <atomic>
#include <functional>
#include <optional>
#include <unordered_map>

#include "core/framework/allocator.h"
#include "core/framework/ortdevice.h"
#include "core/common/status.h"

namespace onnxruntime {
class IExecutionProvider;
// this opaque handle could be anything the target device generated.
// it could be a cuda event, or a npu notification implementation
using NotificationHandle = void*;
// it can be either a cuda stream, or even nullptr for device doesn't have stream support like cpu.
using StreamHandle = void*;

namespace synchronize {
class Notification;
}

/// <summary>
/// Class to represent a stream on the OrtDevice.
/// </summary>
class Stream {
 public:
  Stream(StreamHandle h, const OrtDevice& d)
      : handle_(h), device_(d) {
  }

  virtual ~Stream() = default;

  virtual std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) {
    return {};
  };

  // block the host thread until all the tasks in the stream finished.
  virtual void Flush() {};

  // The framework may reuse the stream instance for multiple iterations.
  // This is the API that provide a chance to let the device stream cleanup
  // resource at the end of a iteration.
  virtual Status CleanUpOnRunEnd() { return Status::OK(); };

  // Get the native stream handle. nullptr if the OrtDevice doesn't support streams.
  StreamHandle GetHandle() const { return handle_; }

  const OrtDevice& GetDevice() const { return device_; }

  // Get the current synchronization ID.
  // The value is 0 until this stream records an event.
  // The sync id is incremented before each new event that is recorded in our stream via Notification::Activate.
  uint64_t GetSyncId() const { return sync_id_; }

  // Return the sync id from when the last synchronization happened between producer_stream and this stream.
  // i.e. the id for the event that the producer stream recorded and we waited on
  //
  // Returns 0 if the streams have not previously been synchronized.
  uint64_t GetSyncIdForLastWaitOnStream(const Stream& producer_stream) const {
    auto it = producer_stream_sync_info_.find(&producer_stream);
    return it == producer_stream_sync_info_.end() ? 0 : it->second;
  }

  // Get the sync information that is added to a notification when it is activated.
  // This contains sync ids for all streams we have waited on, and the new sync id for our stream.
  std::unordered_map<const Stream*, uint64_t> OnNotificationActivation() {
    // copy our sync info so the notification can pass it on to any waiting streams
    auto sync_info = producer_stream_sync_info_;
    // and add our info to the copy, incrementing the sync_id
    sync_info[this] = ++sync_id_;

    return sync_info;
  }

  // Record information from a Notification we waited on.
  //   - copies the producer stream info from the notification.
  void UpdateWithAwaitedNotification(const synchronize::Notification& notification);

  // used in custom ops. doesn't really belong here.
  virtual void* GetResource(int /*version*/, int /*id*/) const {
    return nullptr;
  }

 private:
  StreamHandle handle_;
  const OrtDevice& device_;

  // current sync id. equivalent to a counter for the number of events we have recorded in the underlying stream.
  // 0 == no events recorded. sync_id_ is updated prior to recording a new event.
  std::atomic<uint64_t> sync_id_{0};

  // This is a map to track synchronization points between streams. When we wait on another stream (the producer)
  // we add an entry to the map for that stream.
  // The entry has the sync id from the producer stream for the event we waited on.
  //
  // TODO: use inline container.
  // currently this class is header only, but abseil doesn't compile with nvcc
  // we need to add new symbol to provider_bridge and hide abseil from the header.
  std::unordered_map<const Stream*, uint64_t> producer_stream_sync_info_{};

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Stream);
};

namespace synchronize {
// An abstraction used for synchronization between streams.
// See derived classes (CudaNotification, etc.) for implementation examples.
class Notification {
 public:
  explicit Notification(Stream& s) : stream_(s) {}
  virtual ~Notification() = default;

  // Activate the notification. This records an event in the Stream that created the Notification that other streams can wait on.
  void ActivateAndUpdate() {
    Activate();

    // copy the sync info. this includes an entry for stream_ with the next sync id.
    stream_sync_info_ = stream_.OnNotificationActivation();
  }

  // Get the sync history for the producer stream that created this Notification.
  // The notification must have be activated previously.
  const std::unordered_map<const Stream*, uint64_t>& GetStreamSyncInfo() const {
    return stream_sync_info_;
  }

 protected:
  virtual void Activate() = 0;

  Stream& GetStream() {
    return stream_;
  }

 private:
  // Stream that created the notification (producer stream).
  Stream& stream_;

  // This is a snapshot of the sync history for the stream that created the Notification.
  std::unordered_map<const Stream*, uint64_t> stream_sync_info_{};
};
}  // namespace synchronize

// the definition for the handle for stream commands
// EP can register the handle to the executor.
// in the POC, just use primitive function pointer
// TODO: use a better way to dispatch handles.
using CreateStreamFn = std::function<std::unique_ptr<Stream>(const OrtDevice&)>;

// This SetDevice function is used by TRT EP or CUDA EP to handle the case where ExecutionMode::ORT_PARALLEL is enabled.
// In that case, ORT retrieves a thread from the thread pool to run kernels for a given session.
// Since new threads default to using device 0, but the session may be tightly bound to a device > 0,
// This SetDevice function will be called in RunSince to ensure running kernels on a correct GPU device.
using SetDeviceFn = std::function<void(OrtDevice::DeviceId)>;

// an interface of a simple registry which hold the handles EP registered.
// make it interface so we can pass it through shared library based execution providers
class IStreamCommandHandleRegistry {
 public:
  virtual ~IStreamCommandHandleRegistry() = default;
  // Wait is a little special as we need to consider the source stream the notification generated, and the stream we are waiting.
  // i.e., for an cuda event what notify the memory copy, it could be wait on a CPU stream, or on another cuda stream.
  [[nodiscard]] virtual WaitNotificationFn GetWaitHandle(const OrtDevice& notification_owner_device,
                                                         const OrtDevice& executor_device) const = 0;

  // Get the stream creation function registered for the given device type.
  [[nodiscard]] virtual CreateStreamFn GetCreateStreamFn(OrtDevice::DeviceType execution_device_type) const = 0;

  // register a wait method which will be invoked to await a notification that is
  // created by 'notification_device_type' device on a stream at 'device_type' device.
  virtual void RegisterWaitFn(OrtDevice::DeviceType notification_device_type,
                              OrtDevice::DeviceType device_type,
                              WaitNotificationFn fn) = 0;

  // register a handle about how to create stream on given device type.
  virtual void RegisterCreateStreamFn(OrtDevice::DeviceType device_type, CreateStreamFn f) = 0;

  // Register a SetDevice function.
  // This interface is currently used by TRT EP or CUDA EP only.
  virtual void RegisterSetDeviceFn(OrtDevice::DeviceType device_type, SetDeviceFn f) {
    ORT_UNUSED_PARAMETER(device_type);
    ORT_UNUSED_PARAMETER(f);
  };

  // Get a SetDevice function.
  // This interface is currently used by TRT EP or CUDA EP only and is called in RunSince from stream execution.
  virtual std::optional<SetDeviceFn> GetSetDeviceFn(OrtDevice::DeviceType device_type) const {
    ORT_UNUSED_PARAMETER(device_type);
    return std::nullopt;
  };
};

}  // namespace onnxruntime
