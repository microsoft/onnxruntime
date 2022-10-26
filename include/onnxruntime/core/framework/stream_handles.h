// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <functional>
#include <unordered_map>
#include "core/framework/ortdevice.h"
#include "core/common/status.h"

namespace onnxruntime {
class IExecutionProvider;
// this opaque handle could be anything the target device generated.
// it could be a cuda event, or a cpu notification implementation
using NotificationHandle = void*;
// it can be either a cuda stream, or even nullptr for device doesn't have stream support like cpu.
using StreamHandle = void*;

// a stream abstraction which hold an opaque handle, and a reference to which OrtDevice instance this stream belong to.
// it need to be OrtDevice instance as we might have different stream on different OrtDevice with same type.
// i.e. different cuda stream on different GPU.
namespace synchronize {
struct Notification;
}

class Stream {
 public:
  Stream(StreamHandle h, const OrtDevice& d) : handle_(h), device_(d) {}

  virtual ~Stream() {}
  virtual std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) {
    return {};
  };
  virtual void Flush(){};
  virtual Status CleanUpOnRunEnd() { return Status::OK(); };

  StreamHandle GetHandle() const { return handle_; }

  const OrtDevice& GetDevice() const { return device_; }

  uint64_t GetCurrentTimestamp() const { return timestamp_; }

  uint64_t GetLastSyncTimestampWithTargetStream(Stream* target_stream) const {
    auto it = other_stream_clock_.find(target_stream);
    return it == other_stream_clock_.end() ? 0 : it->second;
  }

  void CloneCurrentStreamSyncTable(std::unordered_map<Stream*, uint64_t>& output) const {
    std::copy(other_stream_clock_.begin(), other_stream_clock_.end(), std::inserter(output, output.end()));
  }

  uint64_t BumpTimeStampAndReturn() {
    return ++timestamp_;
  }

  void UpdateStreamClock(const std::unordered_map<Stream*, uint64_t>& clock) {
    for (auto& kv : clock) {
      auto it = other_stream_clock_.find(kv.first);
      if (it == other_stream_clock_.end())
        other_stream_clock_.insert(kv);
      else
        other_stream_clock_[kv.first] = std::max(it->second, kv.second);
    }
  }

 protected:
  StreamHandle handle_;
  const OrtDevice& device_;
  uint64_t timestamp_{0};
  // TODO: use inline container.
  // currently this class is header only, but abseil doesn't compile with nvcc
  // we need to add new symbol to provider_bridge and hide abseil from the header.
  std::unordered_map<Stream*, uint64_t> other_stream_clock_;
};

namespace synchronize {
class Notification {
 public:
  Notification(Stream* s) : stream_(s) {}
  virtual ~Notification() {}

  void ActivateAndUpdate() {
    Activate();
    stream_->CloneCurrentStreamSyncTable(stream_clock_);
    stream_clock_[stream_] = stream_->BumpTimeStampAndReturn();
  }

  const std::unordered_map<Stream*, uint64_t>& GetStreamSyncTable() {
    return stream_clock_;
  }

 protected:
  virtual void Activate() = 0;
  // which stream create this notificaiton.
  Stream* stream_;
  // TODO: use inline container.
  // currently this class is header only, but abseil doesn't compile with nvcc
  // we need to add new symbol to provider_bridge and hide abseil from the header.
  std::unordered_map<Stream*, uint64_t> stream_clock_;
};
}  // namespace synchronize

// the definition for the handle for stream commands
// EP can register the handle to the executor.
// in the POC, just use primitive function pointer
// TODO: use a better way to dispatch handles.
using WaitNotificationFn = std::function<void(Stream&, synchronize::Notification&)>;
using CreateStreamFn = std::function<std::unique_ptr<Stream>(const OrtDevice&)>;

// an interface of a simple registry which hold the handles EP registered.
// make it interface so we can pass it through shared library based execution providers
class IStreamCommandHandleRegistry {
 public:
  virtual ~IStreamCommandHandleRegistry() {}
  // Wait is a little special as we need to consider the source stream the notification generated, and the stream we are waiting.
  // i.e., for an cuda event what notify the memory copy, it could be wait on a CPU stream, or on another cuda stream.
  virtual WaitNotificationFn GetWaitHandle(const OrtDevice::DeviceType notification_ower_device_type, const OrtDevice::DeviceType executor_device_type) const = 0;

  virtual CreateStreamFn GetCreateStreamFn(const OrtDevice::DeviceType execution_device_type) const = 0;

  virtual void RegisterWaitFn(const OrtDevice::DeviceType notification_device_type, const OrtDevice::DeviceType device_type, WaitNotificationFn fn) = 0;

  virtual void RegisterCreateStreamFn(const OrtDevice::DeviceType device_type, CreateStreamFn f) = 0;
};

}  // namespace onnxruntime
