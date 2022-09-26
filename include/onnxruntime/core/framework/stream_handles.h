#pragma once

#include <functional>
#include <unordered_map>
#include "core/framework/ortdevice.h"

namespace onnxruntime {
class IExecutionProvider;
// this opaque handle could be anything the target device generated.
// it could be a cuda event, or a cpu notification implementation
using NotificationHandle = void*;
// it can be either a cuda stream, or even nullptr for device doesn't have stream support like cpu.
using StreamHandle = void*;

// a stream abstraction which hold an opaque handle, and a reference to which EP instance this stream belong to.
// it need to be EP instance as we might have different stream on different EP with same type.
// i.e. different cuda stream on different GPU.
namespace synchronize {
struct Notification;
}

struct Stream {
  StreamHandle handle;
  const OrtDevice& device;
  uint64_t timestamp{0};
  // TODO: do we really need it to be a dynamic map?
  std::unordered_map<Stream*, uint64_t> other_stream_clock;

  Stream(StreamHandle h, const OrtDevice& d) : handle(h), device(d) {}

  uint64_t BumpTimeStampAndReturn() {
    return ++timestamp;
  }

  void UpdateStreamClock(const std::unordered_map<Stream*, uint64_t>& clock) {
    for (auto& kv : clock) {
      auto it = other_stream_clock.find(kv.first);
      if (it == other_stream_clock.end())
        other_stream_clock.insert(kv);
      else
        other_stream_clock[kv.first] = std::max(it->second, kv.second);
    }
  }

  virtual ~Stream() {}
  virtual std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) {
    return {};
  };
  virtual void Flush(){};
  virtual Status CleanUpOnRunEnd() = 0;
};

namespace synchronize {
struct Notification {
  // which stream create this notificaiton.
  Stream* stream;
  std::unordered_map<Stream*, uint64_t> stream_clock_;

  Notification(Stream* s) : stream(s) {}
  virtual ~Notification() {}

  void ActivateAndUpdate() {
    Activate();
    stream_clock_ = stream->other_stream_clock;
    stream_clock_[stream] = stream->BumpTimeStampAndReturn();
  }

  virtual void Activate() = 0;
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
