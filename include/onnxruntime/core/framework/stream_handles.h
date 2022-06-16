#pragma once

#include <functional>
#include <unordered_map>

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
  const IExecutionProvider* provider;
  uint64_t timestamp{0};
  // TODO: do we really need it to be a dynamic map?
  std::unordered_map<Stream*, uint64_t> other_stream_clock;

  Stream::Stream(StreamHandle h, const IExecutionProvider* p) : handle(h), provider(p) {}

  uint64_t BumpTimeStampAndReturn() {
    return ++timestamp;
  }

  void UpdateStreamClock(Stream* stream, uint64_t clock) {
    other_stream_clock[stream] = clock;
  }

  virtual ~Stream() {}
  virtual std::unique_ptr<synchronize::Notification> CreateNotification(size_t num_consumers) = 0;
  virtual void Flush() = 0;
};

namespace synchronize {
struct Notification {
  // which stream create this notificaiton.
  Stream* stream;
  uint64_t timestamp{0};

  Notification(Stream* s) : stream(s) {}
  virtual ~Notification() {}

  void ActivateAndUpdate() {
    Activate();
    timestamp = stream->BumpTimeStampAndReturn();
  }

  virtual void Activate() = 0;
};
}  // namespace synchronize

// the definition for the handle for stream commands
// EP can register the handle to the executor.
// in the POC, just use primitive function pointer
// TODO: use a better way to dispatch handles.
using WaitNotificationFn = std::function<bool(Stream&, synchronize::Notification&)>;
using CreateStreamFn = std::function<std::unique_ptr<Stream>(const IExecutionProvider*)>;

// an interface of a simple registry which hold the handles EP registered.
// make it interface so we can pass it through shared library based execution providers
class IStreamCommandHandleRegistry {
 public:
  // Wait is a little special as we need to consider the source stream the notification generated, and the stream we are waiting.
  // i.e., for an cuda event what notify the memory copy, it could be wait on a CPU stream, or on another cuda stream.
  virtual WaitNotificationFn GetWaitHandle(const std::string& notification_owner_ep_type, const std::string& executor_ep_type) = 0;

  virtual CreateStreamFn GetCreateStreamFn(const std::string& execution_provider_type) = 0;

  virtual void RegisterWaitFn(const std::string& notification_ep_type, const std::string& ep_type, WaitNotificationFn fn) = 0;

  virtual void RegisterCreateStreamFn(const std::string& ep_type, CreateStreamFn f) = 0;
};

}
