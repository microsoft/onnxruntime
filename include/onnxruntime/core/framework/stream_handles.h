#pragma once

#include <functional>

namespace onnxruntime {

class IExecutionProvider;
    // this opaque handle could be anything the target device generated.
// it could be a cuda event, or a cpu notification implementation
using NotificationHandle = void*;
// it can be either a cuda stream, or even nullptr for device doesn't have stream support like cpu.
using StreamHandle = void*;

// the definition for the handle for stream commands
// EP can register the handle to the executor.
// in the POC, just use primitive function pointer
// TODO: use a better way to dispatch handles.
using WaitNotificationFn = std::function<void(NotificationHandle&)>;
using NotifyNotificationFn = std::function<void(NotificationHandle&)>;
using CreateNotificationFn = std::function<NotificationHandle(const StreamHandle&)>;
using ReleaseNotificationFn = std::function<void(NotificationHandle)>;
using CreateStreamFn = std::function<StreamHandle()>;
using ReleaseStreamFn = std::function<void(StreamHandle)>;
using FlushStreamFn = std::function<void(StreamHandle)>;

// a stream abstraction which hold an opaque handle, and a reference to which EP instance this stream belong to.
// it need to be EP instance as we might have different stream on different EP with same type.
// i.e. different cuda stream on different GPU.
struct Stream {
  StreamHandle handle;
  const IExecutionProvider* provider;

  Stream::Stream(StreamHandle h, const IExecutionProvider* p) : handle(h), provider(p) {}
};

// an interface of a simple registry which hold the handles EP registered.
// make it interface so we can pass it through shared library based execution providers
class IStreamCommandHandleRegistry {
 public:
  virtual CreateNotificationFn GetCreateNotificationFn(Stream* stream) = 0;

  virtual ReleaseNotificationFn GetReleaseNotificationFn(Stream* stream) = 0;

  // Wait is a little special as we need to consider the source stream the notification generated, and the stream we are waiting.
  // i.e., for an cuda event what notify the memory copy, it could be wait on a CPU stream, or on another cuda stream.
  virtual WaitNotificationFn GetWaitHandle(Stream* notification_owner_stream, const std::string& executor_ep_type) = 0;

  virtual NotifyNotificationFn GetNotifyHandle(Stream* notification_owner_stream) = 0;

  virtual CreateStreamFn GetCreateStreamFn(const std::string& execution_provider_type) = 0;

  virtual ReleaseStreamFn GetReleaseStreamFn(const std::string& execution_provider_type) = 0;

  virtual FlushStreamFn GetFlushStreamFn(const std::string& execution_provider_type) = 0;

  virtual void RegisterCreateNotificationFn(const std::string& ep_type, CreateNotificationFn fn) = 0;

  virtual void RegisterReleaseNotificationFn(const std::string& ep_type, ReleaseNotificationFn fn) = 0;

  virtual void RegisterWaitFn(const std::string& notification_ep_type, const std::string& ep_type, WaitNotificationFn fn) = 0;

  virtual void RegisterNotifyFn(const std::string& notification_ep_type, NotifyNotificationFn fn) = 0;

  virtual void RegisterCreateStreamFn(const std::string& ep_type, CreateStreamFn f) = 0;

  virtual void RegisterReleaseStreamFn(const std::string& ep_type, ReleaseStreamFn f) = 0;

  virtual void RegisterFlushStreamFn(const std::string& ep_type, FlushStreamFn f) = 0;
};

IStreamCommandHandleRegistry& GetStreamHandleRegistryInstance();

}
