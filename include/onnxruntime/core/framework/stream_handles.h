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

// a stream abstraction which hold an opaque handle, and a reference to which EP instance this stream belong to.
// it need to be EP instance as we might have different stream on different EP with same type.
// i.e. different cuda stream on different GPU.
struct Stream {
  StreamHandle handle;
  const IExecutionProvider* provider;

  Stream::Stream(StreamHandle h, const IExecutionProvider* p) : handle(h), provider(p) {}
};

// a simple registry which hold the handles EP registered.
class StreamCommandHandleRegistry {
 public:
  CreateNotificationFn GetCreateNotificationFn(Stream* stream);

  ReleaseNotificationFn GetReleaseNotificationFn(Stream* stream);

  // Wait is a little special as we need to consider the source stream the notification generated, and the stream we are waiting.
  // i.e., for an cuda event what notify the memory copy, it could be wait on a CPU stream, or on another cuda stream.
  WaitNotificationFn GetWaitHandle(Stream* notification_owner_stream, const std::string& executor_ep_type);

  NotifyNotificationFn GetNotifyHandle(Stream* notification_owner_stream);

  CreateStreamFn GetCreateStreamFn(const std::string& execution_provider_type);

  ReleaseStreamFn GetReleaseStreamFn(const std::string& execution_provider_type);

  static StreamCommandHandleRegistry& GetInstance() {
    static StreamCommandHandleRegistry instance_;
    return instance_;
  }

  void RegisterCreateNotificationFn(const std::string& ep_type, CreateNotificationFn fn);

  void RegisterReleaseNotificationFn(const std::string& ep_type, ReleaseNotificationFn fn);

  void RegisterWaitFn(const std::string& notification_ep_type, const std::string& ep_type, WaitNotificationFn fn);

  void RegisterNotifyFn(const std::string& notification_ep_type, NotifyNotificationFn fn);

  void RegisterCreateStreamFn(const std::string& ep_type, CreateStreamFn f);

  void RegisterReleaseStreamFn(const std::string& ep_type, ReleaseStreamFn f);

 private:
  StreamCommandHandleRegistry() = default;

  std::unordered_map<std::string, CreateNotificationFn> create_notification_map_;
  std::unordered_map<std::string, ReleaseNotificationFn> release_notification_map_;
  std::unordered_map<std::string, WaitNotificationFn> notification_wait_map_;
  std::unordered_map<std::string, NotifyNotificationFn> notification_notify_map_;
  std::unordered_map<std::string, CreateStreamFn> create_stream_map_;
  std::unordered_map<std::string, ReleaseStreamFn> release_stream_map_;
};
}
