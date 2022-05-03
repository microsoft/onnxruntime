#include "core/framework/stream_handles.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {

static inline std::string GetWaitKey(const std::string& notificaiton_ep_type, const std::string& executor_ep_type) {
  return std::string(notificaiton_ep_type) + ":" + executor_ep_type;
}

class StreamCommandHandleRegistryImpl : public IStreamCommandHandleRegistry {
 public:
  CreateNotificationFn GetCreateNotificationFn(Stream* stream) override {
    auto it = create_notification_map_.find(stream->provider->Type());
    return it == create_notification_map_.end() ? nullptr : it->second;
  }

  ReleaseNotificationFn GetReleaseNotificationFn(Stream* stream) override {
    auto it = release_notification_map_.find(stream->provider->Type());
    return it == release_notification_map_.end() ? nullptr : it->second;
  }

  // Wait is a little special as we need to consider the source stream the notification generated, and the stream we are waiting.
  // i.e., for an cuda event what notify the memory copy, it could be wait on a CPU stream, or on another cuda stream.
  WaitNotificationFn GetWaitHandle(Stream* notification_owner_stream, const std::string& executor_ep_type) override {
    auto it = notification_wait_map_.find(GetWaitKey(notification_owner_stream->provider->Type(), executor_ep_type));
    return it == notification_wait_map_.end() ? nullptr : it->second;
  }

  NotifyNotificationFn GetNotifyHandle(Stream* notification_owner_stream) override {
    auto it = notification_notify_map_.find(notification_owner_stream->provider->Type());
    return it == notification_notify_map_.end() ? nullptr : it->second;
  }

  CreateStreamFn GetCreateStreamFn(const std::string& execution_provider_type) override {
    auto it = create_stream_map_.find(execution_provider_type);
    return it == create_stream_map_.end() ? nullptr : it->second;
  }

  ReleaseStreamFn GetReleaseStreamFn(const std::string& execution_provider_type) override {
    auto it = release_stream_map_.find(execution_provider_type);
    return it == release_stream_map_.end() ? nullptr : it->second;
  }

  void RegisterCreateNotificationFn(const std::string& ep_type, CreateNotificationFn fn) override {
    create_notification_map_.insert({ep_type, fn});
  }

  void RegisterReleaseNotificationFn(const std::string& ep_type, ReleaseNotificationFn fn) override {
    release_notification_map_.insert({ep_type, fn});
  }

  void RegisterWaitFn(const std::string& notification_ep_type, const std::string& ep_type, WaitNotificationFn fn) override {
    notification_wait_map_.insert({GetWaitKey(notification_ep_type, ep_type), fn});
  }

  void RegisterNotifyFn(const std::string& notification_ep_type, NotifyNotificationFn fn) override {
    notification_notify_map_.insert({notification_ep_type, fn});
  }

  void RegisterCreateStreamFn(const std::string& ep_type, CreateStreamFn f) override {
    create_stream_map_.insert({ep_type, f});
  }

  void RegisterReleaseStreamFn(const std::string& ep_type, ReleaseStreamFn f) override {
    release_stream_map_.insert({ep_type, f});
  }

  static StreamCommandHandleRegistryImpl& GetInstance() {
    static StreamCommandHandleRegistryImpl instance;
    return instance;
  }

 private:
  StreamCommandHandleRegistryImpl() = default;

  std::unordered_map<std::string, CreateNotificationFn> create_notification_map_;
  std::unordered_map<std::string, ReleaseNotificationFn> release_notification_map_;
  std::unordered_map<std::string, WaitNotificationFn> notification_wait_map_;
  std::unordered_map<std::string, NotifyNotificationFn> notification_notify_map_;
  std::unordered_map<std::string, CreateStreamFn> create_stream_map_;
  std::unordered_map<std::string, ReleaseStreamFn> release_stream_map_;
};

IStreamCommandHandleRegistry& GetStreamHandleRegistryInstance() {
  return StreamCommandHandleRegistryImpl::GetInstance();
}

}