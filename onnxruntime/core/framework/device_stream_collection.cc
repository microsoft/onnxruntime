// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef ORT_ENABLE_STREAM
#include "core/framework/bfc_arena.h"
#include "core/framework/device_stream_collection.h"
#include "core/framework/session_state.h"

namespace onnxruntime {

struct DummyNotification : public synchronize::Notification {
  DummyNotification(Stream& s) : Notification(s) {}
  void Activate() override {}
};

struct DummyStream : Stream {
  DummyStream(StreamHandle h, const OrtDevice& d) : Stream(h, d) {}
  std::unique_ptr<synchronize::Notification> CreateNotification(size_t /*num_consumers*/) override {
    return std::make_unique<DummyNotification>(*this);
  }
};

class DeviceStreamCollectionImpl {
 public:
  DeviceStreamCollectionImpl(size_t num_streams, const AllocatorMap& allocators, bool is_main_graph) : num_streams_(num_streams), allocators_(allocators), is_main_graph_(is_main_graph) {
    device_streams_.resize(num_streams, nullptr);
    owned_streams_.reserve(num_streams);
    root_stream_ = std::make_unique<DummyStream>(nullptr, root_stream_device_);
  }

  ~DeviceStreamCollectionImpl() {
  }

  void ReleaseSingleStreamBuffers(Stream* stream) {
    if (!stream) return;
    for (auto it : allocators_) {
      if (it.second->Info().device == stream->GetDevice() &&
          it.second->Info().alloc_type == OrtArenaAllocator) {
        auto* arena_alloc = static_cast<BFCArena*>(it.second.get());
        auto* stream_aware_alloc = StreamAwareArena::FromBFCArena(*arena_alloc);
        if (stream_aware_alloc) {
          stream_aware_alloc->ReleaseStreamBuffers(stream);
        }
      }
    }
  }

  Status CleanUp(bool sync_streams) {
    if (sync_streams) {
      for (auto& device_stream : device_streams_) {
        if (device_stream) {
          ORT_RETURN_IF_ERROR(device_stream->CleanUpOnRunEnd());
          if (is_main_graph_) {
            device_stream->Flush();
          }
        }
      }
    }

    // only clean the streams that is owned by current context
    for (auto& stream : owned_streams_) {
      ReleaseSingleStreamBuffers(stream.get());
    }
    ReleaseSingleStreamBuffers(root_stream_.get());
    return Status::OK();
  }

  void AddDeviceStream(size_t idx, std::unique_ptr<Stream> stream) {
    ORT_ENFORCE(idx < num_streams_);
    device_streams_[idx] = stream.get();
    owned_streams_.emplace_back(std::move(stream));
  }

  void SetDeviceStream(size_t idx, Stream* stream) {
    ORT_ENFORCE(idx < num_streams_);
    device_streams_[idx] = stream;
  }

  Stream* GetStream(size_t stream_idx) const {
    ORT_ENFORCE(stream_idx < num_streams_);
    return device_streams_[stream_idx];
  }

  size_t NumStreams() { return num_streams_; }

  Stream* GetRootStream() {
    return root_stream_.get();
  }

 private:
  size_t num_streams_;
  std::vector<Stream*> device_streams_;
  InlinedVector<std::unique_ptr<Stream>> owned_streams_;
  const AllocatorMap& allocators_;
  bool is_main_graph_ = false;
  // This is used in ExecutionFrame when memory pattern is enabled, to allocate the peak size memory
  // labelled this stream in the current thread, instead of the default stream which will be used in all the threads (thus caused thread safe issue)
  std::unique_ptr<Stream> root_stream_;
  OrtDevice root_stream_device_;
  void ReleaseSingleStreamBuffers();
};

DeviceStreamCollection::DeviceStreamCollection(size_t num_streams, const AllocatorMap& allocators, bool is_main_graph)
    : impl_(std::make_unique<DeviceStreamCollectionImpl>(num_streams, allocators, is_main_graph)) {}

DeviceStreamCollection::~DeviceStreamCollection() {}

void DeviceStreamCollection::AddDeviceStream(size_t idx, std::unique_ptr<Stream> stream) {
  impl_->AddDeviceStream(idx, std::move(stream));
}

void DeviceStreamCollection::SetDeviceStream(size_t idx, Stream* stream) {
  impl_->SetDeviceStream(idx, stream);
}

size_t DeviceStreamCollection::NumStreams() const {
  return impl_->NumStreams();
}

Status DeviceStreamCollection::CleanUp(bool sync_streams) {
  return impl_->CleanUp(sync_streams);
}

Stream* DeviceStreamCollection::GetStream(size_t stream_idx) const {
  return impl_->GetStream(stream_idx);
}

Stream* DeviceStreamCollection::GetRootStream() const {
  return impl_->GetRootStream();
}

DeviceStreamCollectionHolder::DeviceStreamCollectionHolder(const SessionState* session_state)
    : session_state_(session_state),
      p_(session_state->AcquireDeviceStreamCollection()) {
}

DeviceStreamCollectionHolder::~DeviceStreamCollectionHolder() {
  if (p_) {
    session_state_->RecycleDeviceStreamCollection(std::move(p_));
  }
}

}  // namespace onnxruntime
#endif
