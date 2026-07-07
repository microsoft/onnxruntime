// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef ORT_ENABLE_STREAM
#include "core/framework/bfc_arena.h"
#include "core/framework/device_stream_collection.h"
#include "core/framework/session_state.h"

#include <optional>

namespace onnxruntime {

class DeviceStreamCollectionImpl {
 public:
  DeviceStreamCollectionImpl(size_t num_streams, const AllocatorMap& allocators, bool is_main_graph)
      : num_streams_(num_streams), allocators_(allocators), is_main_graph_(is_main_graph) {
    device_streams_.resize(num_streams, nullptr);
    owned_streams_.reserve(num_streams);
  }

  ~DeviceStreamCollectionImpl() {
  }

  void ReleaseSingleStreamBuffers(Stream* stream) {
    if (!stream) return;
    for (const auto& it : allocators_) {
      if (it.second->Info().device == stream->GetDevice()) {
        auto* arena = it.second->AsArena();
        if (arena && arena->IsStreamAware()) {
          arena->ReleaseStreamBuffers(stream);
        }
      }
    }
  }

  Status CleanUp(bool sync_streams) {
    if (sync_streams) {
      for (size_t i = 0, lim = device_streams_.size(); i < lim; ++i) {
        Stream* device_stream = device_streams_[i];
        if (stream_override_ && i == stream_override_->first) {
          device_stream = stream_override_->second;
        }
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
    // A user-provided override stream is not in owned_streams_, but memory-pattern and activation
    // buffers are allocated on it (see GetStreamForDevice / GetStream), so untag them here too so a
    // future run can reuse the memory.
    if (stream_override_) {
      ReleaseSingleStreamBuffers(stream_override_->second);
    }

    return Status::OK();
  }

  void AddDeviceStream(size_t idx, std::unique_ptr<Stream> stream) {
    ORT_ENFORCE(idx < num_streams_);
    device_streams_[idx] = stream.get();
    owned_streams_.emplace_back(std::move(stream));
  }

  void SetDeviceStream(size_t idx, Stream* stream) {
    ORT_ENFORCE(idx < num_streams_);
    if (stream_override_) {
      if (idx == stream_override_->first) {
        ORT_THROW("Cannot set device stream for index ", idx,
                  " when there is an active stream override for the same index.");
      }
    }
    device_streams_[idx] = stream;
  }

  Status SetStreamOverride(Stream* stream) {
    ORT_ENFORCE(stream != nullptr);
    for (size_t i = 0, lim = device_streams_.size(); i < lim; ++i) {
      if (device_streams_[i] != nullptr &&
          // Exact match
          device_streams_[i]->GetDevice() == stream->GetDevice()) {
        stream_override_.emplace(i, stream);
        return Status::OK();
      }
    }
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "No matching stream found to override from OrtRunOptions");
  }

  void ResetStreamOverride() {
    stream_override_.reset();
  }

  Stream* GetStream(size_t stream_idx) const {
    ORT_ENFORCE(stream_idx < num_streams_);
    if (stream_override_) {
      if (stream_idx == stream_override_->first) {
        return stream_override_->second;
      }
    }
    return device_streams_[stream_idx];
  }

  size_t NumStreams() { return num_streams_; }

  Stream* GetStreamForDevice(const OrtDevice& device) const {
    // Prefer a user-provided override stream for the device: input data is provided on that stream
    // and allocations must synchronize with it. Otherwise use the device stream bound for this
    // collection.
    if (stream_override_ && stream_override_->second->GetDevice() == device) {
      return stream_override_->second;
    }

    for (Stream* stream : device_streams_) {
      if (stream != nullptr && stream->GetDevice() == device) {
        return stream;
      }
    }

    return nullptr;
  }

 private:
  size_t num_streams_;
  std::vector<Stream*> device_streams_;
  InlinedVector<std::unique_ptr<Stream>> owned_streams_;
  // RunOptions allow specifying a stream override for a specific run.
  // if this is present, it would be used as a stream for a given stream_id
  // we declare it sepately as the original stream in device_streams_ should stay
  // intact for future runs as we cache it in SessionState.
  std::optional<std::pair<size_t, Stream*>> stream_override_;
  const AllocatorMap& allocators_;
  bool is_main_graph_ = false;
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

Status DeviceStreamCollection::SetStreamOverride(Stream* stream) {
  return impl_->SetStreamOverride(stream);
}

void DeviceStreamCollection::ResetStreamOverride() {
  impl_->ResetStreamOverride();
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

Stream* DeviceStreamCollection::GetStreamForDevice(const OrtDevice& device) const {
  return impl_->GetStreamForDevice(device);
}

DeviceStreamCollectionHolder::DeviceStreamCollectionHolder(const SessionState* session_state)
    : session_state_(session_state),
      p_(session_state->AcquireDeviceStreamCollection()) {
}

DeviceStreamCollectionHolder::~DeviceStreamCollectionHolder() {
  if (p_) {
    p_->ResetStreamOverride();
    session_state_->RecycleDeviceStreamCollection(std::move(p_));
  }
}

}  // namespace onnxruntime
#endif
