// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef ORT_ENABLE_STREAM
#include "core/framework/bfc_arena.h"
#include "core/framework/device_stream_collection.h"
#include "core/framework/session_state.h"

namespace onnxruntime {

class DeviceStreamCollectionImpl {
 public:
  DeviceStreamCollectionImpl(size_t num_streams, const SessionState& sess_state) : num_streams_(num_streams) {
    device_streams_.resize(num_streams, nullptr);
    owned_streams_.reserve(num_streams);
    auto& providers = sess_state.GetExecutionProviders();
    eps_.reserve(providers.NumProviders());
    for (auto& ep : providers) {
      eps_.push_back(ep);
    }
    is_main_graph_ = sess_state.GetGraphViewer().ParentNode() == nullptr;
  }

  ~DeviceStreamCollectionImpl() {
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
      if (stream) {
        for (auto& ep : eps_) {
          auto& allocators = ep->GetAllocators();
          for (auto& alloc : allocators) {
            if (alloc->Info().device == stream->GetDevice() &&
                alloc->Info().alloc_type == OrtArenaAllocator) {
              auto* arena_alloc = static_cast<BFCArena*>(alloc.get());
              auto* stream_aware_alloc = StreamAwareArena::FromBFCArena(*arena_alloc);
              if (stream_aware_alloc) {
                stream_aware_alloc->ReleaseStreamBuffers(stream.get());
              }
            }
          }
        }
      }
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
    device_streams_[idx] = stream;
  }

  gsl::span<Stream*> GetStreams() {
    return device_streams_;
  }

  Stream* GetStream(size_t stream_idx) const {
    ORT_ENFORCE(stream_idx < num_streams_);
    return device_streams_[stream_idx];
  }

  size_t NumStreams() { return num_streams_; }

 private:
  size_t num_streams_;
  std::vector<Stream*> device_streams_;
  InlinedVector<std::unique_ptr<Stream>> owned_streams_;
  // due to training's partial execution, the device streams collection may need to be hold
  // with a different lifetime of session state, we need to hold the reference of EPs.
  InlinedVector<std::shared_ptr<IExecutionProvider>> eps_;
  bool is_main_graph_ = false;
};

DeviceStreamCollection::DeviceStreamCollection(size_t num_streams,
                                               const SessionState& sess_state) : impl_(std::make_unique<DeviceStreamCollectionImpl>(num_streams, sess_state)) {}

DeviceStreamCollection::~DeviceStreamCollection() {}

void DeviceStreamCollection::AddDeviceStream(size_t idx, std::unique_ptr<Stream> stream) {
  impl_->AddDeviceStream(idx, std::move(stream));
}

void DeviceStreamCollection::SetDeviceStream(size_t idx, Stream* stream) {
  impl_->SetDeviceStream(idx, stream);
}

gsl::span<Stream*> DeviceStreamCollection::GetStreams() const {
  return impl_->GetStreams();
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

}  // namespace onnxruntime
#endif
