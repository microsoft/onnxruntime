#include "core/framework/stream_pool.h"
#include "core/framework/session_state.h"

namespace onnxruntime {
Status StreamPool::Init(const SessionState& session_state,
                       const std::unordered_map<std::string, size_t>& ep_max_stream_count) {
  auto& eps = session_state.GetExecutionProviders();
  size_t num_eps = eps.NumProviders();
  auto& stream_handle_registry = session_state.GetStreamHandleRegistryInstance();
  size_t cur = 0;
  size_t ep_idx = 0;
  for (auto& ep : eps) {
    auto it = ep_max_stream_count.find(ep->Type());
    ORT_RETURN_IF(it == ep_max_stream_count.end());
    size_t num_stream = it->second;
    // record ep's stream begin index
    ep_to_streams_.insert({ep.get(), ep_idx});
    // record ep's stream start address
    ep_stream_cur_.push_back(cur);
    // init ep's current stream offset
    ep_stream_offset_.push_back(cur);
    // create streams
    auto create_stream_fn = stream_handle_registry.GetCreateStreamFn(ep->Type());
    for (size_t i = 0; i < num_stream; i++) {
      streams_.emplace_back(create_stream_fn(ep.get()));
    }
    cur += num_stream;
  }
  ep_stream_offset_.push_back(cur);
  initialized_ = true;
  return Status::OK();
}

StreamPool::~StreamPool() {}

Stream* StreamPool::GetStream(const IExecutionProvider* ep) {
  if (!initialized_)
    return nullptr;
  auto it = ep_to_streams_.find(ep);
  if (it == ep_to_streams_.end())
    return nullptr;
  size_t ep_idx = it->second;
  // avoid race condition
  std::lock_guard<OrtMutex> lock(lock_);
  auto* stream = streams_[ep_stream_cur_[ep_idx]].get();
  // get next position
  ep_stream_cur_[ep_idx]++;
  if (ep_stream_cur_[ep_idx] == ep_stream_offset_[ep_idx + 1])
    ep_stream_cur_[ep_idx] = ep_stream_offset_[ep_idx];
  return stream;
}

void StreamPool::ReleaseStream(Stream*) {}

}