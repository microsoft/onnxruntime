#pragma once

#include "core/common/common.h"
#include "core/framework/stream_handles.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {
class SessionState;
class IExecutionProvider;

class StreamPool {
public:
  StreamPool() {}
  virtual ~StreamPool();
  // init is not thread safe.
  Status Init(const SessionState& session_state,
              const std::unordered_map<std::string, size_t>& ep_max_stream_count);
  Stream* GetStream(const IExecutionProvider* ep);
  void ReleaseStream(Stream*);

private:
  bool initialized_{false};
  std::vector<std::unique_ptr<Stream>> streams_;
  std::unordered_map<const IExecutionProvider*, size_t> ep_to_streams_;
  std::vector<size_t> ep_stream_offset_;
  std::vector<size_t> ep_stream_cur_;
  mutable OrtMutex lock_;
};

}  // namespace onnxruntime
