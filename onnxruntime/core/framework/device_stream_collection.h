#pragma once
#include "core/framework/stream_handles.h"
#include "gsl/gsl"

namespace onnxruntime {
class SessionState;

class DeviceStreamCollectionImpl;
class DeviceStreamCollection {
 public:
  DeviceStreamCollection(size_t num_streams, const SessionState& sess_state);
  ~DeviceStreamCollection();
  void SetDeviceStream(size_t, std::unique_ptr<Stream> stream);
  void SetDeviceStream(size_t, Stream* stream);
  gsl::span<Stream*> GetStreams() const;
  size_t NumStreams() const;
  Status CleanUp();

 private:
  std::unique_ptr<DeviceStreamCollectionImpl> impl_;
};
}  // namespace onnxruntime
