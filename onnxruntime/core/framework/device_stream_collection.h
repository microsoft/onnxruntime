// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifdef ORT_ENABLE_STREAM
#pragma once
#include "core/framework/stream_handles.h"
#include "gsl/gsl"

namespace onnxruntime {
class SessionState;

class DeviceStreamCollectionImpl;
// a collection of indexed device stream instances.
// the index is the same as the logic sequence index in the execution plan.
// this collection may be cached and reused for future iterations.
class DeviceStreamCollection {
 public:
  DeviceStreamCollection(size_t num_streams, const AllocatorMap& allocators, bool is_main_graph);
  ~DeviceStreamCollection();
  // Add the device stream instance to given index.
  // and set the current collection as the owner of the device stream.
  void AddDeviceStream(size_t stream_idx, std::unique_ptr<Stream> stream);

  // Use an external device stream instance at given index.
  // the current collection is not the owner.
  // this is mainly used in subgraph execution, when we want the
  // subgraph nodes execute on the same stream as parent node.
  // stream could be nullptr, which means the nodes/subgraph is assigned to
  // a EP which doesn't support Stream, i.e. CPU based EPs.
  void SetDeviceStream(size_t stream_idx, Stream* stream);

  // get the Stream instance on given stream index
  // The return value could be nullptr, which means the EP on this
  // logic sequence doesn't support Stream.
  Stream* GetStream(size_t stream_idx) const;

  // get the number of device stream instances.
  size_t NumStreams() const;

  // Since the collection may be reused for future iteration,
  // This API is used to cleanup some resources at the end of an iteration.
  Status CleanUp(bool sync_streams);

  Stream* GetRootStream() const;

 private:
  std::unique_ptr<DeviceStreamCollectionImpl> impl_;
};

struct DeviceStreamCollectionHolder {
  DeviceStreamCollectionHolder(const SessionState* session_state);
  DeviceStreamCollectionHolder() = delete;
  DeviceStreamCollectionHolder(const DeviceStreamCollectionHolder&) = delete;

  ~DeviceStreamCollectionHolder();

  const SessionState* session_state_;
  std::unique_ptr<DeviceStreamCollection> p_;
};

}  // namespace onnxruntime
#endif
