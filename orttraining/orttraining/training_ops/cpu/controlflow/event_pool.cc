// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "event_pool.h"

namespace onnxruntime {
namespace contrib {

void OrtEventPool::CheckRange(const int64_t id) const {
  ORT_ENFORCE(
    id >= 0 && id < MaxNumItems,
    "Got id ", id,
    ". It should be in a range from 0 to ",
    MaxNumItems, ".");
}

void OrtEventPool::SignalEvent(int64_t id) {
  CheckRange(id);
  std::unique_lock<std::mutex> lock(pool_[id].mutex);
  pool_[id].signaled.store(true);
  lock.unlock();
  pool_[id].cv.notify_all();
};

void OrtEventPool::ResetEvent(int64_t id) {
  CheckRange(id);
  std::lock_guard<std::mutex> guard(pool_[id].mutex);
  pool_[id].signaled.store(false);
};

bool OrtEventPool::QueryEvent(int64_t id) const {
  CheckRange(id);
  return pool_[id].signaled.load();
}

void OrtEventPool::WaitEvent(int64_t id) const {
  CheckRange(id);
  std::unique_lock<std::mutex> lock(pool_[id].mutex);
  pool_[id].cv.wait(lock, [this, id] { return pool_[id].signaled.load(); });
};

}  // namespace contrib
}  // namespace onnxruntime
