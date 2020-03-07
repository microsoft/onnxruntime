// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "event_pool.h"

namespace onnxruntime {
namespace contrib {

void OrtEventPool::CreateEvent(int64_t id) { 
  std::lock_guard<std::mutex> guard(mutex_);
  ORT_ENFORCE(
    pool_.find(id) == pool_.end(),
    "Event pool cannot create duplicated events for event ID ", id, ".");
  pool_[id].store(false);
}

void OrtEventPool::RecordEvent(int64_t id) {
  std::lock_guard<std::mutex> guard(mutex_);
  ORT_ENFORCE(
    pool_.find(id) != pool_.end(),
    "Event pool cannot record event for non-existing event ID ", id, ".");
  pool_[id].store(true);
};

void OrtEventPool::DeleteEvent(int64_t id) {
  std::lock_guard<std::mutex> guard(mutex_);
  ORT_ENFORCE(
    pool_.find(id) != pool_.end(),
    "Event pool cannot delete event for non-existing event ID ", id, ".");
  pool_.erase(id);
};

bool OrtEventPool::QueryEvent(int64_t id) {
  return pool_.find(id) != pool_.end() && pool_[id].load();
};

}  // namespace contrib
}  // namespace onnxruntime