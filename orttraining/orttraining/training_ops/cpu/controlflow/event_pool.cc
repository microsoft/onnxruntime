// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "event_pool.h"
#include <iostream>

namespace onnxruntime {
namespace contrib {

void OrtEventPool::SignalEvent(int64_t id) {
  ORT_ENFORCE(id >= 0 && id < MaxNumItems);
  std::unique_lock<std::mutex> lock(pool_[id].mutex);
  std::cout << "(event_pool.cc) Record " << id << std::endl;
  pool_[id].signaled.store(true);
  lock.unlock();
  pool_[id].cv.notify_all();
};

void OrtEventPool::ResetEvent(int64_t id) {
  ORT_ENFORCE(id >= 0 && id < MaxNumItems);
  std::lock_guard<std::mutex> guard(pool_[id].mutex);
  pool_[id].signaled.store(false);
};

bool OrtEventPool::QueryEvent(int64_t id) const {
  ORT_ENFORCE(id >= 0 && id < MaxNumItems);
  return pool_[id].signaled.load();
}

void OrtEventPool::WaitEvent(int64_t id) const {
  ORT_ENFORCE(id >= 0 && id < MaxNumItems);
  std::unique_lock<std::mutex> lock(pool_[id].mutex);
  std::cout << "(event_pool.cc) Wait " << id << std::endl;
  pool_[id].cv.wait(lock, [this, id] { return pool_[id].signaled.load(); });
};

}  // namespace contrib
}  // namespace onnxruntime
