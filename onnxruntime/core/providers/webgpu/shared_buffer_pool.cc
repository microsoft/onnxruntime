// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/shared_buffer_pool.h"

#include <algorithm>

#include "core/common/common.h"

namespace onnxruntime {
namespace webgpu {

SharedBufferPool::SharedBufferPool(std::unordered_map<size_t, size_t> limits)
    : limits_{std::move(limits)} {
  sorted_sizes_.reserve(limits_.size());
  for (const auto& [size, limit] : limits_) {
    sorted_sizes_.push_back(size);
  }
  std::sort(sorted_sizes_.begin(), sorted_sizes_.end());

#ifndef NDEBUG  // if debug build
  ORT_ENFORCE(std::all_of(sorted_sizes_.begin(), sorted_sizes_.end(),
                          [](size_t size) { return size % 16 == 0; }),
              "Bucket sizes must be multiples of 16.");
#endif
}

SharedBufferPool::~SharedBufferPool() {
  for (auto& [size, buffers] : free_buffers_) {
    for (auto* buffer : buffers) {
      wgpuBufferRelease(buffer);
    }
  }
}

size_t SharedBufferPool::CalculateBufferSize(size_t request_size) const {
  auto it = std::lower_bound(sorted_sizes_.begin(), sorted_sizes_.end(), request_size);
  return it == sorted_sizes_.end() ? NormalizeBufferSize(request_size) : *it;
}

WGPUBuffer SharedBufferPool::TryAcquire(size_t buffer_size) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = free_buffers_.find(buffer_size);
  if (it == free_buffers_.end() || it->second.empty()) {
    return nullptr;
  }
  auto buffer = it->second.back();
  it->second.pop_back();
  return buffer;
}

void SharedBufferPool::Return(std::vector<std::pair<WGPUBuffer, size_t>>& buffers) {
  if (buffers.empty()) {
    return;
  }

  // Collected under the lock, released outside it: wgpuBufferRelease is a device call and has no
  // business inside a critical section that other sessions are waiting on.
  std::vector<WGPUBuffer> to_release;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    for (auto& [buffer, size] : buffers) {
      auto& bucket = free_buffers_[size];
      auto limit = limits_.find(size);
      const bool bounded = !limits_.empty();
      if (bounded && (limit == limits_.end() || bucket.size() >= limit->second)) {
        to_release.push_back(buffer);
      } else {
        bucket.push_back(buffer);
      }
    }
  }
  buffers.clear();

  for (auto* buffer : to_release) {
    wgpuBufferRelease(buffer);
  }
}

std::unique_ptr<SharedBufferPool> CreateDefaultStorageBufferPool() {
  return std::make_unique<SharedBufferPool>(
      std::unordered_map<size_t, size_t>{BUCKET_DEFAULT_LIMIT_TABLE});
}

std::unique_ptr<SharedBufferPool> CreateDefaultUniformBufferPool() {
  // Unbounded: uniform buffers are tiny and the working set is naturally small.
  return std::make_unique<SharedBufferPool>(std::unordered_map<size_t, size_t>{});
}

}  // namespace webgpu
}  // namespace onnxruntime
