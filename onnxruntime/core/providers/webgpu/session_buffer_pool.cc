// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/session_buffer_pool.h"

#include "core/providers/webgpu/buffer_manager.h"

namespace onnxruntime {
namespace webgpu {

namespace {
void ReleaseSlotBuffers(std::vector<std::pair<size_t, WGPUBuffer>>& entries) {
  for (auto& entry : entries) {
    if (entry.second) {
      wgpuBufferRelease(entry.second);
    }
  }
  entries.clear();
}
}  // namespace

SessionBufferPool::SessionBufferPool(size_t max_generations)
    : max_generations_{max_generations} {
}

SessionBufferPool::~SessionBufferPool() {
  Clear();
}

void SessionBufferPool::Donate(BufferManager& retiring_mgr) {
  // Safe to take ownership of the retiring manager's buffers because:
  //   1. InferenceSession::session_mutex_ serializes Run and
  //      ReleaseCapturedGraph for the WebGPU EP, so no Run is in progress on
  //      this session while we donate.
  //   2. OnRunEnd flushes the queue (and, in ValidationMode::Basic+, fences
  //      via PopErrorScope) before Run returns, so prior captured-graph work
  //      has been submitted.
  //   3. All work goes through a single WebGPU device queue, which orders
  //      future submits after the prior submitted work, so a subsequent
  //      generator's first Submit using these buffer handles happens-after
  //      any in-flight read/write on the GPU timeline.
  // Donating across a different queue or after a host-side readback on
  // donated buffers would invalidate (3) and require an explicit fence here.
  if (max_generations_ == 0) {
    return;
  }

  Slot slot;
  slot.storage = retiring_mgr.StorageCache().ExtractCachedBuffers();
  slot.uniform = retiring_mgr.UniformCache().ExtractCachedBuffers();

  // The caches are now empty either way; the retiring manager is about to be
  // destroyed, so the early-exit only avoids pushing an empty slot.
  if (slot.storage.empty() && slot.uniform.empty()) {
    return;
  }

  // Evict the oldest slot if at capacity so the freshest buffers (which most
  // accurately reflect the current per-generator shape distribution) are kept.
  while (slots_.size() >= max_generations_) {
    auto& victim = slots_.front();
    ReleaseSlotBuffers(victim.storage);
    ReleaseSlotBuffers(victim.uniform);
    slots_.erase(slots_.begin());
  }

  slots_.emplace_back(std::move(slot));
}

void SessionBufferPool::SeedInto(BufferManager& new_mgr) {
  if (slots_.empty()) {
    return;
  }
  Slot slot = std::move(slots_.back());
  slots_.pop_back();
  new_mgr.StorageCache().AbsorbCachedBuffers(std::move(slot.storage));
  new_mgr.UniformCache().AbsorbCachedBuffers(std::move(slot.uniform));
}

void SessionBufferPool::Clear() {
  for (auto& slot : slots_) {
    ReleaseSlotBuffers(slot.storage);
    ReleaseSlotBuffers(slot.uniform);
  }
  slots_.clear();
}

}  // namespace webgpu
}  // namespace onnxruntime
