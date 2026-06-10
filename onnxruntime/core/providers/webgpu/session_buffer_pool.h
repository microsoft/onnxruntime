// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <utility>
#include <vector>

#include "core/providers/webgpu/webgpu_external_header.h"

namespace onnxruntime {
namespace webgpu {

class BufferManager;

// SessionBufferPool retains buffers from retired per-graph BufferManagers so
// that subsequent generators on the same session can reuse them instead of
// allocating from the device. Scoped per WebGpuExecutionProvider (per session)
// because intermediate buffer shapes are model-dependent.
//
// Thread safety: not internally synchronized. Donate / SeedInto / Clear must
// not be called concurrently. The WebGPU EP relies on the fact that
// InferenceSession serializes Run and ReleaseCapturedGraph under
// session_mutex_ for execution providers where ConcurrentRunSupported() is
// false, which serializes the only call sites (OnRunStart -> SeedInto and
// ReleaseCapturedGraph -> Donate). Clear runs from the EP destructor after
// all Run / ReleaseCapturedGraph calls have returned.
class SessionBufferPool {
 public:
  explicit SessionBufferPool(size_t max_generations);

  ~SessionBufferPool();

  // Move freed buffers from a retiring per-graph BufferManager into the pool.
  // When the pool is at capacity, the oldest slot is evicted (its buffers
  // released to the device) so the freshest buffers are always retained. This
  // lets the pool adapt when intermediate buffer shapes change between
  // generators (for example when max_length differs).
  void Donate(BufferManager& retiring_mgr);

  // Pre-populate a newly created per-graph BufferManager with one slot worth of
  // pooled buffers (LIFO). No-op if the pool is empty. The receiving manager
  // may end up holding at most one donated slot's worth of buffer sizes that
  // the new generator never requests; those sit in the receiving manager's
  // caches until that manager is itself donated or destroyed.
  void SeedInto(BufferManager& new_mgr);

  // Release all pooled buffers. Called on session teardown.
  void Clear();

  size_t Size() const { return slots_.size(); }
  size_t Capacity() const { return max_generations_; }

 private:
  struct Slot {
    std::vector<std::pair<size_t, WGPUBuffer>> storage;
    std::vector<std::pair<size_t, WGPUBuffer>> uniform;
  };
  std::vector<Slot> slots_;
  size_t max_generations_;
};

}  // namespace webgpu
}  // namespace onnxruntime
