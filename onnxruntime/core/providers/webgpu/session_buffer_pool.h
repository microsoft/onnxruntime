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
  // pooled buffers (LIFO). No-op if the pool is empty.
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
