// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cstddef>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/webgpu/webgpu_external_header.h"

namespace onnxruntime {
namespace webgpu {

// WebGPU requires buffer sizes to be a multiple of 16 bytes.
constexpr size_t NormalizeBufferSize(size_t size) {
  return (size + 15) / 16 * 16;
}

// Bucket size -> maximum number of buffers retained for that size.
//
// TODO: maybe use different bucket size for storage and uniform buffers?
constexpr std::initializer_list<std::pair<const size_t, size_t>> BUCKET_DEFAULT_LIMIT_TABLE = {
    {64, 250},
    {128, 200},
    {256, 200},
    {512, 200},
    {2048, 230},
    {4096, 200},
    {8192, 50},
    {16384, 50},
    {32768, 50},
    {65536, 50},
    {131072, 50},
    {262144, 50},
    {524288, 50},
    {1048576, 50},
    {2097152, 30},
    {4194304, 20},
    {8388608, 10},
    {12582912, 10},
    {16777216, 10},
    {26214400, 15},
    {33554432, 22},
    {44236800, 2},
    {58982400, 6},
    // we don't want to cache the bucket sizes below but not caching them
    // results in some major performance hits for models like sd-turbo.
    {67108864, 6},
    {134217728, 6},
    {167772160, 6},
};

//
// SharedBufferPool holds the free buffers that sessions hand back to each other. It is scoped to
// a WebGpuContext, so its lifetime spans every session using that context.
//
// Not to be confused with SessionBufferPool (session_buffer_pool.h), which is scoped per session
// and moves whole generations of buffers between retiring and incoming per-graph BufferManagers.
// The two serve disjoint cache managers: Graph / GraphSimple feed SessionBufferPool, while
// Bucket / Simple feed this one.
//
// This is deliberately the *only* piece of buffer state shared across sessions. Splitting the
// cache per session instead would make steady-state GPU memory scale with the number of live
// sessions (measured: 8 identical sessions went from 241 MB with a shared pool to 1585 MB with
// per-session pools), because every session would keep its own high-water mark of intermediates.
//
// The complementary half - the "released but not yet submitted" list - stays private to each
// session (see the cache managers in buffer_manager.cc), which is what makes sharing safe: a
// buffer only becomes visible to other sessions once the releasing session has submitted its
// commands. Handing it over earlier is what previously produced silently wrong results.
//
// Thread safety: all methods are internally synchronized. The critical section covers only
// map/vector operations - no GPU calls, no command encoding, no shader compilation - so it is
// on the order of 100ns and does not serialize sessions in any meaningful way.
//
class SharedBufferPool {
 public:
  // `limits` maps bucket size -> maximum number of buffers retained for that size. An empty map
  // means unbounded and accepts any size (the "simple" cache policy).
  explicit SharedBufferPool(std::unordered_map<size_t, size_t> limits);
  ~SharedBufferPool();

  // Rounds a request up to the nearest bucket size. Immutable after construction, so no lock.
  size_t CalculateBufferSize(size_t request_size) const;

  // Takes a buffer of exactly `buffer_size` out of the pool, or nullptr if none is available.
  WGPUBuffer TryAcquire(size_t buffer_size);

  // Returns buffers to the pool, releasing any that exceed the configured limit. `buffers` is
  // cleared. Called only after the owning session has submitted the commands using them.
  void Return(std::vector<std::pair<WGPUBuffer, size_t>>& buffers);

 private:
  const std::unordered_map<size_t, size_t> limits_;
  std::vector<size_t> sorted_sizes_;  // immutable after construction
  std::unordered_map<size_t, std::vector<WGPUBuffer>> free_buffers_;
  std::mutex mutex_;
};

// Pools shared by every session on a WebGpuContext: bucketed for storage buffers, unbounded for
// the small uniform buffers.
std::unique_ptr<SharedBufferPool> CreateDefaultStorageBufferPool();
std::unique_ptr<SharedBufferPool> CreateDefaultUniformBufferPool();

}  // namespace webgpu
}  // namespace onnxruntime
