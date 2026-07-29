// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <iosfwd>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/providers/webgpu/webgpu_external_header.h"

#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace webgpu {

class WebGpuContext;

// Command recording state. Owned by whoever owns the recording timeline - the
// WebGpuExecutionProvider (i.e. per session), not the shared WebGpuContext.
//
// Command encoding is explicitly NOT covered by Dawn's ImplicitDeviceSynchronization, and
// ORT holds references to these members across calls, so a context-wide encoder lets one
// session's Flush() finish an encoder another session is still recording into. Per-session
// ownership removes that race without any lock: InferenceSession already serializes a single
// session's Run / Initialize under session_mutex_ (the WebGPU EP reports
// ConcurrentRunSupported() == false), so only one thread records into a given state at a time.
struct CommandRecordingState {
  wgpu::CommandEncoder command_encoder;
  wgpu::ComputePassEncoder compute_pass_encoder;
  uint32_t num_pending_dispatches = 0;

  // True between the creation of a command encoder and the submit that finishes it. Read from
  // buffer release, which can happen on a different thread than the one recording (a tensor may
  // outlive the Run that produced it and be dropped elsewhere), hence atomic.
  std::atomic<bool> has_unsubmitted_work{false};
};

// For command capture and replay
enum class GraphCaptureState {
  Default,
  Capturing,
  Replaying
};

enum class BufferCacheMode {
  Disabled,
  LazyRelease,
  Simple,
  Bucket,
  Graph,
  GraphSimple,
};
std::ostream& operator<<(std::ostream& os, BufferCacheMode mode);

//
// IBufferCacheManager is an interface for buffer cache management.
//
// By implementing this interface, we can have different buffer cache management strategies.
// Currently, we have 5 strategies:
// - Disabled: no cache. always allocate a new buffer and release it immediately after use.
// - LazyRelease: no cache. the difference from Disabled is that it delays the release of buffers until the next refresh.
// - Simple: a simple cache that always keeps buffers. when a buffer is requested, it tries to find a buffer in the cache.
// - Bucket: a cache that keeps buffers in different buckets based on the buffer size, with a maximum number of buffers in each bucket.
// - Graph: used for graph capturing storage buffer cache mode. All buffers will be cached. Buffers can be reused across runs and in one run.
// - GraphSimple: used for graph capturing uniform buffer cache mode. All buffers will be cached. Buffers can be reused across runs but can't be reused in one run.
class IBufferCacheManager {
 public:
  virtual ~IBufferCacheManager() = default;

  // calculate actual buffer size to allocate based on the requested size.
  virtual size_t CalculateBufferSize(size_t request_size) = 0;

  // return a buffer if available in cache. otherwise empty.
  virtual WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) = 0;

  // register a newly created buffer
  virtual void RegisterBuffer(WGPUBuffer buffer, size_t request_size) = 0;

  // release a buffer
  virtual void ReleaseBuffer(WGPUBuffer buffer) = 0;

  // when a stream refresh is requested
  virtual void OnRefresh(GraphCaptureState graph_capture_state) = 0;

  // Extract all cached buffers from this manager, transferring ownership to the
  // caller. The cache's internal containers are cleared (but bucket keys, if any,
  // are preserved). Default returns empty; only graph-mode caches implement this.
  virtual std::vector<std::pair<size_t, WGPUBuffer>> ExtractCachedBuffers() {
    return {};
  }

  // Accept buffers donated from another cache and take ownership of them. Caches
  // that cannot store the buffers must release them via wgpuBufferRelease (the
  // default below) to avoid leaks.
  virtual void AbsorbCachedBuffers(std::vector<std::pair<size_t, WGPUBuffer>>&& buffers) {
    for (auto& entry : buffers) {
      if (entry.second) {
        wgpuBufferRelease(entry.second);
      }
    }
  }
};

//
// SharedBufferPool holds the free buffers that sessions hand back to each other.
//
// It is deliberately the *only* piece of buffer state shared across sessions. Splitting the
// cache per session instead would make steady-state GPU memory scale with the number of live
// sessions (measured: 8 identical sessions went from 241 MB with a shared pool to 1585 MB with
// per-session pools), because every session would keep its own high-water mark of intermediates.
//
// The complementary half - the "released but not yet submitted" list - stays private to each
// session (see the cache managers), which is what makes sharing safe: a buffer only becomes
// visible to other sessions once the releasing session has submitted its commands. Handing it
// over earlier is what previously produced silently wrong results.
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

//
// BufferManager manages operations on buffers.
//
class BufferManager {
 public:
  BufferManager(WebGpuContext& context, CommandRecordingState& recording, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode, BufferCacheMode default_buffer_cache_mode);
  void Upload(void* src, WGPUBuffer dst, size_t size) const;
  void MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size) const;
  WGPUBuffer Create(size_t size, wgpu::BufferUsage usage) const;
  bool SupportsUMA() const;  // Check if CreateUMA is supported (i.e., the device has BufferMapExtendedUsages feature)
  void Release(WGPUBuffer buffer) const;
  void Download(WGPUBuffer src, void* dst, size_t size) const;
  void RefreshPendingBuffers(GraphCaptureState graph_capture_state) const;

  // The recording state that commands issued through this manager are encoded into.
  CommandRecordingState& Recording() const { return recording_; }

  // Direct access to the underlying cache managers. Used by SessionBufferPool to
  // donate/seed buffers across per-graph BufferManager lifetimes.
  IBufferCacheManager& StorageCache() { return *storage_cache_; }
  IBufferCacheManager& UniformCache() { return *uniform_cache_; }

 private:
  IBufferCacheManager& GetCacheManager(wgpu::BufferUsage usage) const;
  IBufferCacheManager& GetCacheManager(WGPUBuffer buffer) const;

  WebGpuContext& context_;
  CommandRecordingState& recording_;
  std::unique_ptr<IBufferCacheManager> storage_cache_;
  std::unique_ptr<IBufferCacheManager> uniform_cache_;
  std::unique_ptr<IBufferCacheManager> query_resolve_cache_;
  std::unique_ptr<IBufferCacheManager> default_cache_;
};

class BufferManagerFactory {
 public:
  static std::unique_ptr<BufferManager> Create(WebGpuContext& context, CommandRecordingState& recording, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode, BufferCacheMode default_buffer_cache_mode);

 private:
  BufferManagerFactory() {}
};

}  // namespace webgpu
}  // namespace onnxruntime
