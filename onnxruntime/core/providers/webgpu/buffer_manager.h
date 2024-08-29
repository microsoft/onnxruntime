// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iosfwd>

#ifdef __EMSCRIPTEN__
#include <emscripten/emscripten.h>
#endif

#include <webgpu/webgpu_cpp.h>

#include "core/framework/data_transfer.h"
#include "core/framework/execution_provider.h"

namespace onnxruntime {
namespace webgpu {

class WebGpuContext;

enum class BufferCacheMode {
  Disabled,
  LazyRelease,
  Simple,
  Bucket
};
std::ostream& operator<<(std::ostream& os, BufferCacheMode mode);

//
// IBufferCacheManager is an interface for buffer cache management.
//
// By implementing this interface, we can have different buffer cache management strategies.
// Currently, we have 3 strategies:
// - Disabled: no cache. always allocate a new buffer and release it immediately after use.
// - LazyRelease: no cache. the difference from Disabled is that it delays the release of buffers until the next refresh.
// - Simple: a simple cache that always keeps buffers. when a buffer is requested, it tries to find a buffer in the cache.
// - Bucket: a cache that keeps buffers in different buckets based on the buffer size, with a maximum number of buffers in each bucket.
//
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
  virtual void OnRefresh() = 0;
};

//
// BufferManager manages operations on buffers.
//
class BufferManager {
 public:
  BufferManager(WebGpuContext& context, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode);

  void Upload(void* src, WGPUBuffer dst, size_t size);
  void MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size);
  WGPUBuffer Create(size_t size, wgpu::BufferUsage usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst);
  void Release(WGPUBuffer buffer);
  void Download(WGPUBuffer src, void* dst, size_t size);
  void RefreshPendingBuffers();

 private:
  IBufferCacheManager& GetCacheManager(WGPUBufferUsage usage) const;
  IBufferCacheManager& GetCacheManager(WGPUBuffer buffer) const;

  WebGpuContext& context_;
  std::unique_ptr<IBufferCacheManager> storage_cache_;
  std::unique_ptr<IBufferCacheManager> uniform_cache_;
  std::unique_ptr<IBufferCacheManager> query_resolve_cache_;
  std::unique_ptr<IBufferCacheManager> default_cache_;

  std::vector<wgpu::Buffer> pending_staging_buffers_;
};

class BufferManagerFactory {
 public:
  static std::unique_ptr<BufferManager> Create(WebGpuContext& context, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode);

 private:
  BufferManagerFactory() {}
};

}  // namespace webgpu
}  // namespace onnxruntime
