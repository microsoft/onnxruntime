// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"

namespace onnxruntime {
namespace webgpu {

constexpr size_t NormalizeBufferSize(size_t size) {
  return (size + 15) / 16 * 16;
}

class DisabledCacheManager : public IBufferCacheManager {
 public:
  DisabledCacheManager() = default;
  virtual ~DisabledCacheManager() = default;

 private:
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t /*buffer_size*/) override {
    // always return empty buffer
    return nullptr;
  }
  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }
  void ReleaseBuffer(WGPUBuffer buffer) override {
    wgpuBufferRelease(buffer);
  }

  void OnRefresh() override {
    // no-op
  }
};

class IBufferCacheManagerUMA {
 public:
  virtual ~IBufferCacheManagerUMA() = default;

  // Ensure all MapAsync callbacks are processed before returning.
  virtual void WaitBufferMapAsync(wgpu::Instance instance) {};
};

class DisabledCacheManagerUMA : public DisabledCacheManager, public IBufferCacheManagerUMA {
 public:
  explicit DisabledCacheManagerUMA(BufferManagerUMA* buffer_manager) : buffer_manager_{buffer_manager} {}
  ~DisabledCacheManagerUMA() = default;

 private:
  void ReleaseBuffer(WGPUBuffer buffer) override {
    // Notify the buffer manager that the buffer is released.
    buffer_manager_->OnBufferRelease(buffer);

    wgpuBufferRelease(buffer);
  }

  BufferManagerUMA* buffer_manager_;
};


class LazyReleaseCacheManager : public IBufferCacheManager {
 public:
  LazyReleaseCacheManager() = default;
  virtual ~LazyReleaseCacheManager() = default;
 private:
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t /*buffer_size*/) override {
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(wgpu::Buffer::Acquire(buffer));
  }

  virtual void OnRefresh() override { pending_buffers_.clear(); }

protected:
  std::vector<wgpu::Buffer> pending_buffers_;
};

class LazyReleaseCacheManagerUMA : public LazyReleaseCacheManager, public IBufferCacheManagerUMA {
 public:
  explicit LazyReleaseCacheManagerUMA(BufferManagerUMA* buffer_manager) : buffer_manager_{buffer_manager} {}
  ~LazyReleaseCacheManagerUMA() = default;

 private:
  void OnRefresh() override {
    for (auto& buffer : pending_buffers_) {
      // Notify the buffer manager that the buffer is released.
      buffer_manager_->OnBufferRelease(buffer.Get());
    }
    pending_buffers_.clear();
  }

  BufferManagerUMA* buffer_manager_;
};



class SimpleCacheManager : public IBufferCacheManager {
 public:
  SimpleCacheManager() = default;
  virtual ~SimpleCacheManager() = default;

 protected:
  virtual void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(wgpu::Buffer::Acquire(buffer));
  }

  virtual void OnRefresh() override {
    for (auto& buffer : pending_buffers_) {
      buffers_[static_cast<size_t>(buffer.GetSize())].emplace_back(std::move(buffer));
    }
    pending_buffers_.clear();
  }

  std::vector<wgpu::Buffer> pending_buffers_;
  std::map<size_t, std::vector<wgpu::Buffer>> buffers_;

 private:
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buffers_.find(buffer_size);
    if (it != buffers_.end() && !it->second.empty()) {
      auto buffer = it->second.back().MoveToCHandle();
      it->second.pop_back();
      return buffer;
    }

    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }
};

// This class extends the existing cache managers with UMA support.
// It maps the returned buffers asynchronously to ensure the buffers are ready for reuse.
template <typename T>
class CacheManagerUMA : public T, public IBufferCacheManagerUMA {
 protected:
  explicit CacheManagerUMA(BufferManagerUMA* buffer_manager) : buffer_manager_{buffer_manager} {}
  virtual ~CacheManagerUMA() {
    auto Cleanup = [this](std::vector<wgpu::Buffer>& buffers) {
      for (auto& buffer : buffers) {
        buffer_manager_->OnBufferRelease(buffer.Get());
      }
      buffers.clear();
    };

    Cleanup(returned_buffers_);
    Cleanup(pending_buffers_);
    Cleanup(bad_buffers_);
  }

  void ReleaseBuffer(WGPUBuffer buffer) override { returned_buffers_.emplace_back(wgpu::Buffer::Acquire(buffer)); }

  void OnRefresh() override {
    // Move the pending buffers to the reuse buffers.
    T::OnRefresh();

    // Move the returned buffers to the mapping buffers.
    for (auto& buffer : returned_buffers_) {
      mapping_buffers_.emplace_back(buffer);
      // Ensure the buffer is mapped before releasing it for reuse.
      buffer.MapAsync(wgpu::MapMode::Write, 0, buffer.GetSize(), wgpu::CallbackMode::AllowSpontaneous,
                      [this, buffer](wgpu::MapAsyncStatus status, wgpu::StringView) {
                        if (status == wgpu::MapAsyncStatus::Success) {
                          wgpuBufferAddRef(buffer.Get());
                          // Add to the pending buffers.
                          T::ReleaseBuffer(buffer.Get());
                        } else {
                          // The buffer that wasn't mapped successfully is considered as "bad" and therefore can't be
                          // used anymore. We will discard it in next OnRefresh();
                          bad_buffers_.emplace_back(buffer);
                        }
                        // Remove the buffer from the mapping buffers.
                        auto it = std::find_if(mapping_buffers_.begin(), mapping_buffers_.end(),
                                               [buffer](const wgpu::Buffer& b) { return b.Get() == buffer.Get(); });
                        if (it != mapping_buffers_.end()) {
                          mapping_buffers_.erase(it);
                        }
                      });
    }

    returned_buffers_.clear();

    // Discard the bad buffers.
    for (auto& buffer : bad_buffers_) {
      // Notify the buffer manager that the buffer is released.
      buffer_manager_->OnBufferRelease(buffer.Get());
    }
    bad_buffers_.clear();
  }

  void WaitBufferMapAsync(wgpu::Instance instance) override {
    while (!mapping_buffers_.empty()) {
      instance.ProcessEvents();
    }
  }

  BufferManagerUMA* buffer_manager_;

 private:
  // Buffers that have been returned by clients since the last refresh.
  std::vector<wgpu::Buffer> returned_buffers_;
  // Buffers that are being mapped for reuse.
  std::vector<wgpu::Buffer> mapping_buffers_;
  // Buffers that weren't mapped successfully, and can't be re-used anymore.
  std::vector<wgpu::Buffer> bad_buffers_;
};

class SimpleCacheManagerUMA : public CacheManagerUMA<SimpleCacheManager> {
 public:
  explicit SimpleCacheManagerUMA(BufferManagerUMA* buffer_manager)
      : CacheManagerUMA<SimpleCacheManager>(buffer_manager) {}
  ~SimpleCacheManagerUMA() {
    for (auto& pair : buffers_) {
      for (auto& buffer : pair.second) {
        // Notify the buffer manager that the buffer is released.
        buffer_manager_->OnBufferRelease(buffer.Get());
      }
    }
  }
};

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

class BucketCacheManager : public IBufferCacheManager {
 public:
  BucketCacheManager() : buckets_limit_{BUCKET_DEFAULT_LIMIT_TABLE} {
    Initialize();
  }
  BucketCacheManager(std::unordered_map<size_t, size_t>&& buckets_limit) : buckets_limit_{buckets_limit} {
    Initialize();
  }
  virtual ~BucketCacheManager() = default;

 protected:
  virtual void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(wgpu::Buffer::Acquire(buffer));
  }

  virtual void OnRefresh() override {
    // TODO: consider graph capture. currently not supported

    for (auto& buffer : pending_buffers_) {
      auto buffer_size = static_cast<size_t>(buffer.GetSize());

      auto it = buckets_.find(buffer_size);
      if (it != buckets_.end() && it->second.size() < buckets_limit_[buffer_size]) {
        it->second.emplace_back(std::move(buffer));
      }
    }

    pending_buffers_.clear();
  }

  void Initialize() {
    buckets_keys_.reserve(buckets_limit_.size());
    buckets_.reserve(buckets_limit_.size());
    for (const auto& pair : buckets_limit_) {
      buckets_keys_.push_back(pair.first);
      buckets_.emplace(pair.first, std::vector<wgpu::Buffer>());
    }
    std::sort(buckets_keys_.begin(), buckets_keys_.end());

#ifndef NDEBUG  // if debug build
    ORT_ENFORCE(std::all_of(buckets_keys_.begin(), buckets_keys_.end(), [](size_t size) { return size % 16 == 0; }),
                "Bucket sizes must be multiples of 16.");

    for (size_t i = 1; i < buckets_keys_.size(); ++i) {
      ORT_ENFORCE(buckets_keys_[i] > buckets_keys_[i - 1], "Bucket sizes must be in increasing order.");
    }
#endif
  }

private:
  size_t CalculateBufferSize(size_t request_size) override {
    // binary serch size
    auto it = std::lower_bound(buckets_keys_.begin(), buckets_keys_.end(), request_size);
    if (it == buckets_keys_.end()) {
      return NormalizeBufferSize(request_size);
    } else {
      return *it;
    }
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end() && !it->second.empty()) {
      auto buffer = it->second.back().MoveToCHandle();
      it->second.pop_back();
      return buffer;
    }
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

protected:
  std::unordered_map<size_t, std::vector<wgpu::Buffer>> buckets_;
  std::vector<wgpu::Buffer> pending_buffers_;

private:
  std::unordered_map<size_t, size_t> buckets_limit_;
  std::vector<size_t> buckets_keys_;
};

class BucketCacheManagerUMA : public CacheManagerUMA<BucketCacheManager> {
 public:
  explicit BucketCacheManagerUMA(BufferManagerUMA* buffer_manager)
      : CacheManagerUMA<BucketCacheManager>(buffer_manager) {}
  ~BucketCacheManagerUMA() {
    for (auto& pair : buckets_) {
      for (auto& buffer : pair.second) {
        // Notify the buffer manager that the buffer is released.
        buffer_manager_->OnBufferRelease(buffer.Get());
      }
    }
  }
};

std::ostream& operator<<(std::ostream& os, BufferCacheMode mode) {
  switch (mode) {
    case BufferCacheMode::Disabled:
      os << "Disabled";
      break;
    case BufferCacheMode::LazyRelease:
      os << "LazyRelease";
      break;
    case BufferCacheMode::Simple:
      os << "Simple";
      break;
    case BufferCacheMode::Bucket:
      os << "Bucket";
      break;
    default:
      os << "Unknown(" << static_cast<int>(mode) << ")";
  }
  return os;
}

std::unique_ptr<IBufferCacheManager> BufferManager::CreateBufferCacheManager(BufferCacheMode cache_mode) {
  switch (cache_mode) {
    case BufferCacheMode::Disabled:
      return std::make_unique<DisabledCacheManager>();
    case BufferCacheMode::LazyRelease:
      return std::make_unique<LazyReleaseCacheManager>();
    case BufferCacheMode::Simple:
      return std::make_unique<SimpleCacheManager>();
    case BufferCacheMode::Bucket:
      return std::make_unique<BucketCacheManager>();
    default:
      ORT_NOT_IMPLEMENTED("Unsupported buffer cache mode");
  }
}

void BufferManager::Initialize(BufferCacheMode storage_buffer_cache_mode,
                               BufferCacheMode uniform_buffer_cache_mode,
                               BufferCacheMode query_resolve_buffer_cache_mode) {
  storage_cache_ = CreateBufferCacheManager(storage_buffer_cache_mode);
  uniform_cache_ = CreateBufferCacheManager(uniform_buffer_cache_mode);
  query_resolve_cache_ = CreateBufferCacheManager(query_resolve_buffer_cache_mode);
  default_cache_ = CreateBufferCacheManager(BufferCacheMode::Simple);
}

void BufferManager::Upload(void* src, WGPUBuffer dst, size_t size) {
  auto buffer_size = NormalizeBufferSize(size);

  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapWrite;
  desc.mappedAtCreation = true;

  auto staging_buffer = context_.Device().CreateBuffer(&desc);
  auto mapped_data = staging_buffer.GetMappedRange();
  memcpy(mapped_data, src, size);
  staging_buffer.Unmap();

  auto& command_encoder = context_.GetCommandEncoder();
  context_.EndComputePass();
  command_encoder.CopyBufferToBuffer(staging_buffer, 0, dst, 0, buffer_size);
  pending_staging_buffers_.push_back(staging_buffer);
}

void BufferManager::MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size) {
  ORT_ENFORCE(src != dst, "Source and destination buffers must be different.");

  auto buffer_size = NormalizeBufferSize(size);
  ORT_ENFORCE(buffer_size <= wgpuBufferGetSize(src) && buffer_size <= wgpuBufferGetSize(dst),
              "Source and destination buffers must have enough space for the copy operation. src_size=",
              wgpuBufferGetSize(src), ", dst_size=", wgpuBufferGetSize(dst), ", copy_size=", buffer_size, ".");

  auto& command_encoder = context_.GetCommandEncoder();
  context_.EndComputePass();
  command_encoder.CopyBufferToBuffer(src, 0, dst, 0, buffer_size);
}

WGPUBuffer BufferManager::Create(size_t size, wgpu::BufferUsage usage, bool mapped_at_creation) {
  auto& cache = GetCacheManager(static_cast<WGPUBufferUsage>(usage));
  auto buffer_size = cache.CalculateBufferSize(size);

  auto buffer = cache.TryAcquireCachedBuffer(buffer_size);
  if (buffer) {
    return buffer;
  }

  // cache miss, create a new buffer
  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = usage;
  desc.mappedAtCreation = mapped_at_creation;
  // desc.label = std::to_string(xx++).c_str();
  buffer = context_.Device().CreateBuffer(&desc).MoveToCHandle();

  ORT_ENFORCE(buffer, "Failed to create GPU buffer: size=", buffer_size, ", usage=", uint64_t(usage), ".");

  cache.RegisterBuffer(buffer, size);
  return buffer;
}

WGPUBuffer BufferManager::Create(size_t size, wgpu::BufferUsage usage) { return Create(size, usage, false); }

void BufferManager::Release(WGPUBuffer buffer) {
  GetCacheManager(buffer).ReleaseBuffer(buffer);
}

void BufferManager::Download(WGPUBuffer src, void* dst, size_t size) {
  auto buffer_size = NormalizeBufferSize(size);

  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;

  auto staging_buffer = context_.Device().CreateBuffer(&desc);
  auto& command_encoder = context_.GetCommandEncoder();
  context_.EndComputePass();
  command_encoder.CopyBufferToBuffer(src, 0, staging_buffer, 0, buffer_size);
  context_.Flush();

  // TODO: revise wait in whole project

  ORT_ENFORCE(context_.Wait(staging_buffer.MapAsync(wgpu::MapMode::Read, 0, buffer_size, wgpu::CallbackMode::WaitAnyOnly, [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
    ORT_ENFORCE(status == wgpu::MapAsyncStatus::Success, "Failed to download data from buffer: ", std::string_view{message});
  })) == Status::OK());

  auto mapped_data = staging_buffer.GetConstMappedRange();
  memcpy(dst, mapped_data, size);
}

void BufferManager::RefreshPendingBuffers() {
  pending_staging_buffers_.clear();
  storage_cache_->OnRefresh();
  uniform_cache_->OnRefresh();
  query_resolve_cache_->OnRefresh();
  default_cache_->OnRefresh();
}

IBufferCacheManager& BufferManager::GetCacheManager(WGPUBufferUsage usage) const {
  if (usage & WGPUBufferUsage_Storage) {
    return *storage_cache_;
  } else if (usage & WGPUBufferUsage_Uniform) {
    return *uniform_cache_;
  } else if (usage & WGPUBufferUsage_QueryResolve) {
    return *query_resolve_cache_;
  } else {
    return *default_cache_;
  }
}

IBufferCacheManager& BufferManager::GetCacheManager(WGPUBuffer buffer) const {
  return GetCacheManager(wgpuBufferGetUsage(buffer));
}

std::unique_ptr<IBufferCacheManager> BufferManagerUMA::CreateBufferCacheManager(BufferCacheMode cache_mode) {
  switch (cache_mode) {
    case BufferCacheMode::Disabled:
      return std::make_unique<DisabledCacheManagerUMA>(this);
    case BufferCacheMode::LazyRelease:
      return std::make_unique<LazyReleaseCacheManagerUMA>(this);
    case BufferCacheMode::Simple:
      return std::make_unique<SimpleCacheManagerUMA>(this);
    case BufferCacheMode::Bucket:
      return std::make_unique<BucketCacheManagerUMA>(this);
    default:
      ORT_NOT_IMPLEMENTED("Unsupported buffer cache mode");
  }
}

void BufferManagerUMA::Initialize(BufferCacheMode storage_buffer_cache_mode,
                               BufferCacheMode uniform_buffer_cache_mode,
                               BufferCacheMode query_resolve_buffer_cache_mode) {
  // Use the UMA buffer cache manager for storage buffers.
  storage_cache_ = CreateBufferCacheManager(storage_buffer_cache_mode);

  uniform_cache_ = BufferManager::CreateBufferCacheManager(uniform_buffer_cache_mode);
  query_resolve_cache_ = BufferManager::CreateBufferCacheManager(query_resolve_buffer_cache_mode);
  default_cache_ = BufferManager::CreateBufferCacheManager(BufferCacheMode::Simple);
}

BufferManagerUMA::~BufferManagerUMA() {
  // Ensure the storage buffer cache manager has no pending MapAsync callbacks.
  auto storage_cache_uma = dynamic_cast<IBufferCacheManagerUMA*>(storage_cache_.get());
  if (storage_cache_uma) {
    storage_cache_uma->WaitBufferMapAsync(context_.Device().GetAdapter().GetInstance());
  }
  storage_cache_.reset();
}

WGPUBuffer BufferManagerUMA::Create(size_t size, wgpu::BufferUsage usage) {
  bool mapped_at_creation = false;
  // Only use the extended map usages for storage buffers. For these buffers, we always map them at creation.
  if (usage & wgpu::BufferUsage::Storage) {
    usage |= wgpu::BufferUsage::MapRead | wgpu::BufferUsage::MapWrite;
    mapped_at_creation = true;
  }

  auto buffer = BufferManager::Create(size, usage, mapped_at_creation);

  // Track the UMA buffer.
  if (usage & wgpu::BufferUsage::Storage) {
    uma_buffers_[buffer] = true;
  }

  return buffer;
}

bool BufferManagerUMA::IsUMABuffer(WGPUBuffer buffer) const {
  auto it = uma_buffers_.find(buffer);
  return it != uma_buffers_.end() && it->second;
}

// Update the bookkeeping when a UMA buffer is released.
void BufferManagerUMA::OnBufferRelease(WGPUBuffer buffer) {
  auto it = uma_buffers_.find(buffer);
  if (it != uma_buffers_.end()) {
    uma_buffers_.erase(it);
  }
}

void BufferManagerUMA::Upload(void* src, WGPUBuffer dst, size_t size) {
  if (!IsUMABuffer(dst)) {
    BufferManager::Upload(src, dst, size);
    return;
  }

  wgpu::Buffer buffer = dst;
  auto mapped_data = buffer.GetMappedRange();
  ORT_ENFORCE(mapped_data, "The dst buffer is not mapped for upload.");
  memcpy(mapped_data, src, size);
  buffer.Unmap();
}

void BufferManagerUMA::Download(WGPUBuffer src, void* dst, size_t size) {
  if (!IsUMABuffer(src)) {
    BufferManager::Download(src, dst, size);
    return;
  }

  context_.Flush();

  wgpu::Buffer buffer = src;
  auto buffer_size = NormalizeBufferSize(size);
  ORT_ENFORCE(context_.Wait(buffer.MapAsync(
                  wgpu::MapMode::Read, 0, buffer_size, wgpu::CallbackMode::WaitAnyOnly,
                  [](wgpu::MapAsyncStatus status, wgpu::StringView message) {
                    ORT_ENFORCE(status == wgpu::MapAsyncStatus::Success,
                                "Failed to map dst buffer for download: ", std::string_view{message});
                  })) == Status::OK());

  auto mapped_data = buffer.GetConstMappedRange(0, size);
  ORT_ENFORCE(mapped_data, "The src buffer is not mapped for download.");
  memcpy(dst, mapped_data, size);
  buffer.Unmap();
}

void BufferManagerUMA::MemCpy(WGPUBuffer src, WGPUBuffer dst, size_t size) {
  // UMA buffers are mapped at creation time, so we may need to unmap them before using them by GPU.
  auto EnsureBufferUnmapped = [this](WGPUBuffer buffer) {
    auto map_state = wgpuBufferGetMapState(buffer);
    if (map_state != WGPUBufferMapState_Unmapped && IsUMABuffer(buffer)) {
      wgpuBufferUnmap(buffer);
    }
  };
  EnsureBufferUnmapped(src);
  EnsureBufferUnmapped(dst);

  BufferManager::MemCpy(src, dst, size);
}

void BufferManagerUMA::RefreshPendingBuffers() {
  // Make sure the pending MapAsync callbacks can have a chance to be processed.
  context_.Device().GetAdapter().GetInstance().ProcessEvents();

  BufferManager::RefreshPendingBuffers();
}

std::unique_ptr<BufferManager> BufferManagerFactory::Create(WebGpuContext& context,
                                                            BufferCacheMode storage_buffer_cache_mode,
                                                            BufferCacheMode uniform_buffer_cache_mode,
                                                            BufferCacheMode query_resolve_buffer_cache_mode) {
  std::unique_ptr<BufferManager> buffer_manager;
  if (context.SupportsBufferMapExtendedUsages()) {
    buffer_manager = std::make_unique<BufferManagerUMA>(context);
  } else {
    buffer_manager = std::make_unique<BufferManager>(context);
  }
  buffer_manager->Initialize(storage_buffer_cache_mode, uniform_buffer_cache_mode, query_resolve_buffer_cache_mode);
  return buffer_manager;
}

}  // namespace webgpu
}  // namespace onnxruntime
