// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <deque>

#include "core/providers/webgpu/buffer_manager.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/common/safeint.h"

namespace onnxruntime {
namespace webgpu {

namespace {

constexpr size_t NormalizeBufferSize(size_t size) {
  return (size + 15) / 16 * 16;
}

size_t CalculateBucketBufferSize(size_t request_size) {
  // Four intervals per doubling limit padding while preserving reuse as tensor sizes grow.
  request_size = std::max(size_t{64}, request_size);
  size_t alignment = 16;
  while ((request_size - 1) / 8 >= alignment) {
    alignment *= 2;
  }
  return (SafeInt<size_t>(request_size) + alignment - 1) / alignment * alignment;
}

// WebGPU requires that the copy size in CopyBufferToBuffer must be a multiple of 4 bytes.
constexpr size_t NormalizeCopySize(size_t size) {
  return (size + 3) / 4 * 4;
}

void EnforceBufferUnmapped(WebGpuContext& context, WGPUBuffer buffer) {
  if (context.ValidationMode() > ValidationMode::Basic) {
    ORT_ENFORCE(wgpuBufferGetMapState(buffer) == WGPUBufferMapState_Unmapped, "Buffer is still mapped.");
  }
}

}  // namespace

class DisabledCacheManager : public IBufferCacheManager {
  bool SupportsBufferReuse() const override { return false; }

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

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    // no-op
  }
};

class LazyReleaseCacheManager : public IBufferCacheManager {
  bool SupportsBufferReuse() const override { return false; }

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
    pending_buffers_.emplace_back(buffer);
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    Release();
    pending_buffers_.clear();
  }

 public:
  ~LazyReleaseCacheManager() {
    Release();
  }

 protected:
  void Release() {
    for (auto& buffer : pending_buffers_) {
      wgpuBufferRelease(buffer);
    }
  }

  std::vector<WGPUBuffer> pending_buffers_;
};

class SimpleCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buffers_.find(buffer_size);
    if (it != buffers_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }

    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(buffer);
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    for (auto& buffer : pending_buffers_) {
      buffers_[static_cast<size_t>(wgpuBufferGetSize(buffer))].emplace_back(buffer);
    }
    pending_buffers_.clear();
  }

 public:
  ~SimpleCacheManager() {
    for (auto& buffer : pending_buffers_) {
      wgpuBufferRelease(buffer);
    }
    for (auto& pair : buffers_) {
      for (auto& buffer : pair.second) {
        wgpuBufferRelease(buffer);
      }
    }
  }

 protected:
  std::map<size_t, std::vector<WGPUBuffer>> buffers_;
  std::vector<WGPUBuffer> pending_buffers_;
};

class BucketCacheManager : public IBufferCacheManager {
  // One shared window ends after 16 successful runs across this context's sessions.
  // Spares have no session owner: demand follows the current borrower, regardless
  // of which recording eventually returns the buffer. Partial flushes do not count.
  static constexpr size_t kRunWindow = 16;
  struct Demand {
    size_t unavailable = 0;
    size_t peak = 0;
    size_t window_peak = 0;
    size_t retained_peak = 0;

    size_t RecentPeak(bool active) const {
      return std::max({retained_peak, window_peak, active ? peak : size_t{0}});
    }

    void AdvanceWindow() {
      retained_peak = window_peak;
      window_peak = 0;
    }
  };
  struct Session {
    bool active = false;
    bool completed_in_window = false;
    bool window_update_pending = false;
    bool trim_late_returns = false;
    InlinedHashMap<size_t, Demand> demand;
  };
  struct BorrowedBuffer {
    std::shared_ptr<Session> session;
    size_t capacity;
  };
  using BufferQueue = std::deque<wgpu::Buffer>;

 public:
  size_t CalculateBufferSize(size_t request_size) override {
    return CalculateBucketBufferSize(request_size);
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

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {}

  void OnAcquire(WGPUBuffer buffer, const CommandRecordingState& recording) override {
    auto it = sessions_.find(&recording);
    if (it == sessions_.end()) {
      return;
    }
    auto& session = it->second;
    const auto capacity = static_cast<size_t>(wgpuBufferGetSize(buffer));
    borrowed_.emplace(buffer, BorrowedBuffer{session, capacity});
    auto& demand = session->demand[capacity];
    ++demand.unavailable;
    if (session->active) {
      demand.peak = std::max(demand.peak, demand.unavailable);
    }
  }

  void OnDiscard(WGPUBuffer buffer) override {
    Retire(buffer);
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    auto owned_buffer = wgpu::Buffer::Acquire(buffer);
    const auto capacity = static_cast<size_t>(owned_buffer.GetSize());
    auto borrower = Retire(buffer);
    auto& buffers = buckets_[capacity];
    buffers.push_back(std::move(owned_buffer));
    if (borrower && !borrower->active && borrower->trim_late_returns) {
      size_t keep = 0;
      for (const auto& [recording, session] : sessions_) {
        if (auto it = session->demand.find(capacity); it != session->demand.end()) {
          const auto& demand = it->second;
          const auto peak = demand.RecentPeak(session->active);
          if (peak > demand.unavailable) {
            keep += peak - demand.unavailable;
          }
        }
      }
      TrimBucket(buffers, keep);
    }
  }

  void OnRunStart(const CommandRecordingState& recording) override {
    auto& session = sessions_[&recording];
    if (!session) {
      session = std::make_shared<Session>();
    }
    session->active = true;
    session->trim_late_returns = false;
    for (auto& [capacity, demand] : session->demand) {
      // Include carried-over live inputs, not just allocations made in this run.
      demand.peak = demand.unavailable;
    }
  }

  void OnRunEnd(const CommandRecordingState& recording, bool completed) override {
    auto it = sessions_.find(&recording);
    if (it == sessions_.end()) {
      return;
    }
    auto& session = *it->second;
    // A run crossing a shared boundary completes that session's deferred window.
    // Updating here prevents continuous overlap from retaining an old peak forever,
    // without adding a pool scan between the ordinary 16-run cleanup batches.
    const bool update = completed && session.window_update_pending;
    for (auto& [capacity, demand] : session.demand) {
      demand.window_peak = std::max(demand.window_peak, demand.peak);
      if (update) {
        demand.AdvanceWindow();
      } else if (!completed) {
        // Failure is not evidence that this session's previous demand decreased.
        demand.retained_peak = std::max(demand.retained_peak, demand.peak);
      }
    }
    session.active = false;
    if (!completed) {
      Trim();
    } else {
      session.completed_in_window = !update;
      session.window_update_pending = false;
      if (++runs_since_cleanup_ == kRunWindow) {
        runs_since_cleanup_ = 0;
        Trim(true);
      }
    }
  }

  void OnSessionRelease(const CommandRecordingState& recording) override {
    // Outstanding allocations retain their old attribution, but teardown drops
    // this session's spare demand even if a new session reuses its address.
    if (auto it = sessions_.find(&recording); it != sessions_.end()) {
      it->second->active = false;
      it->second->trim_late_returns = true;
      sessions_.erase(it);
    }
    Trim();
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {}

 private:
  std::shared_ptr<Session> Retire(WGPUBuffer buffer) {
    auto it = borrowed_.find(buffer);
    if (it == borrowed_.end()) {
      return nullptr;
    }
    auto session = std::move(it->second.session);
    --session->demand.at(it->second.capacity).unavailable;
    borrowed_.erase(it);
    return session;
  }

  void Trim(bool advance_window = false) {
    InlinedHashMap<size_t, size_t> spare_demand;
    for (const auto& [recording, session] : sessions_) {
      // A fast session's boundary preserves paused sessions and defers active
      // sessions until their complete run ends. Slow runs can need more spares
      // later, including buffers returned by other sessions. Previously completed
      // peaks are carried into the deferred window; unavailable buffers stay live.
      const bool update = advance_window && session->completed_in_window && !session->active;
      if (advance_window && session->active) {
        session->window_update_pending = true;
      }
      for (auto it = session->demand.begin(); it != session->demand.end();) {
        auto& demand = it->second;
        if (update) {
          demand.AdvanceWindow();
        }
        // Per capacity, sum remaining observed demand over all sessions. Live
        // and pending buffers are unavailable; free buffers can serve any borrower.
        const auto peak = demand.RecentPeak(session->active);
        if (peak > demand.unavailable) {
          spare_demand[it->first] += peak - demand.unavailable;
        }
        if (demand.unavailable == 0 && peak == 0) {
          session->demand.erase(it++);
        } else {
          ++it;
        }
      }
      if (update) {
        session->completed_in_window = false;
        session->window_update_pending = false;
      }
      // Returns associated with this maintenance opportunity check only their
      // size class. A subsequent run starts a new allocation/reuse opportunity.
      session->trim_late_returns = true;
    }
    for (auto it = buckets_.begin(); it != buckets_.end();) {
      auto& buffers = it->second;
      TrimBucket(buffers, spare_demand[it->first]);
      if (buffers.empty()) {
        buckets_.erase(it++);
      } else {
        ++it;
      }
    }
  }

  void TrimBucket(BufferQueue& buffers, size_t keep) {
    // Keep MRU spares for observed demand; release only surplus. Outside-run
    // allocations share the pool without reserving additional spare demand.
    while (buffers.size() > keep) {
      buffers.pop_front();
    }
  }

  size_t runs_since_cleanup_ = 0;
  InlinedHashMap<size_t, BufferQueue> buckets_;
  InlinedHashMap<const CommandRecordingState*, std::shared_ptr<Session>> sessions_;
  InlinedHashMap<WGPUBuffer, BorrowedBuffer> borrowed_;
};

class GraphCacheManager : public IBufferCacheManager {
 public:
  size_t CalculateBufferSize(size_t request_size) override {
    return CalculateBucketBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buckets_.find(buffer_size);
    if (it != buckets_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }
    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    auto buffer_size = static_cast<size_t>(wgpuBufferGetSize(buffer));
    buckets_[buffer_size].emplace_back(buffer);
  }

  void OnRefresh(GraphCaptureState /*graph_capture_state*/) override {
    // Captured addresses must remain valid until the graph is released.
  }

  std::vector<std::pair<size_t, WGPUBuffer>> ExtractCachedBuffers() override {
    std::vector<std::pair<size_t, WGPUBuffer>> result;
    for (auto& pair : buckets_) {
      for (auto& buffer : pair.second) {
        result.emplace_back(pair.first, buffer);
      }
    }
    buckets_.clear();
    return result;
  }

  void AbsorbCachedBuffers(std::vector<std::pair<size_t, WGPUBuffer>>&& buffers) override {
    for (auto& entry : buffers) {
      if (entry.second) {
        ReleaseBuffer(entry.second);
      }
    }
  }

  ~GraphCacheManager() {
    for (auto& pair : buckets_) {
      for (auto& buffer : pair.second) {
        wgpuBufferRelease(buffer);
      }
    }
  }

 private:
  InlinedHashMap<size_t, std::vector<WGPUBuffer>> buckets_;
};

class GraphSimpleCacheManager : public IBufferCacheManager {
  size_t CalculateBufferSize(size_t request_size) override {
    return NormalizeBufferSize(request_size);
  }

  WGPUBuffer TryAcquireCachedBuffer(size_t buffer_size) override {
    auto it = buffers_.find(buffer_size);
    if (it != buffers_.end() && !it->second.empty()) {
      auto buffer = it->second.back();
      it->second.pop_back();
      return buffer;
    }

    return nullptr;
  }

  void RegisterBuffer(WGPUBuffer /*buffer*/, size_t /*request_size*/) override {
    // no-op
  }

  void ReleaseBuffer(WGPUBuffer buffer) override {
    pending_buffers_.emplace_back(buffer);
  }

  void OnRefresh(GraphCaptureState graph_capture_state) override {
    for (auto& buffer : pending_buffers_) {
      if (graph_capture_state == GraphCaptureState::Default) {
        buffers_[static_cast<size_t>(wgpuBufferGetSize(buffer))].emplace_back(buffer);
      } else {
        captured_buffers_.emplace_back(buffer);
      }
    }
    pending_buffers_.clear();
  }

 public:
  ~GraphSimpleCacheManager() {
    for (auto& buffer : pending_buffers_) {
      wgpuBufferRelease(buffer);
    }
    for (auto& pair : buffers_) {
      for (auto& buffer : pair.second) {
        wgpuBufferRelease(buffer);
      }
    }
    for (auto& buffer : captured_buffers_) {
      wgpuBufferRelease(buffer);
    }
  }

  std::vector<std::pair<size_t, WGPUBuffer>> ExtractCachedBuffers() override {
    // Donation is expected after captured commands have been released and any
    // in-flight work has completed; all three containers therefore hold buffers
    // no longer referenced by the device.
    std::vector<std::pair<size_t, WGPUBuffer>> result;
    for (auto& pair : buffers_) {
      for (auto& buffer : pair.second) {
        result.emplace_back(pair.first, buffer);
      }
      pair.second.clear();
    }
    buffers_.clear();
    for (auto& buffer : pending_buffers_) {
      result.emplace_back(static_cast<size_t>(wgpuBufferGetSize(buffer)), buffer);
    }
    pending_buffers_.clear();
    for (auto& buffer : captured_buffers_) {
      result.emplace_back(static_cast<size_t>(wgpuBufferGetSize(buffer)), buffer);
    }
    captured_buffers_.clear();
    return result;
  }

  void AbsorbCachedBuffers(std::vector<std::pair<size_t, WGPUBuffer>>&& buffers) override {
    for (auto& entry : buffers) {
      if (entry.second) {
        buffers_[entry.first].emplace_back(entry.second);
      }
    }
  }

 protected:
  std::map<size_t, std::vector<WGPUBuffer>> buffers_;
  std::vector<WGPUBuffer> pending_buffers_;
  std::vector<WGPUBuffer> captured_buffers_;
};

std::unique_ptr<IBufferCacheManager> CreateBufferCacheManager(BufferCacheMode cache_mode) {
  switch (cache_mode) {
    case BufferCacheMode::Disabled:
      return std::make_unique<DisabledCacheManager>();
    case BufferCacheMode::LazyRelease:
      return std::make_unique<LazyReleaseCacheManager>();
    case BufferCacheMode::Simple:
      return std::make_unique<SimpleCacheManager>();
    case BufferCacheMode::Bucket:
      return std::make_unique<BucketCacheManager>();
    case BufferCacheMode::Graph:
      return std::make_unique<GraphCacheManager>();
    case BufferCacheMode::GraphSimple:
      return std::make_unique<GraphSimpleCacheManager>();
    default:
      ORT_NOT_IMPLEMENTED("Unsupported buffer cache mode");
  }
}

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
    case BufferCacheMode::Graph:
      os << "Graph";
      break;
    case BufferCacheMode::GraphSimple:
      os << "GraphSimple";
      break;
    default:
      os << "Unknown(" << static_cast<int>(mode) << ")";
  }
  return os;
}

BufferManager::BufferManager(WebGpuContext& context, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode, BufferCacheMode default_buffer_cache_mode)
    : context_{context},
      storage_cache_{CreateBufferCacheManager(storage_buffer_cache_mode)},
      uniform_cache_{CreateBufferCacheManager(uniform_buffer_cache_mode)},
      query_resolve_cache_{CreateBufferCacheManager(query_resolve_buffer_cache_mode)},
      default_cache_{CreateBufferCacheManager(default_buffer_cache_mode)} {
}

void BufferManager::Upload(CommandRecordingState& recording, void* src, WGPUBuffer dst, size_t size) const {
  // If the buffer is mapped, we can directly write to it.
  void* mapped_data = wgpuBufferGetMappedRange(dst, 0, WGPU_WHOLE_MAP_SIZE);  // ensure the buffer is mapped
  if (mapped_data) {
    if (size > 0) {
      memcpy(mapped_data, src, size);
    }
    wgpuBufferUnmap(dst);
    return;
  }

  if (size == 0) {
    return;
  }

  // A100/Vulkan measurements improve through 1 MiB but can regress for larger
  // transfers. Keep the cap at the largest consistently beneficial measured size.
  constexpr size_t kMaxQueueUploadSize = 1024 * 1024;
  if (size <= kMaxQueueUploadSize && recording.graph_capture_state == GraphCaptureState::Default && size % 4 == 0) {
    // Queue writes copy host bytes before returning. Submit recorded readers first so the write
    // cannot overtake them; Dawn owns staging lifetimes and can batch subsequent queue work.
    ORT_THROW_IF_ERROR(context_.Flush(*this, recording));
    context_.Device().GetQueue().WriteBuffer(wgpu::Buffer{dst}, 0, src, size);
    return;
  }

  // Otherwise, we need to use a staging buffer to upload data.
  auto copy_size = NormalizeCopySize(size);

  wgpu::BufferDescriptor desc{};
  desc.size = copy_size;
  desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapWrite;
  desc.mappedAtCreation = true;

  auto staging_buffer = context_.Device().CreateBuffer(&desc);
  mapped_data = staging_buffer.GetMappedRange();
  memcpy(mapped_data, src, size);
  // NOTE: When copy_size != size (due to 4-byte alignment requirement of CopyBufferToBuffer),
  // the trailing bytes [size, copy_size) in the staging buffer contain uninitialized data.
  // This dirty data gets copied into the destination buffer and may cause problems.
  // A possible solution is to use CopyBufferToBuffer for the aligned portion and a compute
  // shader to write the non-aligned remainder.
  staging_buffer.Unmap();

  ORT_THROW_IF_ERROR(context_.EncodeDeferredDispatches(recording));
  auto& command_encoder = context_.GetCommandEncoder(recording);
  context_.EndComputePass(recording);
  command_encoder.CopyBufferToBuffer(staging_buffer, 0, dst, 0, copy_size);
  ORT_THROW_IF_ERROR(context_.Flush(*this, recording));
}

void BufferManager::UploadBatch(CommandRecordingState& recording, gsl::span<const UploadInfo> uploads) const {
  // Bound staging memory while amortizing buffer creation and queue submission across small inputs.
  constexpr size_t kBatchBytes = 16 * 1024 * 1024;
  size_t begin = 0;
  while (begin < uploads.size()) {
    size_t end = begin;
    size_t bytes = 0;
    while (end < uploads.size()) {
      const auto aligned_size = NormalizeCopySize(uploads[end].size);
      if (aligned_size > kBatchBytes - bytes) break;
      bytes += aligned_size;
      ++end;
    }
    if (end == begin) {
      const auto& upload = uploads[begin++];
      Upload(recording, const_cast<void*>(upload.source), upload.target, upload.size);
      continue;
    }
    if (bytes != 0) {
      wgpu::BufferDescriptor desc{};
      desc.size = bytes;
      desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::MapWrite;
      desc.mappedAtCreation = true;
      auto staging_buffer = context_.Device().CreateBuffer(&desc);
      auto* mapped = static_cast<uint8_t*>(staging_buffer.GetMappedRange());
      size_t offset = 0;
      for (size_t i = begin; i < end; ++i) {
        const auto& upload = uploads[i];
        const auto aligned_size = NormalizeCopySize(upload.size);
        if (upload.size != 0) {
          memcpy(mapped + offset, upload.source, upload.size);
          memset(mapped + offset + upload.size, 0, aligned_size - upload.size);
        }
        offset += aligned_size;
      }
      staging_buffer.Unmap();
      ORT_THROW_IF_ERROR(context_.EncodeDeferredDispatches(recording));
      context_.EndComputePass(recording);
      auto& encoder = context_.GetCommandEncoder(recording);
      offset = 0;
      for (size_t i = begin; i < end; ++i) {
        const auto& upload = uploads[i];
        const auto aligned_size = NormalizeCopySize(upload.size);
        if (aligned_size != 0) {
          ORT_ENFORCE(aligned_size <= wgpuBufferGetSize(upload.target));
          EnforceBufferUnmapped(context_, upload.target);
          encoder.CopyBufferToBuffer(staging_buffer, offset, upload.target, 0, aligned_size);
        }
        offset += aligned_size;
      }
      ORT_THROW_IF_ERROR(context_.Flush(*this, recording));
    }
    begin = end;
  }
}

void BufferManager::MemCpy(CommandRecordingState& recording, WGPUBuffer src, WGPUBuffer dst, size_t size) const {
  ORT_ENFORCE(src != dst, "Source and destination buffers must be different.");
  EnforceBufferUnmapped(context_, src);
  EnforceBufferUnmapped(context_, dst);

  auto copy_size = NormalizeCopySize(size);
  auto src_size = static_cast<size_t>(wgpuBufferGetSize(src));
  auto dst_size = static_cast<size_t>(wgpuBufferGetSize(dst));
  ORT_ENFORCE(copy_size <= src_size && copy_size <= dst_size,
              "Source and destination buffers must have enough space for the copy operation. src_size=",
              src_size, ", dst_size=", dst_size, ", copy_size=", copy_size, ".");

  context_.CopyBuffer(src, dst, copy_size, recording);
}

WGPUBuffer BufferManager::Create(CommandRecordingState& recording, size_t size, wgpu::BufferUsage usage,
                                 bool initialize_to_zero,
                                 bool submit_zero_initialize) const {
  size_t buffer_size;
  WGPUBuffer buffer;
  {
    std::lock_guard<std::mutex> lock{mutex_};
    auto& cache = GetCacheManager(usage);
    buffer_size = cache.CalculateBufferSize(size);
    // Rounding must not make an otherwise valid allocation exceed the device limit.
    if (buffer_size > context_.DeviceLimits().maxBufferSize && size <= context_.DeviceLimits().maxBufferSize) {
      // Preserve padding when possible, including when the limit is not 16-byte aligned.
      buffer_size = std::min(NormalizeBufferSize(size), static_cast<size_t>(context_.DeviceLimits().maxBufferSize));
    }
    buffer = cache.TryAcquireCachedBuffer(buffer_size);
    if (buffer) {
      cache.OnAcquire(buffer, recording);
    }
  }
  if (buffer) {
    if (initialize_to_zero) {
      // Streamless allocations outside Run submit their own call-local clear. AllocOnStream
      // and built-in allocations during Run defer the clear on the Session timeline.
      auto buffer_guard = wgpu::Buffer::Acquire(buffer);
      bool initialized = false;
      auto discard_on_error = gsl::finally([&] {
        if (!initialized) {
          std::lock_guard<std::mutex> lock{mutex_};
          GetCacheManager(usage).OnDiscard(buffer);
        }
      });
      ORT_THROW_IF_ERROR(context_.EncodeDeferredDispatches(recording));
      context_.EndComputePass(recording);
      context_.GetCommandEncoder(recording).ClearBuffer(buffer, 0, buffer_size);
      if (submit_zero_initialize) {
        ORT_THROW_IF_ERROR(context_.Flush(*this, recording));
      }
      initialized = true;
      return buffer_guard.MoveToCHandle();
    }
    return buffer;
  }

  // cache miss, create a new buffer
  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = usage;
  if (usage & wgpu::BufferUsage::MapWrite) {
    desc.mappedAtCreation = true;  // ensure the buffer is mapped for writing at creation
  }
  buffer = context_.Device().CreateBuffer(&desc).MoveToCHandle();

  ORT_ENFORCE(buffer, "Failed to create GPU buffer: size=", buffer_size, ", usage=", uint64_t(usage), ".");

  {
    std::lock_guard<std::mutex> lock{mutex_};
    auto& cache = GetCacheManager(usage);
    cache.RegisterBuffer(buffer, size);
    cache.OnAcquire(buffer, recording);
  }
  return buffer;
}

bool BufferManager::SupportsUMA() const {
#if !defined(__wasm__)
  // Check if the device supports the BufferMapExtendedUsages feature
  return context_.DeviceHasFeature(wgpu::FeatureName::BufferMapExtendedUsages);
#else
  return false;
#endif  // !defined(__wasm__)
}

void BufferManager::Release(WGPUBuffer buffer, const CommandRecordingState* recording) const {
  EnforceBufferUnmapped(context_, buffer);
  std::lock_guard<std::mutex> lock{mutex_};
  auto& cache = GetCacheManager(buffer);
  // Keep reusable buffers out of the cache while this recording has unsubmitted work.
  // Graph caches also need this protection from streamless allocations.
  // Relaxed ordering suffices: recording a buffer's use (and setting the flag) must happen-before
  // its release, on the same thread or through a synchronized ownership handoff. Only independent
  // buffer releases may race with recording. The flag publishes no command data, and mutex_ orders
  // releases against clearing it after submission or abandonment.
  if (cache.SupportsBufferReuse() && recording != nullptr &&
      recording->has_unsubmitted_work.load(std::memory_order_relaxed)) {
    pending_buffers_[recording].emplace_back(wgpu::Buffer::Acquire(buffer));
    return;
  }

  cache.ReleaseBuffer(buffer);
}

void BufferManager::Download(CommandRecordingState& recording, WGPUBuffer src, void* dst, size_t size) const {
  // Encode pending deferred dispatches before recording the readback; the flush below submits both
  // in order.
  ORT_THROW_IF_ERROR(context_.EncodeDeferredDispatches(recording));

  EnforceBufferUnmapped(context_, src);
  auto buffer_size = NormalizeBufferSize(size);

  wgpu::BufferDescriptor desc{};
  desc.size = buffer_size;
  desc.usage = wgpu::BufferUsage::CopyDst | wgpu::BufferUsage::MapRead;

  auto staging_buffer = context_.Device().CreateBuffer(&desc);
  auto& command_encoder = context_.GetCommandEncoder(recording);
  context_.EndComputePass(recording);
  command_encoder.CopyBufferToBuffer(src, 0, staging_buffer, 0, buffer_size);
  ORT_THROW_IF_ERROR(context_.Flush(*this, recording));

  // TODO: revise wait in whole project

  struct MapAsyncResult {
    wgpu::MapAsyncStatus status{};
    std::string message{};
  } map_async_result;

  ORT_THROW_IF_ERROR(context_.Wait(
      staging_buffer.MapAsync(
          wgpu::MapMode::Read, 0, buffer_size, wgpu::CallbackMode::WaitAnyOnly,
          // Note: Don't throw from a Dawn callback.
          [](wgpu::MapAsyncStatus status, wgpu::StringView message,
             MapAsyncResult* result) noexcept {
            result->status = status;
            if (auto message_sv = static_cast<std::string_view>(message);
                !message_sv.empty()) {
              result->message = std::string{message_sv};
            }
          },
          &map_async_result)));

  ORT_ENFORCE(map_async_result.status == wgpu::MapAsyncStatus::Success,
              "Failed to download data from buffer. wgpu::MapAsyncStatus value: ",
              static_cast<int>(map_async_result.status), ", message: ", map_async_result.message);

  auto mapped_data = staging_buffer.GetConstMappedRange();
  memcpy(dst, mapped_data, size);
  staging_buffer.Unmap();
}

void BufferManager::RefreshPendingBuffers(CommandRecordingState& recording, GraphCaptureState graph_capture_state) const {
  std::lock_guard<std::mutex> lock{mutex_};
  // Serialize the transition to idle with Free, so a concurrent release cannot miss this refresh.
  recording.has_unsubmitted_work.store(false, std::memory_order_relaxed);
  if (auto it = pending_buffers_.find(&recording); it != pending_buffers_.end()) {
    for (auto& buffer : it->second) {
      GetCacheManager(buffer.Get()).ReleaseBuffer(buffer.MoveToCHandle());
    }
    pending_buffers_.erase(it);
  }

  if (graph_capture_state == GraphCaptureState::Replaying) {
    return;
  }

  storage_cache_->OnRefresh(graph_capture_state);
  uniform_cache_->OnRefresh(graph_capture_state);
  query_resolve_cache_->OnRefresh(graph_capture_state);
  default_cache_->OnRefresh(graph_capture_state);
}

void BufferManager::DiscardPendingBuffers(CommandRecordingState& recording) const {
  std::lock_guard<std::mutex> lock{mutex_};
  recording.has_unsubmitted_work.store(false, std::memory_order_relaxed);
  if (auto it = pending_buffers_.find(&recording); it != pending_buffers_.end()) {
    for (auto& buffer : it->second) {
      GetCacheManager(buffer.Get()).OnDiscard(buffer.Get());
    }
    pending_buffers_.erase(it);
  }
}

void BufferManager::OnRunStart(const CommandRecordingState& recording) const {
  std::lock_guard<std::mutex> lock{mutex_};
  for (auto* cache : {storage_cache_.get(), uniform_cache_.get(), query_resolve_cache_.get(), default_cache_.get()}) {
    cache->OnRunStart(recording);
  }
}

void BufferManager::OnRunEnd(const CommandRecordingState& recording, bool completed) const {
  std::lock_guard<std::mutex> lock{mutex_};
  for (auto* cache : {storage_cache_.get(), uniform_cache_.get(), query_resolve_cache_.get(), default_cache_.get()}) {
    cache->OnRunEnd(recording, completed);
  }
}

void BufferManager::OnSessionRelease(const CommandRecordingState& recording) const {
  std::lock_guard<std::mutex> lock{mutex_};
  for (auto* cache : {storage_cache_.get(), uniform_cache_.get(), query_resolve_cache_.get(), default_cache_.get()}) {
    cache->OnSessionRelease(recording);
  }
}

std::vector<std::pair<size_t, WGPUBuffer>> BufferManager::ExtractCachedBuffers(wgpu::BufferUsage usage) {
  std::lock_guard<std::mutex> lock{mutex_};
  return GetCacheManager(usage).ExtractCachedBuffers();
}

void BufferManager::AbsorbCachedBuffers(
    wgpu::BufferUsage usage,
    std::vector<std::pair<size_t, WGPUBuffer>>&& buffers) {
  std::lock_guard<std::mutex> lock{mutex_};
  GetCacheManager(usage).AbsorbCachedBuffers(std::move(buffers));
}

IBufferCacheManager& BufferManager::GetCacheManager(wgpu::BufferUsage usage) const {
  if (usage & wgpu::BufferUsage::Storage) {
    return *storage_cache_;
  } else if (usage & wgpu::BufferUsage::Uniform) {
    return *uniform_cache_;
  } else if (usage & wgpu::BufferUsage::QueryResolve) {
    return *query_resolve_cache_;
  } else {
    return *default_cache_;
  }
}

IBufferCacheManager& BufferManager::GetCacheManager(WGPUBuffer buffer) const {
  auto usage = static_cast<wgpu::BufferUsage>(wgpuBufferGetUsage(buffer));
  return GetCacheManager(usage);
}

std::unique_ptr<BufferManager> BufferManagerFactory::Create(WebGpuContext& context, BufferCacheMode storage_buffer_cache_mode, BufferCacheMode uniform_buffer_cache_mode, BufferCacheMode query_resolve_buffer_cache_mode, BufferCacheMode default_buffer_cache_mode) {
  return std::make_unique<BufferManager>(context, storage_buffer_cache_mode, uniform_buffer_cache_mode, query_resolve_buffer_cache_mode, default_buffer_cache_mode);
}

}  // namespace webgpu
}  // namespace onnxruntime
