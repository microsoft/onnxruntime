// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/graph/constants.h"
#include "opencl_allocator.h"
#include "opencl_utils.h"

#include <iostream>

namespace onnxruntime {
namespace opencl {

// TODO: making them configurible
#define BUFFER_CACHING_POLICY 2
#define BUFFER_ALLOWED_WASTING 0.75
#define IMAGE2D_CACHING_POLICY 4
#define IMAGE2D_ALLOWED_WASTING 0.9

struct BufferCreator {
  static void* Create(cl_context ctx, size_t size) {
    ZoneScopedN("BufferCreator::Create");
    cl_int err{};
    auto* ptr = clCreateBuffer(ctx, CL_MEM_READ_WRITE, size, nullptr, &err);
    ORT_THROW_IF_CL_ERROR(err);
    VLOGF_DEFAULT(V_ALLOC, "Allocated Buffer(%p){size=%zu}", ptr, BufferCreator::GetSizeCost(size));
    TracyAlloc(ptr, size);
    return ptr;
  }

  static void Destroy(void* ptr) {
    ZoneScopedN("BufferCreator::Destroy");
    TracyFree(ptr);
    ORT_THROW_IF_CL_ERROR(clReleaseMemObject(static_cast<cl_mem>(ptr)));
  }

  static size_t GetSizeCost(size_t size) {
    return size;
  }

  static bool InfoMatch(size_t existing, size_t requested) {
    VLOGF_DEFAULT(60, "requested: %zu bytes, existing %zu", requested, existing);
    return requested <= existing && existing - requested <= static_cast<size_t>(BUFFER_ALLOWED_WASTING * existing);
  }
};

struct Image2DCreator {
  static void* Create(cl_context ctx, const Image2DDesc& desc) {
    ZoneScopedN("Image2DCreator::Create");
    cl_int err{};
    cl_image_format image_format;
    image_format.image_channel_data_type = desc.DType();
    image_format.image_channel_order = CL_RGBA;
    cl_image_desc image_desc;
    {
      image_desc.image_type = CL_MEM_OBJECT_IMAGE2D;
      image_desc.image_width = desc.UWidth();
      image_desc.image_height = desc.UHeight();
      image_desc.image_depth = 0;        // unused
      image_desc.image_array_size = 0;   // unused
      image_desc.image_row_pitch = 0;    // must be 0 if host_ptr is nullptr
      image_desc.image_slice_pitch = 0;  // must be 0 if host_ptr is nullptr
      image_desc.num_mip_levels = 0;     // must be 0
      image_desc.num_samples = 0;        // must be 0
      image_desc.buffer = nullptr;
    }
    auto* ptr = clCreateImage(ctx, CL_MEM_READ_WRITE, &image_format, &image_desc, nullptr, &err);
    ORT_THROW_IF_CL_ERROR(err);
    TracyAlloc(ptr, Image2DCreator::GetSizeCost(desc));
    VLOGF_DEFAULT(V_ALLOC, "Allocated Image2D(%p){w=%ld, h=%ld})", ptr, desc.Width(), desc.Height());
    return ptr;
  }

  static void Destroy(void* ptr) {
    ZoneScopedN("Image2DCreator::Destroy");
    TracyFree(ptr);
    ORT_THROW_IF_CL_ERROR(clReleaseMemObject(static_cast<cl_mem>(ptr)));
  }

  static size_t GetSizeCost(const Image2DDesc& desc) {
    size_t dtype_size = 0;
    switch (desc.DType()) {
      // NOTE: FpAuto is not reachable here, it should have been resolved in Alloc
      case Image2DDesc::FpAuto:
        dtype_size = 0;
        break;
      case Image2DDesc::Fp32:
        dtype_size = 4;
        break;
      case Image2DDesc::Fp16:
        dtype_size = 2;
        break;
    }
    return desc.UWidth() * desc.UHeight() * 4 * dtype_size;
  }

  static bool InfoMatch(const Image2DDesc& existing, const Image2DDesc& requested) {
    VLOGF_DEFAULT(60, "requested: %zux%zu, existing: %zux%zu", requested.Width(), requested.Height(), existing.Width(), existing.Height());
    return existing.Height() >= requested.Height() &&
               existing.Width() >= requested.Width() &&
               existing.DType() == existing.DType() &&
               [&]() -> bool {
      auto existing_area = existing.Height() * existing.Height();
      auto requested_area = requested.Height() * requested.Width();
      auto allowed_wasting = static_cast<int64_t>(IMAGE2D_ALLOWED_WASTING * existing_area);
      auto wasted = existing_area - requested_area;
      return wasted <= allowed_wasting;
    }();
  }
};

template <typename InfoT, typename CreatorT>
class CacheNone : public CachingPolicy<InfoT, CreatorT> {
  using InfoType = InfoT;
  using CreatorType = CreatorT;

 public:
  ~CacheNone() override {}

  void* CreateOrGetFromCache(cl_context ctx, InfoType info) override final {
    return CreatorType::Create(ctx, info);
  }

  void DestroyOrReturnToCache(void* ptr) override final {
    CreatorType::Destroy(ptr);
  }

  void EvictAllCache() override final {
    /* no cache, cache eviction is not needed. */
  }
};

template <typename InfoT, typename CreatorT>
class CacheAll : public CachingPolicy<InfoT, CreatorT> {
  using InfoType = InfoT;
  using CreatorType = CreatorT;

 public:
  ~CacheAll() override { EvictAllCache(); }

  void* CreateOrGetFromCache(cl_context ctx, InfoType info) override final {
    auto it = cache_.find(info);

    if (it == cache_.end() || it->second.empty()) {
      void* ptr = CreatorType::Create(ctx, info);
      meta_.emplace(ptr, info);
      return ptr;
    }

    auto* ptr = it->second.front();
    VLOGF_DEFAULT(V_ALLOC, "Reused %p", ptr);
    it->second.pop_front();
    if (it->second.empty()) {
      cache_.erase(it);
    }
    return ptr;
  }

  void DestroyOrReturnToCache(void* ptr) override final {
    auto info = meta_.at(ptr);
    auto it = cache_.find(info);
    if (it == cache_.end()) {
      it = cache_.insert({info, {}}).first;
    }
    it->second.push_front(ptr);
  }

  void EvictAllCache() override final {
    for (auto& [ptr, _] : meta_) {
      ORT_UNUSED_PARAMETER(_);
      CreatorType::Destroy(ptr);
    }
    meta_.clear();
    cache_.clear();
  }

 private:
  std::unordered_map<void*, InfoType> meta_;
  std::unordered_map<InfoType, std::list<void*>> cache_;
};

template <typename InfoT, typename CreatorT>
class CacheLRU : public CachingPolicy<InfoT, CreatorT> {
  using InfoType = InfoT;
  using CreatorType = CreatorT;

 public:
  CacheLRU(uint8_t size_limit) : lru_limit_{size_limit} {}
  ~CacheLRU() override { EvictAllCache(); }

  void* CreateOrGetFromCache(cl_context ctx, InfoType info) override final {
    auto best_it = lru_.end();
    auto best_cost = std::numeric_limits<size_t>::max();
    for (auto it = lru_.begin(); it != lru_.end(); it++) {
      auto cost = CreatorType::GetSizeCost(it->second);
      if (cost < best_cost && CreatorType::InfoMatch(it->second, info)) {
        best_it = it;
        best_cost = cost;
      }
    }
    if (best_it != lru_.end()) {
      ZoneScopedN("LRU Hit");
      lru_.erase(best_it);
      return best_it->first;
    }

    ZoneScopedN("LRU Miss");
    auto ptr = CreatorType::Create(ctx, info);
    meta_.emplace(ptr, info);
    return ptr;
  }

  void DestroyOrReturnToCache(void* ptr) override final {
    auto info = meta_.at(ptr);
    lru_.push_front({ptr, info});
    if (lru_.size() > lru_limit_) {
      const auto& it = lru_.back();
      meta_.erase(it.first);
      CreatorType::Destroy(it.first);
      lru_.pop_back();
    }
  }

  void EvictAllCache() override final {
    for (auto& [ptr, _] : lru_) {
      ORT_UNUSED_PARAMETER(_);
      CreatorType::Destroy(ptr);
    }
    lru_.clear();
    meta_.clear();
  }

 private:
  std::unordered_map<void*, InfoType> meta_;

  // selection is based on InfoMatch(elem_in_lru_, requested) so we need to scan the whole list.
  // Since we want to maintain a small LRU cache size generally, we don't need to accelerate the scan process.
  // The cache entry is refreshed on IAllocator::Free(ptr)
  std::list<std::pair<void*, InfoType>> lru_;
  uint8_t lru_limit_;
};

OpenCLBufferAllocator::OpenCLBufferAllocator(cl_context ctx)
    : IAllocator(
          OrtMemoryInfo(
              BufferAllocatorName,
              OrtAllocatorType::OrtDeviceAllocator,
              OrtDevice(OrtDevice::GPU, CLMemType::OPENCL_BUFFER, /*device_id_=*/0),
              /*id_*/ 0,
              /* We blindly cast an integer CLMemType::OPENCL_BUFFER to a C style enum OrtMemType here. Because we
                 don't want to extend the OrtMemType enum at the moment, as it is in public interface header.
                 We manage allocator fully by at EP level so it does not go through AllocatorManager, thus we don't
                 need to worry about the magic value collide with existing value. */
              /*mem_type_=*/static_cast<OrtMemType>(CLMemType::OPENCL_BUFFER))),
      ctx_(ctx) {
  if (BUFFER_CACHING_POLICY > 0) {
    caching_ = std::make_unique<CacheLRU<size_t, BufferCreator>>(BUFFER_CACHING_POLICY);
  } else if (BUFFER_CACHING_POLICY == 0) {
    caching_ = std::make_unique<CacheAll<size_t, BufferCreator>>();
  } else {
    caching_ = std::make_unique<CacheNone<size_t, BufferCreator>>();
  }
}

void* OpenCLBufferAllocator::Alloc(size_t size) {
  ZoneScopedN("OpenCLBufferAllocator::Alloc");
  return caching_->CreateOrGetFromCache(ctx_, size);
}

void OpenCLBufferAllocator::Free(void* ptr) {
  ZoneScopedN("OpenCLBufferAllocator::Free");
  return caching_->DestroyOrReturnToCache(ptr);
}

OpenCLImage2DAllocator::OpenCLImage2DAllocator(cl_context ctx, bool use_fp16)
    : IAllocator(OrtMemoryInfo(Image2DAllocatorName, OrtAllocatorType::OrtDeviceAllocator,
                               OrtDevice(OrtDevice::GPU, CLMemType::OPENCL_IMAGE_2D, /*device_id_=*/0))),
      ctx_(ctx),
      use_fp16_{use_fp16} {
  if (IMAGE2D_CACHING_POLICY > 0) {
    caching_ = std::make_unique<CacheLRU<Image2DDesc, Image2DCreator>>(IMAGE2D_CACHING_POLICY);
  } else if (IMAGE2D_CACHING_POLICY == 0) {
    caching_ = std::make_unique<CacheAll<Image2DDesc, Image2DCreator>>();
  } else {
    caching_ = std::make_unique<CacheNone<Image2DDesc, Image2DCreator>>();
  }
}

void* OpenCLImage2DAllocator::Alloc(size_t) {
  // not supported
  return nullptr;
}

void* OpenCLImage2DAllocator::Alloc(const TensorShape& shape) {
  auto desc = Image2DDesc::PackFromTensor(shape);
  return Alloc(desc);
}

void* OpenCLImage2DAllocator::Alloc(Image2DDesc desc) {
  ZoneScopedN("OpenCLImage2DAllocator::Alloc");
  if (desc.DType() == Image2DDesc::FpAuto) {
    desc.DType() = use_fp16_ ? Image2DDesc::Fp16 : Image2DDesc::Fp32;
  }

  return caching_->CreateOrGetFromCache(ctx_, desc);
}

void OpenCLImage2DAllocator::Free(void* ptr) {
  ZoneScopedN("OpenCLImage2DAllocator::Free");
  caching_->DestroyOrReturnToCache(ptr);
}

}  // namespace opencl
}  // namespace onnxruntime
