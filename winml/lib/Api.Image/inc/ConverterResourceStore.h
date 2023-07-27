// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <mutex>
#include <winrt/windows.media.h>
#include "VideoFrameToTensorConverter.h"
#include "TensorToVideoFrameConverter.h"

namespace _winml {

// Forward Declare
class ConverterResourceStore;

struct ConverterResourceDescription {
  DWORD pixel_format;
  int width;
  int height;
  LUID luid;

  bool operator==(_In_ ConverterResourceDescription& desc) {
    // Converter resources DON'T match if
    // 1) the resources have different dimensions
    // 2) the resources are on different devices
    // 3) the resources have different pixel formats
    if (desc.width != width || desc.height != height || desc.luid.HighPart != luid.HighPart ||
            desc.luid.LowPart != luid.LowPart || desc.pixel_format != pixel_format) {
      return false;
    }

    return true;
  }
};

class ConverterResources : public std::enable_shared_from_this<ConverterResources> {
  using Pool = std::weak_ptr<ConverterResourceStore>;

 public:
  template <typename... TArgs>
  static std::shared_ptr<ConverterResources> Create(Pool pool, ConverterResourceDescription& descriptor) {
    return std::make_shared<ConverterResources>(pool, descriptor);
  }

  ConverterResources(Pool& pool, ConverterResourceDescription& descriptor);

  void ReturnToCache();

 public:
  ConverterResourceDescription Descriptor;

  std::unique_ptr<_winml::VideoFrameToTensorConverter> Tensorizer;
  std::unique_ptr<_winml::TensorToVideoFrameConverter> Detensorizer;

 private:
  Pool m_pool;
};

// This class retains  tensorization and detensorization
// resources in a store, and evicts the oldest resource object
// when the size of the pool is maxed out. Objects in the pool
// can be reused for caching purposes to enhance performance during
// tensorization.
class ConverterResourceStore : public std::enable_shared_from_this<ConverterResourceStore> {
  struct PoolObject {
    std::shared_ptr<ConverterResources> Resource;
    uint64_t StoreId;
  };

 public:
  template <typename... TArgs>
  static std::shared_ptr<ConverterResourceStore> Create(TArgs&&... args) {
    return std::make_shared<ConverterResourceStore>(std::forward<TArgs>(args)...);
  }

  ConverterResourceStore(size_t nCacheSize);

  std::shared_ptr<ConverterResources> Fetch(ConverterResourceDescription& descriptor);
  void Store(std::shared_ptr<ConverterResources> object);

 private:
  std::shared_ptr<ConverterResources> FetchAndRemoveObject(ConverterResourceDescription& desc);
  void EvictOldestPoolObject();

 private:
  std::vector<PoolObject> m_objects;
  size_t m_cacheSize;
  std::mutex m_mutex;
  uint64_t storeId = 0;
};

class PoolObjectWrapper {
 public:
  template <typename... TArgs>
  static std::shared_ptr<PoolObjectWrapper> Create(TArgs&&... args) {
    return std::make_shared<PoolObjectWrapper>(std::forward<TArgs>(args)...);
  }

  explicit PoolObjectWrapper(std::shared_ptr<ConverterResources>&& resources) : m_resources(resources) {}

  ~PoolObjectWrapper() {
    if (m_resources) {
      m_resources->ReturnToCache();
    }
  }

  std::shared_ptr<ConverterResources> Get() { return m_resources; }

 private:
  std::shared_ptr<ConverterResources> m_resources;
};

}  // namespace _winml
